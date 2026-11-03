"""Persist Vista4D/SAM3 per-keyword instance masks without claiming identity."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _value(instance: Any, name: str, default: Any = None) -> Any:
    if isinstance(instance, Mapping):
        return instance.get(name, default)
    return getattr(instance, name, default)


def _mask_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    mask = np.asarray(value)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"SAM3 instance mask must be 2D after squeeze, got {mask.shape}")
    return mask.astype(bool)


def _slug(keyword: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", keyword.strip().lower()).strip("_")
    return value or "entity"


def _field_item(value: Any, index: int, count: int) -> Any:
    if isinstance(value, (str, bytes)) or value is None:
        return value
    try:
        return value[index] if len(value) == count else value
    except (TypeError, IndexError):
        return value


def _instances_in_frame(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, Mapping):
        return tuple(value or ())
    nested = value.get("instances") or value.get("objects")
    if nested is not None:
        return tuple(nested)
    masks = value.get("mask", value.get("masks"))
    keywords = value.get("keyword", value.get("keywords"))
    ids = value.get("id", value.get("ids"))
    count = 1
    for candidate in (ids, keywords):
        if not isinstance(candidate, (str, bytes)):
            try:
                count = max(count, len(candidate))
            except TypeError:
                pass
    try:
        if np.asarray(masks).ndim >= 3:
            count = max(count, len(masks))
    except (TypeError, ValueError):
        pass
    if count == 1:
        return (value,)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "id": _field_item(ids, index, count),
                "keyword": _field_item(keywords, index, count),
                "score": _field_item(value.get("score", value.get("scores")), index, count),
                "box_xyxy": _field_item(
                    value.get("box_xyxy", value.get("boxes")), index, count
                ),
                "mask": _field_item(masks, index, count),
            }
        )
    return tuple(rows)


def _frame_items(seg_frames: Any) -> list[tuple[int, Sequence[Any]]]:
    if isinstance(seg_frames, Mapping):
        raw_frames = seg_frames.get("frames")
        if raw_frames is not None:
            seg_frames = raw_frames
        else:
            rows = sorted(seg_frames.items(), key=lambda item: int(item[0]))
            return [
                (int(index), _instances_in_frame(instances))
                for index, instances in rows
            ]
    values = list(seg_frames)
    if values and all(
        isinstance(item, Mapping)
        and ("frame_index" in item or "frame_idx" in item)
        and ("mask" in item or "masks" in item)
        for item in values
    ):
        grouped: dict[int, list[Any]] = {}
        for item in values:
            frame_index = int(item.get("frame_index", item.get("frame_idx")))
            grouped.setdefault(frame_index, []).extend(_instances_in_frame(item))
        return sorted(grouped.items())
    return [(index, _instances_in_frame(instances)) for index, instances in enumerate(values)]


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / max(1, union))


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically publish the audit manifest; real I/O errors stay fatal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".entity_masks.json.candidate-",
            delete=False,
        ) as candidate:
            json.dump(manifest, candidate, ensure_ascii=False, indent=2)
            candidate.write("\n")
            candidate.flush()
            os.fsync(candidate.fileno())
            candidate_path = Path(candidate.name)
        os.replace(candidate_path, output_dir / "entity_masks.json")
        candidate_path = None
    finally:
        if candidate_path is not None:
            candidate_path.unlink(missing_ok=True)


def export_entity_masks(
    seg_frames: Any,
    output_dir: Path,
    *,
    score_threshold: float = 0.6,
    frame_shape: tuple[int, int] | None = None,
    min_detection_ratio: float = 0.8,
    min_unique_ratio: float = 0.8,
    max_ambiguous_ratio: float = 0.1,
    min_median_iou: float = 0.3,
) -> dict[str, Any]:
    """Save the best high-score mask per keyword and an unverified audit.

    ``seg_frames`` is the in-memory result already returned by
    ``run_sam3_video``. Each frame contains instances with ``keyword``,
    ``score``, ``mask`` and optionally ``box``. Multiple high-score instances
    for one keyword are recorded as ambiguity, even though the best mask is
    retained for inspection.

    This function always writes ``identity_verified=false``. Stability makes
    a track a *candidate* only; identity needs a reference-image, human, or
    model confirmation in a separate step.
    """
    frames = _frame_items(seg_frames)
    export_instance_tracks(frames, output_dir, score_threshold=score_threshold,
                           frame_shape=frame_shape)
    collected: dict[str, dict[int, list[tuple[float, np.ndarray]]]] = {}
    image_shape: tuple[int, int] | None = None
    for frame_index, instances in frames:
        for instance in instances:
            keyword = str(
                _value(instance, "keyword")
                or _value(instance, "label")
                or _value(instance, "text")
                or ""
            ).strip()
            if not keyword:
                continue
            score = float(_value(instance, "score", 0.0) or 0.0)
            if score < float(score_threshold):
                continue
            mask = _mask_array(_value(instance, "mask"))
            if image_shape is None:
                image_shape = mask.shape
            if mask.shape != image_shape:
                raise ValueError(
                    f"inconsistent SAM3 mask shapes: {mask.shape} versus {image_shape}"
                )
            collected.setdefault(keyword, {}).setdefault(frame_index, []).append((score, mask))
    if image_shape is None or not collected:
        manifest = {
            "schema_version": 1,
            "source": "vista4d_run_sam3_video_seg_frames",
            "status": "no_verified_entity_masks",
            "identity_verified": False,
            "reason": "seg_frames contains no high-score keyword masks",
            "score_threshold": float(score_threshold),
            "frame_count": len(frames),
            "identity_policy": (
                "absence of entity candidates is fail-soft for L0 geometry only; "
                "entity-aware anchors must reject because no identity is verified"
            ),
            "entities": {},
        }
        _write_manifest(output_dir, manifest)
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_root = output_dir / "dynamic_mask_by_entity"
    mask_root.mkdir(parents=True, exist_ok=True)
    frame_indices = [index for index, _instances in frames]
    frame_count = len(frame_indices)
    entities: dict[str, dict[str, Any]] = {}
    used_slugs: set[str] = set()
    for keyword, by_frame in sorted(collected.items()):
        slug = _slug(keyword)
        if slug in used_slugs:
            suffix = 2
            while f"{slug}_{suffix}" in used_slugs:
                suffix += 1
            slug = f"{slug}_{suffix}"
        used_slugs.add(slug)
        entity_dir = mask_root / slug
        entity_dir.mkdir(parents=True, exist_ok=True)
        chosen: dict[int, tuple[float, np.ndarray]] = {}
        ambiguous_frames: list[int] = []
        for frame_index in frame_indices:
            candidates = sorted(
                by_frame.get(frame_index, ()),
                key=lambda item: item[0],
                reverse=True,
            )
            if candidates:
                chosen[frame_index] = candidates[0]
            if len(candidates) > 1:
                ambiguous_frames.append(frame_index)
            mask = candidates[0][1] if candidates else np.zeros(image_shape, dtype=bool)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                entity_dir / f"{frame_index:06d}.png"
            )

        observed = sorted(chosen)
        scores = [chosen[index][0] for index in observed]
        adjacent_ious = [
            _mask_iou(chosen[left][1], chosen[right][1])
            for left, right in zip(observed, observed[1:])
            if right == left + 1
        ]
        detection_ratio = len(observed) / frame_count
        unique_ratio = (
            sum(len(by_frame.get(index, ())) == 1 for index in frame_indices)
            / frame_count
        )
        ambiguous_ratio = len(ambiguous_frames) / frame_count
        median_iou = float(np.median(adjacent_ious)) if adjacent_ious else 0.0
        candidate = bool(
            detection_ratio >= float(min_detection_ratio)
            and unique_ratio >= float(min_unique_ratio)
            and ambiguous_ratio <= float(max_ambiguous_ratio)
            and median_iou >= float(min_median_iou)
        )
        entities[keyword] = {
            "mask_dir": str(entity_dir.relative_to(output_dir)),
            "identity_verified": False,
            "candidate": candidate,
            "status": "candidate_unverified" if candidate else "ambiguous_or_unstable",
            "confidence": float(np.median(scores)) if scores else 0.0,
            "score_threshold": float(score_threshold),
            "frame_count": frame_count,
            "detected_frame_count": len(observed),
            "detection_ratio": float(detection_ratio),
            "unique_instance_ratio": float(unique_ratio),
            "ambiguous_frame_indices": ambiguous_frames,
            "ambiguous_ratio": float(ambiguous_ratio),
            "median_adjacent_mask_iou": median_iou,
            "identity_confirmation_required": True,
        }

    manifest = {
        "schema_version": 1,
        "source": "vista4d_run_sam3_video_seg_frames",
        "status": "candidate_unverified",
        "identity_verified": False,
        "identity_policy": (
            "automatic uniqueness and temporal stability create candidates only; "
            "identity_verified is never set without separate confirmation"
        ),
        "entities": entities,
    }
    _write_manifest(output_dir, manifest)
    return manifest


def export_instance_tracks(frames, output_dir, *, score_threshold=.6, frame_shape=None):
    """Preserve every SAM track separately; IDs are NOT character identities."""
    tracks={}; shape=frame_shape
    for index,instances in frames:
        for instance in instances:
            track=_value(instance,"id")
            if track is None: continue
            score=float(_value(instance,"score",0) or 0)
            if score<score_threshold: continue
            mask=_mask_array(_value(instance,"mask"))
            shape=mask.shape
            keyword=str(_value(instance,"keyword") or "object")
            key=f"{_slug(keyword)}_{int(track)}"
            tracks.setdefault(key,{"keyword":keyword,"frames":{}})["frames"][int(index)]=(mask,score)
    # Empty detections are a completed segmentation result, not a missing
    # export. Callers processing video supply its shape explicitly.
    if shape is None:
        raise ValueError("frame_shape is required when no SAM tracks were detected")
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("frame_shape must contain positive height and width")
    root=Path(output_dir); records={}; count=len(frames)
    root.mkdir(parents=True,exist_ok=True)
    for key,track in tracks.items():
        folder=root/"dynamic_mask_by_instance"/key; folder.mkdir(parents=True,exist_ok=True)
        boxes={}
        for index,_ in frames:
            mask,score=track["frames"].get(index,(np.zeros(shape,bool),0.))
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/f"{index:05d}.png")
            y,x=np.nonzero(mask)
            if len(x):
                boxes[str(index)]={"bbox":[float(x.min()/shape[1]),float(y.min()/shape[0]),
                    float((x.max()+1)/shape[1]),float((y.max()+1)/shape[0])],"score":score,"area":float(mask.mean())}
        records[key]={"keyword":track["keyword"],"mask_dir":str(folder.relative_to(root)),
            "frame_count":count,"observed_frames":len(boxes),"frames":boxes,"identity_verified":False}
    # Use the same atomic writer mechanism, but never overwrite keyword identities.
    path=root/"instance_tracks.json"
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version":1,"frame_count":count,"width":shape[1],"height":shape[0],"tracks":records,
        "segmentation_status":"tracks_detected" if records else "no_tracks_detected"},indent=2))
    os.replace(temporary,path)
