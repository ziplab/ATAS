"""Frame-addressed camera context. Geometry stays on GPU; no model calls here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from atas_geometry.service.paths import l0_dir, l1_path, resolve_home_shot, home_shot_marker
from atas_geometry.world.recapture import identifier, load_source
def camera_context(root: Path, payload: dict) -> dict:
    run, story, scene = (identifier(payload.get(k), k) for k in ("run_id", "story_id", "scene_id"))
    identifier(payload.get("shot_id"), "shot_id")
    if payload.get("render_mode") == "recapture":
        source, meta, static = load_source(root, payload)
        source_id = payload["source_shot_id"]
        selection = "source_event"
    else:
        if not home_shot_marker(root, run, story, scene).is_file():
            raise ValueError("scene Home marker missing; do not guess a Home from another scene")
        home = resolve_home_shot(root, run, story, scene)
        source_id = home
        selection = "scene_home"
        requested = str(payload.get("previous_shot_id") or payload.get("start_from_shot_id") or "")
        if requested:
            requested = identifier(requested, "start_from_shot_id")
            candidate = l0_dir(root, run, story, requested) / "observation.json"
            if candidate.is_file():
                data = json.loads(candidate.read_text())
                if (data.get("registered") is True and data.get("scene_id") == scene
                        and data.get("run_id") == run and data.get("story_id") == story):
                    source_id, selection = requested, "latest_registered_observation"
        source = l0_dir(root, run, story, source_id)
        path = source / "observation.json"
        meta = json.loads(path.read_text()) if path.is_file() else {}
        static = l1_path(root, run, story, scene)
    if not static.is_file():
        raise ValueError("camera planning requires scene L1")
    with np.load(source / "cameras.npz") as cameras:
        camera_count = len(cameras["cam_c2w"])
    video = cv2.VideoCapture(str(source / "video.mp4"))
    try:
        count = min(camera_count, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
    finally:
        video.release()
    if count < 1 or min(width, height) < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid reconstructed source video")
    index = count - 1
    if payload.get("render_mode") == "recapture":
        # An exact event query cannot silently stretch a center-cropped clip.
        if "original_frame_count" not in meta or "source_fps" not in meta:
            raise ValueError("exact recapture needs source-frame timing metadata; re-register historical source first")
        fps = float(meta["source_fps"])
        offset = (int(meta["original_frame_count"]) - count) // 2
        start = float(payload.get("source_time_start", 0))
        index = min(count - 1, round(start * fps) - offset)
        last_t = start + (max(2, round(float(payload["duration"]) * int(payload.get("fps", 24)))) - 1) / int(payload.get("fps", 24))
        if start < offset / fps or last_t >= (offset + count) / fps or index < 0:
            raise ValueError("requested event interval extends beyond the actually reconstructed center slice")
    requested_frame = payload.get('camera_grounding_frame_index')
    if requested_frame is not None:
        if type(requested_frame) is not int or not 0 <= requested_frame < count:
            raise ValueError('grounding frame index outside reconstructed observation')
        if payload.get('render_mode') == 'recapture' and requested_frame != index:
            raise ValueError('recapture grounding must retain event start time')
        index = requested_frame
    result = {"schema_version": 1, "run_id": run, "story_id": story, "scene_id": scene,
        "source_shot_id": source_id, "source_selection": selection, "frame_index": index,
        "recon_frame_count": count, "width": width, "height": height,
        "source_sampling": "center_slice_then_center_crop", "source_fps": fps,
        "original_frame_count": meta.get("original_frame_count"),
        "original_video_sha256": meta.get("original_video_sha256")}
    stamps = []
    for path in (source / "video.mp4", source / "cameras.npz", source / "observation.json", static):
        if path.exists():
            stat = path.stat()
            stamps.append((path.name, stat.st_size, stat.st_mtime_ns))
    from atas_geometry.world.world_state import load_world
    from atas_geometry.service.paths import scene_dir
    result["world_version"]=(load_world(scene_dir(root,run,story,scene)) or {}).get("version")
    result["context_id"] = hashlib.sha256(json.dumps([result, stamps], sort_keys=True).encode()).hexdigest()
    return result
