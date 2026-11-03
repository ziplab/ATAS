from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atas.providers.wan.cast_assets import resolve_cast_assets
from atas.providers.wan.types import (
    CameraPose,
    DeterministicVisualCheck,
    Region,
    RouteDecision,
    ShotSpec,
)


@dataclass
class _SceneCluster:
    scene_id: str
    bbox_xyz: tuple[tuple[float, float, float], tuple[float, float, float]]
    center: tuple[float, float, float]
    tokens: set[str]


SCENE_ID_MAX_TOKENS = 4


class DirectorAgent:
    def __init__(self) -> None:
        self._location_scene_ids: dict[str, str] = {}

    def plan(self, story: Mapping[str, Any]) -> tuple[ShotSpec, ...]:
        raw_shots = story.get("shots")
        if not isinstance(raw_shots, Sequence):
            raise ValueError("Story must contain a sequence of shots")
        planned: list[ShotSpec] = []
        # Scoped to one plan: location ids are only guaranteed unique inside a
        # single screenplay.
        self._location_scene_ids = {}
        clusters: list[_SceneCluster] = []
        current_start = 0.0
        for index, raw in enumerate(raw_shots):
            if not isinstance(raw, Mapping):
                raise ValueError(f"Shot {index} must be a mapping")
            duration = float(raw.get("duration", 12.0))
            start = float(raw.get("start", current_start))
            current_start = start + duration
            region = self._region(raw, index, clusters)
            camera_path = self._camera_path(raw, start, duration)
            entity_ids = self._strings(raw.get("entity_ids", raw.get("entities")))
            cast_assets = resolve_cast_assets(story, raw, entity_ids)
            explicit_reference_image = self._optional_path(raw.get("reference_image"))
            yaml_cards = self._paths(raw.get("character_reference_images"))
            yaml_roles = self._strings(raw.get("character_reference_roles"))
            reference_image = (
                yaml_cards[0]
                if yaml_cards
                else (
                    cast_assets.image
                    if cast_assets.board_id and cast_assets.image is not None
                    else explicit_reference_image or cast_assets.image
                )
            )
            planned.append(
                ShotSpec(
                    shot_id=str(raw.get("id", f"shot_{index + 1}")),
                    prompt=str(raw.get("prompt", "")),
                    start=start,
                    duration=duration,
                    region=region,
                    camera_path=camera_path,
                    seed=int(raw.get("seed", index + 1)),
                    provider_prompt=str(
                        raw.get(
                            "provider_prompt",
                            raw.get("model_prompt", raw.get("cinematic_prompt", "")),
                        )
                    ),
                    force_route=self._route(raw.get("force_route")),
                    target_camera_npz=self._optional_path(raw.get("target_camera_npz")),
                    bootstrap_image=self._optional_path(raw.get("bootstrap_image")),
                    first_frame_image=self._optional_path(raw.get("first_frame_image")),
                    reference_image=reference_image,
                    reference_images=self._paths(raw.get("reference_images")),
                    # YAML character cards (reference stills) win when present.
                    # Otherwise keep the two-slot production face/body boards.
                    character_reference_images=(
                        yaml_cards
                        or tuple(
                            path
                            for path in (
                                cast_assets.face_image,
                                cast_assets.hero_face_image if cast_assets.legacy else None,
                            )
                            if path is not None
                        )
                    ),
                    character_reference_roles=(
                        yaml_roles[: len(yaml_cards)]
                        if yaml_cards and yaml_roles
                        else (
                            tuple("character_card" for _ in yaml_cards)
                            if yaml_cards
                            else tuple(
                                role
                                for role, path in (
                                    ("face_board", cast_assets.face_image),
                                    (
                                        "hero_face",
                                        cast_assets.hero_face_image if cast_assets.legacy else None,
                                    ),
                                )
                                if path is not None
                            )
                        )
                    ),
                    style_contract=str(
                        raw.get(
                            "style_contract",
                            raw.get("visual_style", ""),
                        )
                    ),
                    prop_reference_images=self._paths(raw.get("prop_reference_images")),
                    reference_videos=self._paths(raw.get("reference_videos")),
                    audio_reference=self._optional_path(raw.get("audio_reference")),
                    beat_id=str(raw.get("beat_id", "")),
                    event_id=str(raw.get("event_id", raw.get("beat_id", ""))),
                    entity_ids=entity_ids,
                    visible_cast=cast_assets.visible_cast,
                    cast_board_id=cast_assets.board_id,
                    cast_board_bindings=cast_assets.bindings,
                    cast_board_legacy=cast_assets.legacy,
                    location_id=str(raw.get("location_id", region.scene_id)),
                    scene_revisit=self._bool(raw.get("scene_revisit"), False),
                    suppress_event_video_reference=self._bool(
                        raw.get("suppress_event_video_reference"), False
                    ),
                    scene_graph_text=str(raw.get("scene_graph_text", "")),
                    felt_intent=str(raw.get("felt_intent", "")),
                    arc_position=str(raw.get("arc_position", "")),
                    project_voice=str(raw.get("project_voice", "")),
                    narrative_purpose=str(raw.get("narrative_purpose", "")),
                    objective=str(raw.get("objective", "")),
                    obstacle=str(raw.get("obstacle", "")),
                    tactic=str(raw.get("tactic", "")),
                    subtext=str(raw.get("subtext", "")),
                    emotion_before=str(raw.get("emotion_before", "")),
                    emotion_after=str(raw.get("emotion_after", "")),
                    visible_performance=str(raw.get("visible_performance", "")),
                    action_window_seconds=self._optional_float_pair(
                        raw.get("action_window_seconds")
                    ),
                    viewer_position=str(raw.get("viewer_position", "")),
                    visual_traffic=str(raw.get("visual_traffic", "")),
                    shot_size=str(raw.get("shot_size", "")),
                    camera_angle=str(raw.get("camera_angle", "")),
                    lens_feel=str(raw.get("lens_feel", "")),
                    camera_support=str(raw.get("camera_support", "")),
                    camera_movement=str(raw.get("camera_movement", "")),
                    camera_move_name=self._camera_move_name(raw),
                    camera_side_degrees=self._camera_side_degrees(raw),
                    camera_movement_degrees=self._camera_movement_degrees(raw),
                    subject_relation=str(raw.get("subject_relation", "")),
                    camera_endpoint=str(raw.get("camera_endpoint", "")),
                    camera_focus_point=self._optional_xyz(raw.get("camera_focus_point")),
                    practical_light=str(raw.get("practical_light", "")),
                    color_thesis=str(raw.get("color_thesis", "")),
                    capture_substrate=str(raw.get("capture_substrate", "")),
                    screen_direction=str(raw.get("screen_direction", "")),
                    planned_start_state=str(raw.get("planned_start_state", "")),
                    planned_end_state=str(raw.get("planned_end_state", "")),
                    already_happened=self._strings(raw.get("already_happened")),
                    this_clip_only=self._strings(raw.get("this_clip_only")),
                    reserved_for_later=self._strings(raw.get("reserved_for_later")),
                    fragile_anchors=self._strings(raw.get("fragile_anchors")),
                    must_show=self._strings(raw.get("must_show")),
                    forbidden_changes=self._strings(raw.get("forbidden_changes")),
                    memory_reference_shot_ids=self._strings(
                        raw.get("memory_reference_shot_ids")
                    ),
                    depends_on=self._strings(raw.get("depends_on")),
                    generation_order=self._optional_int(raw.get("generation_order")),
                    edit_order=self._optional_int(raw.get("edit_order")),
                    edit_in_seconds=float(raw.get("edit_in_seconds", 0.0)),
                    edit_out_seconds=self._optional_float(raw.get("edit_out_seconds")),
                    transition=str(raw.get("transition", "hard_cut")),
                    transition_duration=float(raw.get("transition_duration", 0.0)),
                    cut_in_type=self._optional_string(raw.get("cut_in_type")),
                    cut_out_type=self._optional_string(raw.get("cut_out_type")),
                    prev_shot_end_state=str(raw.get("prev_shot_end_state", "")),
                    next_shot_start_state=str(raw.get("next_shot_start_state", "")),
                    edit_in_state=str(raw.get("edit_in_state", "")),
                    edit_out_state=str(raw.get("edit_out_state", "")),
                    resume_master_after=self._bool(raw.get("resume_master_after"), True),
                    allow_world_time_replay=self._bool(
                        raw.get("allow_world_time_replay"),
                        False,
                    ),
                    condition_mode=str(raw.get("condition_mode", "rgb_splat")),
                    deterministic_visual_checks=self._visual_checks(
                        raw.get("deterministic_visual_checks")
                    ),
                    coverage_required=self._bool(raw.get("coverage_required"), True),
                    coverage_group_id=str(raw.get("coverage_group_id", "")),
                    coverage_role=str(raw.get("coverage_role", "")),
                    refreshes_global=self._bool(raw.get("refreshes_global"), False),
                    relation_hint=str(raw.get("relation_hint", "")),
                    global_variable_ids=self._strings(raw.get("global_variable_ids")),
                    auto_planned=self._bool(raw.get("auto_planned"), False),
                    planner_version=str(raw.get("planner_version", "")),
                    max_attempts=self._optional_int(raw.get("max_attempts")),
                    max_package_tokens=self._optional_int(raw.get("max_package_tokens")),
                )
            )
        return tuple(planned)

    def _region(
        self,
        raw: Mapping[str, Any],
        index: int,
        clusters: list[_SceneCluster],
    ) -> Region:
        region = raw.get("region", {})
        if not isinstance(region, Mapping):
            raise ValueError("shot.region must be a mapping")
        bbox = region.get("bbox_xyz", [[-1.0, -1.0, 0.0], [1.0, 1.0, 3.0]])
        if not isinstance(bbox, Sequence) or len(bbox) != 2:
            raise ValueError("region.bbox_xyz must have min and max coordinates")
        min_xyz = tuple(float(v) for v in bbox[0])
        max_xyz = tuple(float(v) for v in bbox[1])
        if len(min_xyz) != 3 or len(max_xyz) != 3:
            raise ValueError("region.bbox_xyz coordinates must be length 3")
        return Region(
            bbox_xyz=(
                (min_xyz[0], min_xyz[1], min_xyz[2]),
                (max_xyz[0], max_xyz[1], max_xyz[2]),
            ),
            label=str(region.get("label", "region")),
            scene_id=self._scene_id(
                raw,
                region,
                index,
                clusters,
                ((min_xyz[0], min_xyz[1], min_xyz[2]), (max_xyz[0], max_xyz[1], max_xyz[2])),
            ),
        )

    def _scene_id(
        self,
        raw: Mapping[str, Any],
        region: Mapping[str, Any],
        index: int,
        clusters: list[_SceneCluster],
        bbox_xyz: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> str:
        explicit = raw.get("scene_id", region.get("scene_id"))
        if explicit is not None:
            return str(explicit)
        # The screenplay's location id is the stable identity of a place. Slugging
        # the scene *description* instead made scene_id depend on LLM wording, so
        # a revisit phrased differently silently addressed a fresh, empty L1 --
        # exactly the continuity this project claims to provide.
        location_id = raw.get("location_id")
        if location_id is not None and str(location_id).strip():
            return self._location_scene_id(str(location_id))
        label = str(region.get("label", "scene"))
        prompt = str(raw.get("prompt", ""))
        tokens = self._scene_tokens(label, prompt)
        center = self._bbox_center(bbox_xyz)
        match = self._match_scene_cluster(clusters, bbox_xyz, center, tokens)
        if match is not None:
            match.tokens.update(tokens)
            match.bbox_xyz = self._merge_bbox(match.bbox_xyz, bbox_xyz)
            match.center = self._bbox_center(match.bbox_xyz)
            return match.scene_id
        scene_id = self._new_scene_id(label, index, clusters)
        clusters.append(_SceneCluster(scene_id, bbox_xyz, center, set(tokens)))
        return scene_id

    def _location_scene_id(self, location_id: str) -> str:
        """Map a screenplay location id to a stable, collision-free scene id.

        The same location id must always give the same scene id, and two
        different locations must never give the same one: a silent merge would
        fuse two rooms into one L1, which is worse than splitting one room in
        two. The short slug is only an affordance for reading run directories,
        so it yields to the full slug whenever it would collide.
        """
        known = self._location_scene_ids
        if location_id in known:
            return known[location_id]
        short = self._scene_base(location_id, max_tokens=SCENE_ID_MAX_TOKENS)
        taken = set(known.values())
        candidate = short
        if candidate in taken:
            candidate = self._scene_base(location_id, max_tokens=None)
            suffix = 2
            while candidate in taken:
                candidate = f"{short}_{suffix}"
                suffix += 1
        known[location_id] = candidate
        return candidate

    def _match_scene_cluster(
        self,
        clusters: list[_SceneCluster],
        bbox_xyz: tuple[tuple[float, float, float], tuple[float, float, float]],
        center: tuple[float, float, float],
        tokens: set[str],
    ) -> _SceneCluster | None:
        best: tuple[float, _SceneCluster] | None = None
        for cluster in clusters:
            overlap = self._bbox_iou(cluster.bbox_xyz, bbox_xyz)
            token_overlap = len(tokens & cluster.tokens)
            distance = self._center_distance(center, cluster.center)
            score = overlap * 3.0 + min(token_overlap, 3) * 0.75
            if distance <= 1.2:
                score += 0.75
            if overlap >= 0.28 or (token_overlap >= 2 and distance <= 2.8):
                if best is None or score > best[0]:
                    best = (score, cluster)
        return best[1] if best is not None else None

    def _new_scene_id(
        self,
        label: str,
        index: int,
        clusters: list[_SceneCluster],
    ) -> str:
        base = self._scene_base(label) or f"scene_{index + 1:02d}"
        existing = {cluster.scene_id for cluster in clusters}
        if base not in existing:
            return base
        suffix = 2
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"

    def _scene_base(self, label: str, *, max_tokens: int | None = None) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
        for suffix in ("_revisit", "_wide", "_close", "_detail", "_angle", "_shot"):
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        parts = [part for part in cleaned.split("_") if part]
        if len(parts) > 2 and parts[-1] in {"main", "side", "room", "area", "view"}:
            parts = parts[:-1]
        if max_tokens is not None:
            parts = parts[:max_tokens]
        return "_".join(parts) or "scene"

    def _scene_tokens(self, label: str, prompt: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "as",
            "at",
            "in",
            "of",
            "on",
            "same",
            "the",
            "to",
            "with",
        }
        text = f"{label} {prompt}".lower()
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text)
            if len(token) >= 4 and token not in stopwords
        }

    def _bbox_center(
        self,
        bbox_xyz: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        return (
            (bbox_xyz[0][0] + bbox_xyz[1][0]) / 2.0,
            (bbox_xyz[0][1] + bbox_xyz[1][1]) / 2.0,
            (bbox_xyz[0][2] + bbox_xyz[1][2]) / 2.0,
        )

    def _center_distance(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return float(sum((a[axis] - b[axis]) ** 2 for axis in range(3)) ** 0.5)

    def _merge_bbox(
        self,
        a: tuple[tuple[float, float, float], tuple[float, float, float]],
        b: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            (
                min(a[0][0], b[0][0]),
                min(a[0][1], b[0][1]),
                min(a[0][2], b[0][2]),
            ),
            (
                max(a[1][0], b[1][0]),
                max(a[1][1], b[1][1]),
                max(a[1][2], b[1][2]),
            ),
        )

    def _bbox_iou(
        self,
        a: tuple[tuple[float, float, float], tuple[float, float, float]],
        b: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> float:
        intersection_min = (
            max(a[0][0], b[0][0]),
            max(a[0][1], b[0][1]),
            max(a[0][2], b[0][2]),
        )
        intersection_max = (
            min(a[1][0], b[1][0]),
            min(a[1][1], b[1][1]),
            min(a[1][2], b[1][2]),
        )
        intersection = self._bbox_volume((intersection_min, intersection_max))
        if intersection <= 0.0:
            return 0.0
        union = self._bbox_volume(a) + self._bbox_volume(b) - intersection
        return intersection / max(union, 1e-6)

    def _bbox_volume(
        self,
        bbox_xyz: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> float:
        spans = [max(0.0, bbox_xyz[1][axis] - bbox_xyz[0][axis]) for axis in range(3)]
        return spans[0] * spans[1] * spans[2]

    def _camera_path(
        self,
        raw: Mapping[str, Any],
        start: float,
        duration: float,
    ) -> tuple[CameraPose, ...]:
        camera = raw.get("camera_path")
        if not isinstance(camera, Sequence) or isinstance(camera, str):
            focus = self._optional_xyz(raw.get("camera_focus_point")) or (0.0, 0.0, 1.0)
            return (
                CameraPose(t=start, position=(0.0, 0.0, 0.0), look_at=focus),
                CameraPose(t=start + duration, position=(0.25, 0.0, 0.0), look_at=focus),
            )
        poses: list[CameraPose] = []
        for item in camera:
            if not isinstance(item, Mapping):
                raise ValueError("camera path entries must be mappings")
            position = tuple(float(v) for v in item.get("position", [0.0, 0.0, 0.0]))
            look_at = tuple(float(v) for v in item.get("look_at", [0.0, 0.0, 1.0]))
            if len(position) != 3 or len(look_at) != 3:
                raise ValueError("camera position/look_at must be length 3")
            poses.append(
                CameraPose(
                    t=float(item.get("t", start)),
                    position=(position[0], position[1], position[2]),
                    look_at=(look_at[0], look_at[1], look_at[2]),
                    fov_degrees=float(item.get("fov_degrees", 55.0)),
                )
            )
        return tuple(poses)

    def _route(self, value: object) -> RouteDecision | None:
        if value is None:
            return None
        if str(value).upper() == RouteDecision.RESHOT.value:
            return RouteDecision.RESHOT
        if str(value).upper() == RouteDecision.GENERATED.value:
            return RouteDecision.GENERATED
        raise ValueError(f"Unknown force_route {value!r}")

    def _nested_camera(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        camera = raw.get("camera")
        return camera if isinstance(camera, Mapping) else {}

    def _camera_move_name(self, raw: Mapping[str, Any]) -> str:
        nested = self._nested_camera(raw)
        return str(raw.get("camera_move_name") or nested.get("movement") or "")

    def _camera_side_degrees(self, raw: Mapping[str, Any]) -> float:
        nested = self._nested_camera(raw)
        value = raw.get("camera_side_degrees", nested.get("side_degrees", 0.0))
        return float(value or 0.0)

    def _camera_movement_degrees(self, raw: Mapping[str, Any]) -> float:
        nested = self._nested_camera(raw)
        value = raw.get("camera_movement_degrees", nested.get("movement_degrees", 0.0))
        return float(value or 0.0)

    def _optional_path(self, value: object) -> Path | None:
        if value is None:
            return None
        return Path(str(value))

    def _paths(self, value: object) -> tuple[Path, ...]:
        return tuple(Path(item) for item in self._strings(value))

    def _strings(self, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, Sequence):
            raise ValueError(f"Expected a string sequence, got {type(value).__name__}")
        return tuple(str(item) for item in value)

    def _optional_int(self, value: object) -> int | None:
        return None if value is None else int(str(value))

    def _optional_float(self, value: object) -> float | None:
        return None if value is None else float(str(value))

    def _optional_float_pair(self, value: object) -> tuple[float, float] | None:
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError("Expected a two-value numeric sequence")
        return float(value[0]), float(value[1])

    def _optional_xyz(self, value: object) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 3:
            raise ValueError("Expected a three-value numeric sequence")
        values = tuple(float(item) for item in value)
        if not all(value == value and abs(value) != float("inf") for value in values):
            raise ValueError("camera_focus_point must contain finite numbers")
        return values

    def _optional_string(self, value: object) -> str | None:
        return None if value is None else str(value)

    def _bool(self, value: object, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, got {value!r}")

    def _visual_checks(self, value: object) -> tuple[DeterministicVisualCheck, ...]:
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("deterministic_visual_checks must be a sequence")
        checks: list[DeterministicVisualCheck] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"deterministic_visual_checks[{index}] must be a mapping"
                )
            roi = item.get("roi_xyxy")
            roi_xyxy: tuple[float, float, float, float] | None = None
            if roi is not None:
                if isinstance(roi, str) or not isinstance(roi, Sequence) or len(roi) != 4:
                    raise ValueError("visual check roi_xyxy must contain four values")
                roi_xyxy = tuple(float(part) for part in roi)  # type: ignore[assignment]
            raw_thresholds = item.get("thresholds", {})
            if not isinstance(raw_thresholds, Mapping):
                raise ValueError("visual check thresholds must be a mapping")
            checks.append(
                DeterministicVisualCheck(
                    check_id=str(item.get("id", f"check_{index + 1}")),
                    detector=str(item.get("detector", "")),
                    predicate=str(item.get("predicate", "")),
                    roi_xyxy=roi_xyxy,
                    action_window_seconds=self._optional_float_pair(
                        item.get("action_window_seconds")
                    ),
                    thresholds=tuple(
                        (str(key), float(number))
                        for key, number in raw_thresholds.items()
                    ),
                )
            )
        return tuple(checks)
