from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

# Parameterized array aliases used in place of bare ``np.ndarray`` so that mypy
# ``--strict`` (disallow_any_generics) is satisfied. The shape parameter is
# ``Any`` so concrete-shape arrays (e.g. ``imread`` returning a 3-tuple shape)
# stay assignable through invariant containers like ``list``.
FrameArray: TypeAlias = np.ndarray[Any, np.dtype[np.uint8]]
AnyArray: TypeAlias = np.ndarray[Any, np.dtype[Any]]
BBoxXYZ: TypeAlias = tuple[tuple[float, float, float], tuple[float, float, float]]
TimeRange: TypeAlias = tuple[float, float]


class RouteDecision(str, Enum):
    GENERATED = "GENERATED"
    RESHOT = "RE-SHOT"


class ExecutionType(str, Enum):
    """Production operation after relation compile: generate, continue, or reobserve."""

    GENERATED = "GENERATED"
    CONTINUE = "CONTINUE"
    REOBSERVE = "REOBSERVE"


class V2VTask(str, Enum):
    """Temporal intent for the unified memory-to-video executor."""

    CONTINUE = "CONTINUE"
    REOBSERVE = "REOBSERVE"

    @property
    def route(self) -> RouteDecision:
        return (
            RouteDecision.GENERATED
            if self is V2VTask.CONTINUE
            else RouteDecision.RESHOT
        )


class ReferenceRole(str, Enum):
    """Semantic role of a multimodal reference presented to a V2V model."""

    TARGET_GEOMETRY = "target_geometry"
    EVENT_MASTER = "event_master"
    CONTINUITY_MEMORY = "continuity_memory"
    FIRST_FRAME = "first_frame"
    CHARACTER = "character"
    PROP = "prop"
    SCENE_STYLE = "scene_style"
    USER_VIDEO = "user_video"
    GLOBAL_STATE = "global_state"
    GLOBAL_CROP = "global_crop"
    AUDIO = "audio"


@dataclass(frozen=True)
class DeterministicVisualCheck:
    """Machine-checkable acceptance condition for a narrow visual signal.

    Free-form ``must_show`` text remains a semantic contract for the VLM critic.
    This structure is only for checks that can be evaluated deterministically.
    """

    check_id: str
    detector: str
    predicate: str
    roi_xyxy: tuple[float, float, float, float] | None = None
    action_window_seconds: TimeRange | None = None
    thresholds: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class MediaReference:
    """Ordered provider-neutral media reference.

    ``ordinal`` is one-based within its media type and maps directly to prompt
    handles such as ``@视频1`` and ``@图片1``.
    """

    path: Path
    role: ReferenceRole
    ordinal: int
    media_type: str
    label: str
    duration_seconds: float | None = None
    source_shot_id: str | None = None
    asset_role: str = ""
    character_mapping: tuple[str, ...] = ()
    uri: str | None = None

    @property
    def prompt_handle(self) -> str:
        prefix = {"video": "@视频", "image": "@图片", "audio": "@音频"}.get(
            self.media_type,
            "@素材",
        )
        return f"{prefix}{self.ordinal}"


@dataclass(frozen=True)
class CameraPose:
    """Authored CV camera pose; ``fov_degrees`` is the vertical field of view."""

    t: float
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_degrees: float = 55.0


@dataclass(frozen=True)
class Region:
    bbox_xyz: BBoxXYZ
    label: str = "region"
    scene_id: str = "default"


@dataclass(frozen=True)
class RGBPoint:
    x: float
    y: float
    z: float
    t: float
    r: int
    g: int
    b: int
    feature: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ShotSpec:
    shot_id: str
    prompt: str
    start: float
    duration: float
    region: Region
    camera_path: tuple[CameraPose, ...]
    seed: int
    provider_prompt: str = ""
    force_route: RouteDecision | None = None
    target_camera_npz: Path | None = None
    bootstrap_image: Path | None = None
    first_frame_image: Path | None = None
    reference_image: Path | None = None
    reference_images: tuple[Path, ...] = ()
    character_reference_images: tuple[Path, ...] = ()
    character_reference_roles: tuple[str, ...] = ()
    style_contract: str = ""
    prop_reference_images: tuple[Path, ...] = ()
    reference_videos: tuple[Path, ...] = ()
    audio_reference: Path | None = None
    beat_id: str = ""
    event_id: str = ""
    entity_ids: tuple[str, ...] = ()
    location_id: str = ""
    scene_revisit: bool = False
    suppress_event_video_reference: bool = False
    scene_graph_text: str = ""
    felt_intent: str = ""
    arc_position: str = ""
    project_voice: str = ""
    narrative_purpose: str = ""
    objective: str = ""
    obstacle: str = ""
    tactic: str = ""
    subtext: str = ""
    emotion_before: str = ""
    emotion_after: str = ""
    visible_performance: str = ""
    action_window_seconds: TimeRange | None = None
    viewer_position: str = ""
    visual_traffic: str = ""
    shot_size: str = ""
    camera_angle: str = ""
    lens_feel: str = ""
    camera_support: str = ""
    camera_movement: str = ""
    camera_move_name: str = ""
    camera_side_degrees: float = 0.0
    camera_movement_degrees: float = 0.0
    subject_relation: str = ""
    camera_endpoint: str = ""
    camera_focus_point: tuple[float, float, float] | None = None
    practical_light: str = ""
    color_thesis: str = ""
    capture_substrate: str = ""
    screen_direction: str = ""
    planned_start_state: str = ""
    planned_end_state: str = ""
    already_happened: tuple[str, ...] = ()
    this_clip_only: tuple[str, ...] = ()
    reserved_for_later: tuple[str, ...] = ()
    fragile_anchors: tuple[str, ...] = ()
    must_show: tuple[str, ...] = ()
    forbidden_changes: tuple[str, ...] = ()
    memory_reference_shot_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    generation_order: int | None = None
    edit_order: int | None = None
    edit_in_seconds: float = 0.0
    edit_out_seconds: float | None = None
    transition: str = "hard_cut"
    transition_duration: float = 0.0
    cut_in_type: str | None = None
    cut_out_type: str | None = None
    prev_shot_end_state: str = ""
    next_shot_start_state: str = ""
    edit_in_state: str = ""
    edit_out_state: str = ""
    resume_master_after: bool = True
    allow_world_time_replay: bool = False
    condition_mode: str = "rgb_splat"
    deterministic_visual_checks: tuple[DeterministicVisualCheck, ...] = ()
    coverage_required: bool = True
    coverage_group_id: str = ""
    coverage_role: str = ""
    refreshes_global: bool = False
    relation_hint: str = ""
    global_variable_ids: tuple[str, ...] = ()
    evidence_references: tuple[MediaReference, ...] = ()
    auto_planned: bool = False
    planner_version: str = ""
    max_attempts: int | None = None
    max_package_tokens: int | None = None
    cast_board_id: str = ""
    cast_board_bindings: tuple[str, ...] = ()
    cast_board_legacy: bool = False
    visible_cast: tuple[str, ...] = ()

    @property
    def t_range(self) -> TimeRange:
        return (self.start, self.start + self.duration)


@dataclass
class VideoClip:
    shot_id: str
    backend: str
    route: RouteDecision
    fps: int
    duration: float
    frames: list[FrameArray]
    start_time: float = 0.0
    artifact_path: Path | None = None
    points: list[RGBPoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class GenerationCondition:
    """Model-independent files prepared from the current world state."""

    mode: str
    start_image: Path | None = None
    reference_image: Path | None = None
    control_video: Path | None = None
    control_kind: str | None = None
    report_path: Path | None = None
    memory_time: float | None = None
    coverage: float = 0.0


@dataclass(frozen=True)
class V2VCondition:
    """Provider-independent video condition rendered from 4D world memory."""

    task: V2VTask
    input_video: Path
    camera_query: Path
    report_path: Path
    geometry_source: str = "fused_scene_memory"
    coverage_video: Path | None = None
    presentation_coverage_video: Path | None = None
    reference_image: Path | None = None
    target_world_times: tuple[float, ...] = ()
    memory_world_times: tuple[float, ...] = ()
    coverage: float = 0.0
    coverage_by_frame: tuple[float, ...] = ()
    coverage_zones: dict[str, float] = field(default_factory=dict)
    presentation_coverage: float = 0.0
    condition_mode: str = "rgb_splat"
    event_master_shot_id: str | None = None
    schema_version: int = 2
    references: tuple[MediaReference, ...] = ()


@dataclass(frozen=True)
class V2VRequest:
    """Stable AVWM boundary for Seedance, Wan, or another V2V provider."""

    shot: ShotSpec
    task: V2VTask
    prompt: str
    condition: V2VCondition
    fps: int
    width: int
    height: int
    frame_count: int
    output_dir: Path
    mock_mode: bool = False


@dataclass(frozen=True)
class VideoRequest:
    shot: ShotSpec
    fps: int
    width: int
    height: int
    output_dir: Path
    mock_mode: bool
    condition: GenerationCondition | None = None


@dataclass(frozen=True)
class ShotDecision:
    shot_id: str
    route: RouteDecision
    coverage: float
    threshold: float
    backend: str
    reason: str
    camera_path: tuple[CameraPose, ...] = ()
    scene_id: str = "default"
    attempts: int = 1
    critic_reports: tuple[dict[str, object], ...] = ()
    retry_notes: tuple[str, ...] = ()
    execution_type: ExecutionType = ExecutionType.GENERATED

    def to_json_dict(self) -> dict[str, object]:
        semantic_route = (
            self.execution_type.value
            if self.execution_type is not ExecutionType.GENERATED
            else self.route.value
        )
        return {
            "shot_id": self.shot_id,
            "route": semantic_route,
            "legacy_route": self.route.value,
            "execution_type": self.execution_type.value,
            "scene_id": self.scene_id,
            "coverage": round(self.coverage, 6),
            "threshold": round(self.threshold, 6),
            "backend": self.backend,
            "reason": self.reason,
            "attempts": self.attempts,
            "critic_reports": list(self.critic_reports),
            "retry_notes": list(self.retry_notes),
            "camera_path": [
                {
                    "t": round(pose.t, 6),
                    "position": [round(value, 6) for value in pose.position],
                    "look_at": [round(value, 6) for value in pose.look_at],
                    "fov_degrees": round(pose.fov_degrees, 6),
                }
                for pose in self.camera_path
            ],
        }
