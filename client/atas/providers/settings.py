from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendToggle(BaseModel):
    enabled: bool = False


class V2VAPIConfig(BackendToggle):
    """Provider-neutral defaults; concrete API payloads live in injected clients."""

    width: int = 1280
    height: int = 720
    frame_num: int = 121
    fps: int = 24
    model: str | None = None
    max_attempts: int = 1
    geometry_point_radius: int = 2
    # Production memory conditions retain every persisted point.  The
    # renderer uses bounded chunks internally; this must not become a sparse
    # sample cap.
    memory_render_max_points_per_frame: int = Field(default=0, ge=0)
    use_dense_vggt_condition: bool = False
    use_dense_vista4d_condition: bool = True
    require_dense_full_frame: bool = True
    geometry_coverage_preview_frames: int = Field(default=12, ge=2)
    # A zero-radius projection exposes the pixel lattice of RGB-D reprojection
    # at the provider resolution.  Radius two is still a local footprint of
    # each dense sample, while closing the one-pixel raster gaps.
    dense_geometry_point_radius: int = 2
    dense_geometry_small_gap_radius: int = 1
    dense_geometry_max_relative_depth_jump: float = 0.08
    condition_mode_override: str | None = None
    condition_known_confidence: float = 0.5
    condition_hybrid_low_confidence: float = 0.25
    condition_hybrid_high_confidence: float = 0.65
    condition_local_fill_radius: int = 3
    camera_fallback_target_weights: tuple[float, ...] = (0.75, 0.5, 0.25)
    camera_fallback_fov_steps: tuple[float, ...] = (5.0, 10.0)
    adapt_camera_for_geometry_coverage: bool = False
    min_continue_geometry_coverage: float = 0.03
    min_reobserve_geometry_coverage: float = 0.05
    min_geometry_frame_coverage: float = 0.01
    min_geometry_covered_frame_ratio: float = 0.80
    min_geometry_zone_coverage: float = Field(default=0.01, ge=0.0, le=1.0)
    # A target-camera video with less coverage is not a usable visual
    # condition.  This gate is applied before a paid provider request; it is
    # separate from the permissive legacy coverage thresholds above.
    min_dense_target_coverage: float = Field(default=0.45, ge=0.0, le=1.0)
    reference_only_reobserve: bool = False
    reference_only_continue_on_low_coverage: bool = False


class Wan27Config(V2VAPIConfig):
    """Alibaba Model Studio Wan 2.7 async video API configuration."""

    enabled: bool = True

    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    create_path: str = "/services/aigc/video-generation/video-synthesis"
    task_path_template: str = "/tasks/{task_id}"
    api_key: SecretStr | None = Field(default=None, repr=False)
    api_key_env: str = "DASHSCOPE_API_KEY"
    model_r2v: str = "wan2.7-r2v"
    model_i2v: str = "wan2.7-i2v"
    model_t2v: str = "wan2.7-t2v"
    width: int = 1280
    height: int = 720
    resolution: str = "720P"
    ratio: str = "16:9"
    fps: int = 24
    poll_interval_seconds: float = 5.0
    request_timeout_seconds: float = 120.0
    task_timeout_seconds: float = 3600.0
    transport_retries: int = Field(default=4, ge=1, le=8)
    transport_retry_backoff_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    prompt_extend: bool = False
    watermark: bool = False
    # Wan 3.0: omit = provider default (soundtrack on). False mutes the track.
    generate_audio: bool | None = None
    negative_prompt: str = (
        "blurry, low quality, distorted face, distorted body, changed identity, "
        "changed clothing, text, subtitles, logo, watermark"
    )
    max_reference_media: int = 6
    # Some Wan generations impose a total-duration budget across all video
    # references.  Keep this unset for Wan 2.7; newer models can opt in and
    # let the adapter pack local reference copies without changing story time.
    max_reference_video_seconds: float | None = Field(default=None, ge=1.0)
    max_persisted_tasks_per_shot: int = Field(default=3, ge=1)
    min_duration_seconds: int = 2
    max_duration_seconds: int = 15
    continuation_first_clip_seconds: float = Field(default=2.0, ge=2.0, le=4.0)
    use_dense_vggt_condition: bool = False
    require_dense_full_frame: bool = True
    # Wan receives this target render directly. These settings match the
    # dense camera showcase: keep depth-continuous surfaces through moderate
    # changes and close only local raster gaps, without generative filling.
    dense_geometry_small_gap_radius: int = 3
    dense_geometry_max_relative_depth_jump: float = 0.30
    # The rendered scene/event memory is sent to Wan as the reference video;
    # this is the conditioning the whole world state exists for.
    include_geometry_reference: bool = True
    geometry_reference_packing: Literal["video", "keyframes"] = "video"
    dense_geometry_reference_packing: Literal["video", "keyframes"] = "video"
    geometry_reference_keyframes: int = Field(default=4, ge=1, le=4)
    stage_local_media: bool = True
    temp_upload_path: str = "/uploads"
    temp_oss_expiration_seconds: int = 172800
    temp_oss_refresh_margin_seconds: int = 1800
    download_result: bool = True
    store_redacted_requests: bool = True
    reference_only_reobserve: bool = False


class BackendConfig(BaseModel):
    wan27: Wan27Config = Field(default_factory=Wan27Config)


class ImageConfig(BaseModel):
    """gpt-image-2 character/still generation."""

    model: str = "gpt-image-2"
    retries: int = 3


class AVWMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AVWM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    image: ImageConfig = Field(default_factory=ImageConfig)
    backends: BackendConfig = Field(default_factory=BackendConfig)
