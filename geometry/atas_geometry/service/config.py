from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from atas_geometry import BACKBONES_DIR, GEOMETRY_ROOT


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _path_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    hot_root: Path
    archive_root: Path | None
    host: str
    port: int
    api_token: str
    max_gpus: int
    gpu_ids: tuple[int, ...]
    gpu_min_free_mib: int
    gpu_poll_seconds: float
    exclude_gpu0: bool
    vista4d_repo: Path
    vista4d_python: Path
    pi3_model_id: str
    sam3_checkpoint: Path
    recon_width: int
    recon_height: int
    render_width: int
    render_height: int
    pi3_pixel_limit: int
    pi3_vo_chunk_size: int
    pi3_vo_overlap: int
    low_vram: bool
    max_num_frames: int
    recon_deadline_seconds: float = 900.0
    render_deadline_seconds: float = 300.0
    worker_heartbeat_timeout: float = 90.0
    rpc_deadline_seconds: float = 120.0
    recon_min_free_mib: int = 24576

    @classmethod
    def from_env(cls) -> "Settings":
        user = os.getenv("USER", "user")
        gpu_ids_raw = os.getenv("GEOMETRY_GPU_IDS", "").strip()
        gpu_ids = (
            tuple(int(part.strip()) for part in gpu_ids_raw.split(",") if part.strip())
            if gpu_ids_raw
            else ()
        )
        archive = os.getenv("GEOMETRY_ARCHIVE_ROOT", "").strip()
        models = _path_env("GEOMETRY_MODEL_ROOT", str(GEOMETRY_ROOT / "models"))
        return cls(
            hot_root=_path_env(
                "GEOMETRY_HOT_ROOT", f"/var/tmp/geometry-service-{user}"
            ),
            archive_root=Path(archive) if archive else None,
            host=os.getenv("GEOMETRY_HOST", "127.0.0.1"),
            port=_int_env("GEOMETRY_PORT", 18300),
            api_token=os.getenv("GEOMETRY_API_TOKEN", "").strip(),
            max_gpus=max(1, _int_env("GEOMETRY_MAX_GPUS", 1)),
            gpu_ids=gpu_ids,
            gpu_min_free_mib=_int_env("GEOMETRY_GPU_MIN_FREE_MIB", 15360),
            gpu_poll_seconds=_float_env("GEOMETRY_GPU_POLL_SECONDS", 5.0),
            exclude_gpu0=os.getenv("GEOMETRY_EXCLUDE_GPU0", "0").strip().lower()
            not in {"0", "false", "no"},
            vista4d_repo=_path_env("GEOMETRY_BACKBONES_DIR", str(BACKBONES_DIR)),
            vista4d_python=_path_env("GEOMETRY_PYTHON", sys.executable),
            pi3_model_id=os.getenv("GEOMETRY_PI3_MODEL", str(models / "pi3x")),
            sam3_checkpoint=_path_env("GEOMETRY_SAM3_CHECKPOINT", str(models / "sam3" / "sam3.pt")),
            recon_width=_int_env("GEOMETRY_RECON_WIDTH", 1280),
            recon_height=_int_env("GEOMETRY_RECON_HEIGHT", 720),
            render_width=_int_env("GEOMETRY_RENDER_WIDTH", 1920),
            render_height=_int_env("GEOMETRY_RENDER_HEIGHT", 1080),
            pi3_pixel_limit=_int_env("GEOMETRY_PI3_PIXEL_LIMIT", 255000),
            pi3_vo_chunk_size=_int_env("GEOMETRY_PI3_VO_CHUNK", 48),
            pi3_vo_overlap=_int_env("GEOMETRY_PI3_VO_OVERLAP", 6),
            low_vram=os.getenv("GEOMETRY_LOW_VRAM", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            max_num_frames=_int_env("GEOMETRY_MAX_NUM_FRAMES", 2000),
            recon_deadline_seconds=_float_env("GEOMETRY_RECON_DEADLINE_SECONDS", 2400),
            render_deadline_seconds=_float_env("GEOMETRY_RENDER_DEADLINE_SECONDS", 900),
            worker_heartbeat_timeout=_float_env("GEOMETRY_WORKER_HEARTBEAT_TIMEOUT", 90),
            rpc_deadline_seconds=_float_env("GEOMETRY_RPC_DEADLINE_SECONDS", 300),
            recon_min_free_mib=_int_env("GEOMETRY_RECON_MIN_FREE_MIB", 24576),
        )
