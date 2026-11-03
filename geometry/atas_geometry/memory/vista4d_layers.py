"""L0 Vista4D RGB-D → L1 fused static + L2 time-local dynamic → one z-buffer."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from atas_geometry.memory.video_io import read_video_frames
from atas_geometry.memory.fusion import MemoryPointBatch
from atas_geometry.memory.observation import CameraCalibration
def _read_exr_depth(path: Path) -> np.ndarray:
    imath = importlib.import_module("Imath")
    openexr = importlib.import_module("OpenEXR")
    pixel_type = cast(Any, imath).PixelType(cast(Any, imath).PixelType.FLOAT)
    file = cast(Any, openexr).InputFile(str(path))
    try:
        data_window = file.header()["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1
        channel = file.channel("Y", pixel_type)
        return np.frombuffer(channel, dtype=np.float32).reshape(height, width)
    finally:
        file.close()


L1_MASK_AREA_FLOOR_RATIO = 0.35
DEFAULT_MASK_DILATE = 2


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by 4-connected steps.

    SAM masks cut tight against hair and cloth edges, so the outermost actor
    pixels read as mask-off and get fused into L1 as if they were wall.
    """
    grown = mask
    for _ in range(max(0, int(radius))):
        padded = np.pad(grown, 1, mode="edge")
        grown = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return grown


def _mask(path: Path, shape: tuple[int, int], *, dilate: int = 0) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (shape[1], shape[0]):
            mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        binary = np.asarray(mask, dtype=np.uint8) >= 128
    return _dilate(binary, dilate) if dilate else binary


def mask_area_report(recon: ReconStore) -> dict[str, Any]:
    """Per-frame dynamic-mask area plus the frames too small to trust.

    SAM3 routinely loses the subject for the opening frames of a clip. On
    `story_24/shot_01` the mask covered 2.4% of frame 0 and 27% of frame 209
    with the same three characters on screen, so the early frames fused actors
    into L1 as static room. Frames far below the clip median are excluded from
    L1 while remaining valid L0 and valid L2: the static surfaces they saw are
    covered by the rest of the clip anyway.
    """
    areas: list[float] = []
    for index in range(recon.count):
        if index >= len(recon.dynamic_files):
            areas.append(0.0)
            continue
        with Image.open(recon.dynamic_files[index]) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        areas.append(float(values.mean()))
    array = np.asarray(areas, dtype=np.float64)
    positive = array[array > 0.0]
    median = float(np.median(positive)) if positive.size else 0.0
    floor = median * L1_MASK_AREA_FLOOR_RATIO
    unreliable = (
        np.flatnonzero(array < floor).tolist() if median > 0.0 else []
    )
    return {
        "frame_count": int(array.size),
        "mask_area_by_frame": [float(value) for value in array],
        "mask_area_median": median,
        "mask_area_floor": float(floor),
        "mask_dilate_radius": int(recon.mask_dilate),
        "excluded_frame_indices": [int(value) for value in unreliable],
        "excluded_frame_count": len(unreliable),
        "policy": (
            "frames below 0.35x the clip median mask area are excluded from L1 "
            "fusion only; they stay valid L0 and valid L2"
        ),
    }


def _voxel_keys(xyz: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Pack a voxel coordinate into one int64 key, 21 bits per axis.

    Raising here is deliberate. This is the only hard limit left on L1 now that
    no point budget silently coarsens the grid, and a scene that trips it is
    one whose reconstruction spans more than two million voxels on some axis --
    the depth scale has run away, and any render from it would be geometry the
    camera cannot use. Failing loudly beats shipping an unusable point cloud.
    """
    ijk = np.floor((xyz.astype(np.float64) - origin) / voxel_size).astype(np.int64) + (1 << 20)
    if np.any(ijk < 0) | np.any(ijk >= (1 << 21)):
        span = float(np.ptp(xyz, axis=0).max())
        raise RuntimeError(
            "static voxel index overflow: scene spans "
            f"{span:.1f} units at voxel {voxel_size:.5f} "
            f"({span / max(voxel_size, 1e-9):,.0f} cells on the longest axis, "
            f"limit {1 << 21:,}). The reconstruction's depth scale is unbounded; "
            "this observation cannot back a point-cloud recapture."
        )
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


class ReconStore:
    def __init__(
        self,
        recon_dir: Path,
        *,
        duration: float,
        exclude_sky: bool,
        mask_dilate: int = DEFAULT_MASK_DILATE,
    ) -> None:
        self.recon_dir = recon_dir
        self.mask_dilate = int(mask_dilate)
        self.frames = read_video_frames(recon_dir / "video.mp4")
        camera_data = np.load(recon_dir / "cameras.npz")
        self.cam_c2w = camera_data["cam_c2w"].astype(np.float64)
        self.intrinsics = camera_data["intrinsics"].astype(np.float64)
        self.depth_files = sorted((recon_dir / "depths").glob("*.exr"))
        self.dynamic_files = sorted((recon_dir / "dynamic_mask").glob("*.png"))
        self.sky_files = sorted((recon_dir / "sky_mask").glob("*.png"))
        self.count = min(len(self.frames), len(self.depth_files), self.cam_c2w.shape[0])
        self.duration = duration
        self.exclude_sky = exclude_sky

    def depth_at(self, index: int) -> np.ndarray:
        """Metric depth for one recon frame, for callers that need no colour."""

        return _read_exr_depth(self.depth_files[int(index)])

    def dynamic_mask_at(self, index: int, shape: tuple[int, int]) -> np.ndarray:
        position = int(index)
        if position >= len(self.dynamic_files):
            return np.zeros(shape, dtype=bool)
        return _mask(self.dynamic_files[position], shape, dilate=self.mask_dilate)

    def entity_points_at(self, index: int, entity_id: str) -> np.ndarray:
        """Unproject an identity-verified per-entity L2 mask.

        This is intentionally stricter than ``dynamic_mask_at``. The latter is
        the union of all dynamic SAM results and cannot distinguish two actors
        of the same class. Offline entity-anchor experiments may provide
        ``entity_masks.json`` whose entries point to existing per-frame mask
        directories; no new 3D memory is stored.
        """
        manifest_path = self.recon_dir / "entity_masks.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"entity mask manifest missing at {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entities = payload.get("entities") if isinstance(payload, dict) else None
        record = entities.get(str(entity_id)) if isinstance(entities, dict) else None
        if not isinstance(record, dict):
            raise KeyError(f"no entity mask evidence for {entity_id!r}")
        confidence = float(record.get("confidence") or 0.0)
        if record.get("identity_verified") is not True or confidence < 0.8:
            raise ValueError(f"entity mask for {entity_id!r} is not identity verified")
        raw_mask_dir = str(record.get("mask_dir") or "").strip()
        if not raw_mask_dir:
            raise ValueError(f"entity mask directory missing for {entity_id!r}")
        mask_dir = (self.recon_dir / raw_mask_dir).resolve()
        try:
            mask_dir.relative_to(self.recon_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"entity mask directory escapes L0 for {entity_id!r}") from exc
        mask_files = sorted(mask_dir.glob("*.png")) if mask_dir.is_dir() else []
        position = int(index)
        if position < 0 or position >= len(mask_files):
            raise FileNotFoundError(
                f"entity mask frame {position} missing for {entity_id!r} in {mask_dir}"
            )

        depth = _read_exr_depth(self.depth_files[position])
        valid = np.isfinite(depth) & (depth > 1e-5)
        if self.exclude_sky and position < len(self.sky_files):
            valid &= ~_mask(self.sky_files[position], depth.shape)
        entity_mask = _mask(mask_files[position], depth.shape, dilate=self.mask_dilate)
        observation = self.unproject(position)
        return np.asarray(observation.xyz)[entity_mask[valid]]

    def unproject(self, index: int) -> MemoryPointBatch:
        depth = _read_exr_depth(self.depth_files[index])
        z = depth.astype(np.float32)
        valid = np.isfinite(z) & (z > 1e-5)
        dynamic = (
            _mask(self.dynamic_files[index], depth.shape, dilate=self.mask_dilate)
            if index < len(self.dynamic_files)
            else np.zeros_like(valid)
        )
        if self.exclude_sky and index < len(self.sky_files):
            valid &= ~_mask(self.sky_files[index], depth.shape)
        ys, xs = np.nonzero(valid)
        u = xs.astype(np.float64) + 0.5
        v = ys.astype(np.float64) + 0.5
        z_valid = z[valid].astype(np.float64)
        fx, fy, cx, cy = [float(value) for value in self.intrinsics[index, :4]]
        points_cam = np.stack(
            [(u - cx) / max(fx, 1e-6) * z_valid, (v - cy) / max(fy, 1e-6) * z_valid, z_valid],
            axis=1,
        )
        world = points_cam @ self.cam_c2w[index, :3, :3].T + self.cam_c2w[index, :3, 3]
        rgb = self.frames[index][ys, xs, :3]
        n = int(world.shape[0])
        t = index / max(1, self.count - 1) * self.duration
        return MemoryPointBatch(
            xyz=world.astype(np.float32),
            rgb=rgb.astype(np.uint8),
            world_time=np.full(n, t, dtype=np.float32),
            confidence=np.full(n, 0.7, dtype=np.float32),
            input_index=np.zeros(n, dtype=np.int32),
            camera_index=np.zeros(n, dtype=np.int16),
            foreground=dynamic[valid].astype(np.float32),
            camera_ids=("vista4d_master",),
        )

    def source_index(self, world_time: float) -> int:
        return int(
            np.clip(
                round((world_time / max(self.duration, 1e-6)) * (self.count - 1)),
                0,
                self.count - 1,
            )
        )


def voxel_size_from_median_depth(median_z: float, fy: float, *, k: float = 1.5) -> float:
    return max(float(k) * float(median_z) / max(float(fy), 1e-6), 1e-4)


NEAR_DEPTH_PERCENTILE = 20.0


def pixel_footprint_voxel_size(recon: ReconStore, *, k: float = 1.5) -> float:
    """Voxel edge sized for the closest surface this scene shows, not the median.

    The median depth is the middle of the room, so a later close-up renders a
    voxel that is several pixels wide and the surface breaks into spray. Sizing
    from a near percentile keeps tight framings solid.
    """
    mid = recon.count // 2
    depth = _read_exr_depth(recon.depth_files[mid])
    fy = float(recon.intrinsics[mid, 1])
    valid = np.isfinite(depth) & (depth > 1e-5)
    near_z = float(np.percentile(depth[valid], NEAR_DEPTH_PERCENTILE)) if np.any(valid) else 1.0
    return voxel_size_from_median_depth(near_z, fy, k=k)


def _reference_camera_at(
    recon: ReconStore,
    width: int,
    height: int,
    index: int,
) -> CameraCalibration:
    """Vista4D-world pose at one recon frame + K scaled to the PC render size."""

    index = int(np.clip(index, 0, recon.cam_c2w.shape[0] - 1))
    native_h, native_w = recon.frames[min(index, len(recon.frames) - 1)].shape[:2]
    scale_x = float(width) / max(float(native_w), 1.0)
    scale_y = float(height) / max(float(native_h), 1.0)
    fx, fy, cx, cy = [float(value) for value in recon.intrinsics[index, :4]]
    intrinsic = np.asarray(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraCalibration(
        camera_id=f"vista4d_event_master_{index:04d}",
        world_to_camera=np.linalg.inv(recon.cam_c2w[index]),
        intrinsic=intrinsic,
        image_size=(width, height),
    )


