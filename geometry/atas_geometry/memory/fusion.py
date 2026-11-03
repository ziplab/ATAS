from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from atas_geometry.memory.types import AnyArray
if TYPE_CHECKING:
    from atas_geometry.memory.query import MemoryQueryResult
@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: AnyArray
    translation: AnyArray

    def apply(self, points: AnyArray) -> AnyArray:
        return self.scale * (points @ self.rotation.T) + self.translation

    def to_json_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }


@dataclass(frozen=True)
class MemoryPointBatch:
    xyz: AnyArray
    rgb: AnyArray
    world_time: AnyArray
    confidence: AnyArray
    input_index: AnyArray
    camera_index: AnyArray
    foreground: AnyArray
    camera_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.xyz.shape[0]
        arrays = (
            self.rgb,
            self.world_time,
            self.confidence,
            self.input_index,
            self.camera_index,
            self.foreground,
        )
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        if any(array.shape[0] != count for array in arrays):
            raise ValueError("memory point arrays must have the same first dimension")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A point batch is the lossless hand-off representation between a
        # reconstruction and world-memory writeback.  Do not make a smaller
        # archive by changing or subsampling its points.
        np.savez(
            path,
            xyz=self.xyz.astype(np.float32),
            rgb=self.rgb.astype(np.uint8),
            world_time=self.world_time.astype(np.float32),
            confidence=self.confidence.astype(np.float32),
            input_index=self.input_index.astype(np.int32),
            camera_index=self.camera_index.astype(np.int16),
            foreground=self.foreground.astype(np.float32),
            camera_ids=np.asarray(self.camera_ids),
        )

    @classmethod
    def load(cls, path: Path) -> MemoryPointBatch:
        data = np.load(path)
        return cls(
            xyz=data["xyz"],
            rgb=data["rgb"],
            world_time=data["world_time"],
            confidence=data["confidence"],
            input_index=data["input_index"],
            camera_index=data["camera_index"],
            foreground=data["foreground"],
            camera_ids=tuple(str(value) for value in data["camera_ids"].tolist()),
        )


@dataclass(frozen=True)
class FusedSceneMemory:
    """Persistent dense static-scene evidence in the shared world frame.

    Dynamic observations are deliberately not retained here.  They belong to
    the time-local RGB-D layer used while rendering a target world time.
    """

    xyz: AnyArray
    rgb: AnyArray
    world_time: AnyArray
    confidence: AnyArray
    foreground: AnyArray
    weight: AnyArray
    point_count: AnyArray
    source_count: AnyArray
    conflict_score: AnyArray
    provenance_offsets: AnyArray
    provenance_input_indices: AnyArray
    voxel_size: float
    time_bin_size: float
    camera_ids: tuple[str, ...]
    voxel_origin: AnyArray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    storage_mode: str = "dense_uncompressed"

    def query(
        self,
        world_time: float,
        *,
        time_radius: float | None = None,
        min_confidence: float = 0.0,
    ) -> MemoryQueryResult:
        from atas_geometry.memory.query import query_memory
        return query_memory(
            self,
            world_time,
            time_radius=time_radius,
            min_confidence=min_confidence,
        )

    def save(self, path: Path, metadata: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # ``fused_memory.npz`` is historical terminology.  The current
        # representation deliberately preserves every static input point; it
        # is an uncompressed NPZ so file encoding does not hide a reduction.
        np.savez(
            path,
            xyz=self.xyz.astype(np.float32),
            rgb=self.rgb.astype(np.uint8),
            world_time=self.world_time.astype(np.float32),
            confidence=self.confidence.astype(np.float32),
            foreground=self.foreground.astype(np.float32),
            weight=self.weight.astype(np.float32),
            point_count=self.point_count.astype(np.int32),
            source_count=self.source_count.astype(np.int16),
            conflict_score=self.conflict_score.astype(np.float32),
            provenance_offsets=self.provenance_offsets.astype(np.int64),
            provenance_input_indices=self.provenance_input_indices.astype(np.int32),
            voxel_size=np.asarray(self.voxel_size, dtype=np.float32),
            time_bin_size=np.asarray(self.time_bin_size, dtype=np.float32),
            camera_ids=np.asarray(self.camera_ids),
            voxel_origin=self.voxel_origin.astype(np.float32),
            storage_mode=np.asarray(self.storage_mode),
        )
        if metadata is not None:
            defaults = {
                "memory_representation": "dense_static_points",
                "point_reduction": "none",
                "voxel_merge": False,
                "archive_compression": "none",
            }
            metadata = {**defaults, **metadata}
            path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, path: Path) -> FusedSceneMemory:
        data = np.load(path)
        voxel_size = float(data["voxel_size"])
        xyz = data["xyz"]
        voxel_origin = (
            data["voxel_origin"]
            if "voxel_origin" in data.files
            else _canonical_voxel_origin(xyz, voxel_size)
        )
        return cls(
            xyz=xyz,
            rgb=data["rgb"],
            world_time=data["world_time"],
            confidence=data["confidence"],
            foreground=data["foreground"],
            weight=data["weight"],
            point_count=data["point_count"],
            source_count=data["source_count"],
            conflict_score=data["conflict_score"],
            provenance_offsets=data["provenance_offsets"],
            provenance_input_indices=data["provenance_input_indices"],
            voxel_size=voxel_size,
            time_bin_size=float(data["time_bin_size"]),
            camera_ids=tuple(str(value) for value in data["camera_ids"].tolist()),
            voxel_origin=voxel_origin,
            storage_mode=(
                str(data["storage_mode"].item())
                if "storage_mode" in data.files
                else "legacy_voxel_aggregated"
            ),
        )


def fuse_memory_points(
    batch: MemoryPointBatch,
    *,
    voxel_size: float | None = None,
    time_bin_size: float = 0.04,
    voxel_origin: AnyArray | None = None,
) -> FusedSceneMemory:
    """Create a lossless dense memory snapshot from a point observation.

    ``voxel_size`` is retained only as a physical tolerance scale for legacy
    calibration and depth-anchor callers.  It no longer controls persistence:
    no two points are merged, averaged, sampled, or otherwise replaced.
    """
    if batch.xyz.shape[0] == 0:
        raise ValueError("Cannot fuse an empty memory point batch")
    if time_bin_size <= 0:
        raise ValueError("time_bin_size must be positive")
    if voxel_size is None:
        lo, hi = np.percentile(batch.xyz, [1.0, 99.0], axis=0)
        voxel_size = max(float(np.max(hi - lo)) / 300.0, 1e-5)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    origin = (
        _canonical_voxel_origin(batch.xyz, voxel_size)
        if voxel_origin is None
        else np.asarray(voxel_origin, dtype=np.float64)
    )
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("voxel_origin must be a finite 3D point")
    # Stable chronological ordering preserves the fast time-local query path
    # while retaining the exact input rows and their source ordering inside a
    # frame.  There is intentionally no spatial or time-bin aggregation here.
    order = np.argsort(batch.world_time, kind="stable")
    count = int(order.size)
    weights = np.clip(batch.confidence[order].astype(np.float64), 0.01, 1.0)
    offsets = np.arange(count + 1, dtype=np.int64)
    return FusedSceneMemory(
        xyz=batch.xyz[order].astype(np.float32, copy=False),
        rgb=batch.rgb[order].astype(np.uint8, copy=False),
        world_time=batch.world_time[order].astype(np.float32, copy=False),
        confidence=batch.confidence[order].astype(np.float32, copy=False),
        foreground=batch.foreground[order].astype(np.float32, copy=False),
        weight=weights.astype(np.float32),
        point_count=np.ones(count, dtype=np.int32),
        source_count=np.ones(count, dtype=np.int16),
        conflict_score=np.zeros(count, dtype=np.float32),
        provenance_offsets=offsets,
        provenance_input_indices=batch.input_index[order].astype(np.int32, copy=False),
        voxel_size=voxel_size,
        time_bin_size=time_bin_size,
        camera_ids=batch.camera_ids,
        voxel_origin=origin.astype(np.float32),
    )


def _canonical_voxel_origin(xyz: AnyArray, voxel_size: float) -> AnyArray:
    lower = np.percentile(xyz.astype(np.float64), 1.0, axis=0)
    return np.floor(lower / voxel_size) * voxel_size


def _umeyama(source: AnyArray, target: AnyArray) -> SimilarityTransform:
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Similarity alignment requires matching point arrays")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.sum(source_centered**2) / source.shape[0])
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    scale = float(np.sum(singular_values * np.diag(sign)) / max(source_variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


