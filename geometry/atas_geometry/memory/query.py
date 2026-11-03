from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atas_geometry.memory.fusion import FusedSceneMemory
from atas_geometry.memory.types import AnyArray
# Fused memories are emitted in nondecreasing world-time order.  Keep the
# validation result tied to the array object so repeated frame queries do not
# rescan a multi-million-point memory before using searchsorted.
_SORTED_TIME_CACHE: dict[int, tuple[AnyArray, bool]] = {}


def _world_time_is_sorted(world_time: AnyArray) -> bool:
    key = id(world_time)
    cached = _SORTED_TIME_CACHE.get(key)
    if cached is not None and cached[0] is world_time:
        return cached[1]
    result = bool(world_time.size < 2 or np.all(world_time[:-1] <= world_time[1:]))
    _SORTED_TIME_CACHE[key] = (world_time, result)
    return result


@dataclass(frozen=True)
class MemoryQueryResult:
    """A strict time-local view into fused 4D memory."""

    memory: FusedSceneMemory
    world_time: float
    time_radius: float
    indices: AnyArray

    @property
    def point_count(self) -> int:
        return int(self.indices.size)

    @property
    def xyz(self) -> AnyArray:
        return self.memory.xyz[self.indices]

    @property
    def time_distance(self) -> AnyArray:
        return np.abs(self.memory.world_time[self.indices] - self.world_time)

    def to_json_dict(self) -> dict[str, object]:
        if self.point_count:
            selected_times = self.memory.world_time[self.indices]
            selected_range: list[float] | None = [
                float(selected_times.min()),
                float(selected_times.max()),
            ]
        else:
            selected_range = None
        return {
            "world_time": self.world_time,
            "time_radius": self.time_radius,
            "point_count": self.point_count,
            "selected_time_range": selected_range,
        }


def query_memory(
    memory: FusedSceneMemory,
    world_time: float,
    *,
    time_radius: float | None = None,
    min_confidence: float = 0.0,
) -> MemoryQueryResult:
    if not np.isfinite(world_time):
        raise ValueError("world_time must be finite")
    radius = memory.time_bin_size * 0.51 if time_radius is None else float(time_radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("time_radius must be finite and non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")

    world_times = memory.world_time
    if _world_time_is_sorted(world_times):
        lower = int(np.searchsorted(world_times, world_time - radius, side="left"))
        upper = int(np.searchsorted(world_times, world_time + radius, side="right"))
        indices = np.arange(lower, upper, dtype=np.int64)
        if min_confidence > 0.0 and indices.size:
            indices = indices[memory.confidence[indices] >= min_confidence]
    else:
        selected = np.abs(world_times - world_time) <= radius
        selected &= memory.confidence >= min_confidence
        indices = np.flatnonzero(selected)
    return MemoryQueryResult(memory, float(world_time), radius, indices)
