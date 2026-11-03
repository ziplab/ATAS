from __future__ import annotations

import csv
import subprocess
import threading
import time
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int
    utilization_percent: int
    telemetry_valid: bool = True


def query_gpus(nvidia_smi_bin: str = "nvidia-smi") -> list[GPUInfo]:
    command = [
        nvidia_smi_bin,
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=10
    )
    gpus: list[GPUInfo] = []
    for row in csv.reader(StringIO(completed.stdout), skipinitialspace=True):
        if len(row) != 5:
            continue
        try:
            gpu = GPUInfo(
                index=int(row[0]),
                name=row[1].strip(),
                memory_total_mib=int(row[2]),
                memory_free_mib=int(row[3]),
                utilization_percent=int(row[4]),
            )
        except ValueError:
            # Unhealthy/unavailable telemetry (e.g. utilization [N/A]) must
            # neither admit this card nor disable the other healthy cards.
            try:
                gpu = GPUInfo(int(row[0]), row[1].strip(), 0, 0, 100, False)
            except ValueError:
                continue
        if not (gpu.memory_total_mib > 0 and
                0 <= gpu.memory_free_mib <= gpu.memory_total_mib and
                0 <= gpu.utilization_percent <= 100):
            gpu = GPUInfo(gpu.index, gpu.name, 0, 0, 100, False)
        gpus.append(gpu)
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return gpus


class GpuPool:
    """At most ``max_gpus`` physical cards, leased one job at a time."""

    def __init__(
        self,
        *,
        max_gpus: int = 3,
        configured: tuple[int, ...] = (),
        min_free_mib: int = 15360,
        poll_seconds: float = 5.0,
        exclude_gpu0: bool = True,
    ) -> None:
        self.max_gpus = max(1, int(max_gpus))
        self.min_free_mib = int(min_free_mib)
        self.poll_seconds = float(poll_seconds)
        self.exclude_gpu0 = bool(exclude_gpu0)
        observed = query_gpus()
        if configured:
            if self.exclude_gpu0 and 0 in configured:
                raise ValueError("GPU 0 is excluded from the geometry pool")
            allowed = {gpu.index for gpu in observed}
            missing = [index for index in configured if index not in allowed]
            if missing:
                raise RuntimeError(f"GEOMETRY_GPU_IDS not present: {missing}")
            self.indices = tuple(dict.fromkeys(configured))
        else:
            ranked = sorted(
                (
                    gpu
                    for gpu in observed
                    if not (self.exclude_gpu0 and gpu.index == 0)
                ),
                key=lambda gpu: (gpu.memory_free_mib, gpu.index),
                reverse=True,
            )
            if not ranked:
                raise RuntimeError("no eligible GPUs for the geometry pool")
            self.indices = tuple(gpu.index for gpu in ranked[: self.max_gpus])
        self._busy: set[int] = set()
        self._cursor = 0
        self._cv = threading.Condition()

    def snapshot(self) -> dict[str, object]:
        observed = {gpu.index: gpu for gpu in query_gpus()}
        return {
            "pool": list(self.indices),
            "max_concurrent_gpus": self.max_gpus,
            "busy": sorted(self._busy),
            "cards": [
                {
                    "index": index,
                    "free_mib": observed[index].memory_free_mib
                    if index in observed
                    else None,
                    "busy": index in self._busy,
                    "telemetry_valid": observed[index].telemetry_valid if index in observed else False,
                }
                for index in self.indices
            ],
        }

    def acquire(self, *, min_free_mib: int | None = None, stop: threading.Event,
                timeout_seconds: float | None = None) -> int:
        need = int(self.min_free_mib if min_free_mib is None else min_free_mib)
        started = time.monotonic()
        while not stop.is_set():
            if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                from atas_geometry.service.resource_errors import ResourceUnavailableError
                raise ResourceUnavailableError("gpu_capacity_unavailable", "GPU resource queue deadline exceeded",
                    min_free_mib=need, candidate_gpu_indices=list(self.indices),
                    waited_seconds=round(time.monotonic()-started, 3))
            with self._cv:
                if len(self._busy) >= self.max_gpus:
                    self._cv.wait(timeout=min(self.poll_seconds, .1))
                    continue
                observed = {gpu.index: gpu for gpu in query_gpus()}
                order = list(self.indices)
                if order:
                    rotated = order[self._cursor :] + order[: self._cursor]
                else:
                    rotated = []
                # Admission is by free memory; utilisation only orders candidates,
                # so a shared card with enough memory is not starved forever.
                rotated.sort(key=lambda i: observed[i].utilization_percent if i in observed else 101)
                for index in rotated:
                    if index in self._busy:
                        continue
                    gpu = observed.get(index)
                    if gpu is None:
                        continue
                    if not gpu.telemetry_valid:
                        continue
                    if gpu.memory_free_mib < need:
                        continue
                    self._busy.add(index)
                    self._cursor = (order.index(index) + 1) % len(order)
                    return index
            stop.wait(self.poll_seconds)
        raise RuntimeError("stopped while waiting for a GPU")

    def release(self, index: int) -> None:
        with self._cv:
            self._busy.discard(index)
            self._cv.notify_all()

    def quarantine(self, index: int) -> None:
        with self._cv:
            if index in self.indices:
                self._busy.add(index)
