from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atas_geometry.memory.fusion import FusedSceneMemory
from atas_geometry.memory.observation import CameraCalibration
from atas_geometry.memory.types import AnyArray
@dataclass(frozen=True)
class CameraQuery:
    """A model-independent target-camera query over world time."""

    world_times: tuple[float, ...]
    cameras: tuple[CameraCalibration, ...]
    mode: str
    focus_world: AnyArray

    def __post_init__(self) -> None:
        if not self.world_times:
            raise ValueError("camera query needs at least one world time")
        if len(self.world_times) != len(self.cameras):
            raise ValueError("world_times and cameras must have equal length")
        if tuple(sorted(self.world_times)) != self.world_times:
            raise ValueError("world_times must be monotonic")
        if self.focus_world.shape != (3,) or not np.isfinite(self.focus_world).all():
            raise ValueError("focus_world must be a finite 3D point")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "avwm_camera_query",
            "mode": self.mode,
            "focus_world": self.focus_world.tolist(),
            "frames": [
                {
                    "world_time": world_time,
                    "camera": camera.to_json_dict(),
                }
                for world_time, camera in zip(self.world_times, self.cameras, strict=True)
            ],
            "summary": self.summary(),
        }
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CameraQuery:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("type") != "avwm_camera_query":
            raise ValueError(f"Not an AVWM camera query: {path}")
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Camera query contains no frames: {path}")
        world_times: list[float] = []
        cameras: list[CameraCalibration] = []
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("camera"), dict):
                raise ValueError(f"Invalid camera query frame: {frame!r}")
            camera = frame["camera"]
            world_times.append(float(frame["world_time"]))
            cameras.append(
                CameraCalibration(
                    camera_id=str(camera["camera_id"]),
                    world_to_camera=np.asarray(camera["world_to_camera"], dtype=np.float64),
                    intrinsic=np.asarray(camera["intrinsic"], dtype=np.float64),
                    image_size=(int(camera["image_size"][0]), int(camera["image_size"][1])),
                )
            )
        return cls(
            world_times=tuple(world_times),
            cameras=tuple(cameras),
            mode=str(payload.get("mode", "custom")),
            focus_world=np.asarray(payload["focus_world"], dtype=np.float64),
        )

    def summary(self) -> dict[str, object]:
        centers = np.stack([camera.center_world for camera in self.cameras])
        travel = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        # A query may pan from a fixed camera position.  The shared focus point
        # is a useful scene anchor, but it cannot describe that rotation.
        first_view = _normalize(self.cameras[0].camera_to_world[:3, 2])
        last_view = _normalize(self.cameras[-1].camera_to_world[:3, 2])
        view_angle = math.degrees(
            math.acos(float(np.clip(np.dot(first_view, last_view), -1.0, 1.0)))
        )
        distances = np.linalg.norm(centers - self.focus_world[None], axis=1)
        return {
            "mode": self.mode,
            "frame_count": len(self.world_times),
            "world_time_range": [self.world_times[0], self.world_times[-1]],
            "focus_world": self.focus_world.tolist(),
            "camera_travel": travel,
            "view_angle_degrees": view_angle,
            "focus_distance_range": [float(distances.min()), float(distances.max())],
        }


def estimate_memory_focus(
    memory: FusedSceneMemory,
    *,
    foreground_threshold: float = 0.5,
    min_points: int = 100,
) -> AnyArray:
    """Estimate a stable subject focus without assuming a dataset coordinate system."""
    finite = np.isfinite(memory.xyz).all(axis=1)
    foreground = finite & (memory.foreground >= foreground_threshold)
    selected = foreground if np.count_nonzero(foreground) >= min_points else finite
    if not np.any(selected):
        raise ValueError("memory has no finite points for camera focus estimation")
    return np.asarray(
        np.median(memory.xyz[selected].astype(np.float64), axis=0),
        dtype=np.float64,
    )


def generate_camera_query(
    memory: FusedSceneMemory,
    world_times: tuple[float, ...],
    reference_camera: CameraCalibration,
    *,
    mode: str = "orbit",
    orbit_degrees: float = 70.0,
    travel_ratio: float = 0.35,
    lift_ratio: float = 0.12,
    truck_ratio: float = 0.08,
    azimuth_degrees: float = 0.0,
    focus_world: AnyArray | None = None,
) -> CameraQuery:
    """Generate a scale-relative camera trajectory around the 4D memory.

    ``azimuth_degrees`` yaws the whole path around the look-at before the named
    move. Recapture uses this for planner ``side_degrees`` instead of story-bbox xyz.
    """
    supported = {"locked", "orbit", "dolly", "truck", "crane", "cinematic_arc"}
    if mode not in supported:
        raise ValueError(f"Unsupported camera mode {mode!r}; expected one of {sorted(supported)}")
    if not world_times:
        raise ValueError("camera query needs at least one world time")
    if not np.isfinite(orbit_degrees):
        raise ValueError("orbit_degrees must be finite")
    if not np.isfinite(travel_ratio):
        raise ValueError("travel_ratio must be finite")
    if not np.isfinite(lift_ratio) or not np.isfinite(truck_ratio):
        raise ValueError("lift_ratio and truck_ratio must be finite")
    if not np.isfinite(azimuth_degrees):
        raise ValueError("azimuth_degrees must be finite")

    focus = (
        estimate_memory_focus(memory)
        if focus_world is None
        else np.asarray(focus_world, dtype=np.float64)
    )
    if focus.shape != (3,) or not np.isfinite(focus).all():
        raise ValueError("focus_world must be a finite 3D point")

    reference_c2w = reference_camera.camera_to_world.astype(np.float64)
    start = reference_c2w[:3, 3]
    offset = start - focus
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-6:
        raise ValueError("reference camera is too close to the estimated memory focus")
    up = _normalize(-reference_c2w[:3, 1])
    if abs(float(azimuth_degrees)) > 1e-6:
        offset = _rotate_axis(offset, up, math.radians(float(azimuth_degrees)))
        start = focus + offset
        distance = float(np.linalg.norm(offset))
    forward = _normalize(focus - start)
    start_c2w = _look_at_camera(start, focus, up)
    right = _normalize(start_c2w[:3, 0])

    cameras: list[CameraCalibration] = []
    for index in range(len(world_times)):
        linear = 0.0 if len(world_times) == 1 else index / (len(world_times) - 1)
        progress = 0.5 - 0.5 * math.cos(math.pi * linear)
        if mode == "orbit":
            position = focus + _rotate_axis(offset, up, math.radians(orbit_degrees) * progress)
        elif mode == "cinematic_arc":
            rotated = _rotate_axis(offset, up, math.radians(orbit_degrees) * progress)
            radial_scale = 1.0 - travel_ratio * (
                0.65 * progress + 0.35 * math.sin(math.pi * progress)
            )
            lift = up * (distance * lift_ratio * math.sin(math.pi * progress))
            truck = right * (distance * truck_ratio * math.sin(math.pi * progress))
            position = focus + rotated * radial_scale + lift + truck
        elif mode == "dolly":
            position = start + forward * (distance * travel_ratio * progress)
        elif mode == "truck":
            position = start + right * (distance * travel_ratio * progress)
        elif mode == "crane":
            position = start + up * (distance * travel_ratio * progress)
        else:
            position = start.copy()
        c2w = _look_at_camera(position, focus, up)
        cameras.append(
            CameraCalibration(
                camera_id=f"{mode}_{index:04d}",
                world_to_camera=np.linalg.inv(c2w),
                intrinsic=reference_camera.intrinsic.astype(np.float64, copy=True),
                image_size=reference_camera.image_size,
            )
        )
    return CameraQuery(
        world_times=world_times,
        cameras=tuple(cameras),
        mode=mode,
        focus_world=focus,
    )


MIN_END_DISTANCE_RATIO = 0.45
MIN_SURFACE_CLEARANCE_RATIO = 0.08
PRECHECK_SAMPLE_POINTS = 100_000
MAX_TRAVEL_SHRINK_ATTEMPTS = 3
SAME_SEMANTIC_SAFETY_POLICY = "shrink_or_reject_same_semantic"


def resolve_camera_safety_contract(payload: Mapping[str, object]) -> tuple[str, bool]:
    """Return safety policy and whether a named move may become another move.

    Missing fields deliberately preserve the legacy locked-pose fallback.
    The same-semantic policy is authoritative even if a contradictory caller
    also sends ``allow_camera_semantic_substitution=true``.
    """
    policy = str(payload.get("camera_safety_policy") or "legacy_locked_fallback")
    raw = payload.get("allow_camera_semantic_substitution")
    if raw is None:
        allow_substitution = True
    elif isinstance(raw, bool):
        allow_substitution = raw
    else:
        normalized = str(raw).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            allow_substitution = True
        elif normalized in {"0", "false", "no", "off"}:
            allow_substitution = False
        else:
            raise ValueError(f"invalid boolean value {raw!r}")
    if policy == SAME_SEMANTIC_SAFETY_POLICY:
        allow_substitution = False
    return policy, allow_substitution


def sparse_geometry_sample(
    memory: FusedSceneMemory,
    *,
    max_points: int = PRECHECK_SAMPLE_POINTS,
) -> AnyArray:
    """A deterministic thin sample of static geometry for cheap collision tests."""

    points = np.asarray(memory.xyz, dtype=np.float64)
    if points.shape[0] <= max_points:
        return points
    take = np.linspace(0, points.shape[0] - 1, max_points)
    return points[np.rint(take).astype(np.int64)]


def check_camera_query(
    query: CameraQuery,
    geometry: AnyArray,
    *,
    min_end_distance_ratio: float = MIN_END_DISTANCE_RATIO,
    min_surface_clearance_ratio: float = MIN_SURFACE_CLEARANCE_RATIO,
) -> dict[str, object]:
    """Reject trajectories that pass through the subject or sit inside geometry.

    Runs on a thin point sample so it costs milliseconds. The expensive render
    is 1080p times the shot length; discovering there that the camera drove
    through the cast costs twenty minutes and, if submitted, a paid Wan job.
    """
    focus = np.asarray(query.focus_world, dtype=np.float64)
    centers = np.stack([camera.center_world for camera in query.cameras]).astype(np.float64)
    distances = np.linalg.norm(centers - focus, axis=1)
    start_distance = float(distances[0])
    end_distance = float(distances[-1])
    closest_to_focus = float(distances.min())

    failures: list[str] = []
    if start_distance > 1e-9:
        end_ratio = closest_to_focus / start_distance
        if end_ratio < float(min_end_distance_ratio):
            failures.append(
                f"camera approaches within {end_ratio:.2f} of the subject distance "
                f"(minimum {float(min_end_distance_ratio):.2f}); "
                "the move passes through the subject"
            )
    else:
        end_ratio = 0.0
        failures.append("start camera coincides with the subject anchor")

    clearance = float("inf")
    points = np.asarray(geometry, dtype=np.float64)
    if points.size:
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
    if points.size:
        # Nearest surface to any pose on the path. Chunked so a large L1 does
        # not materialise a poses x points distance matrix.
        for start in range(centers.shape[0]):
            deltas = points - centers[start]
            squared = float(np.min(np.einsum("ij,ij->i", deltas, deltas)))
            clearance = min(clearance, float(np.sqrt(squared)))
        required = start_distance * float(min_surface_clearance_ratio)
        if clearance < required:
            failures.append(
                f"camera passes within {clearance:.4f} of a static surface "
                f"(minimum {required:.4f}); the path is inside geometry"
            )

    return {
        "passed": not failures,
        "failures": failures,
        "start_focus_distance": start_distance,
        "end_focus_distance": end_distance,
        "closest_focus_distance": closest_to_focus,
        "closest_focus_ratio": float(end_ratio),
        "surface_clearance": None if not np.isfinite(clearance) else float(clearance),
        "camera_travel": float(query.summary()["camera_travel"]),
    }




class CameraQueryRejected(RuntimeError):
    """A named camera move stayed unsafe after all permitted shrink attempts."""

    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        super().__init__(str(report.get("reason") or "camera query rejected"))


def resolve_checked_camera_query(
    builder: Callable[[float], CameraQuery],
    geometry: AnyArray,
    *,
    attempts: int = MAX_TRAVEL_SHRINK_ATTEMPTS,
    allow_semantic_substitution: bool = True,
) -> tuple[CameraQuery, dict[str, object]]:
    """Build a trajectory, shrinking it until it clears the scene geometry.

    Returns the accepted query plus a report suitable for
    ``geometry_report.json``. Legacy callers retain the locked-pose fallback.
    A caller enforcing a same-semantic director contract can disable that
    substitution; an unsafe move then raises :class:`CameraQueryRejected`
    with the complete precheck report instead of silently changing the shot.
    """
    tried: list[dict[str, object]] = []
    scale = 1.0
    for _ in range(max(1, int(attempts))):
        query = builder(scale)
        report = check_camera_query(query, geometry)
        tried.append({"motion_scale": scale, **report})
        if report["passed"]:
            return query, {
                "passed": True,
                "motion_scale": scale,
                "degraded_to_locked": False,
                "rejected": False,
                "shrink_attempts": len(tried) - 1,
                "attempts": tried,
            }
        scale *= 0.5
    if not allow_semantic_substitution:
        final_failures = tried[-1].get("failures") or []
        detail = "; ".join(str(item) for item in final_failures)
        reason = "named move remained unsafe after all same-semantic shrink attempts"
        if detail:
            reason = f"{reason}: {detail}"
        rejection = {
            "passed": False,
            "motion_scale": float(tried[-1]["motion_scale"]),
            "degraded_to_locked": False,
            "rejected": True,
            "shrink_attempts": len(tried) - 1,
            "attempts": tried,
            "reason": reason,
        }
        raise CameraQueryRejected(rejection)
    locked = builder(0.0)
    locked_report = check_camera_query(locked, geometry)
    tried.append({"motion_scale": 0.0, **locked_report})
    return locked, {
        "passed": bool(locked_report["passed"]),
        "motion_scale": 0.0,
        "degraded_to_locked": True,
        "rejected": False,
        "shrink_attempts": len(tried) - 1,
        "attempts": tried,
        "reason": "named move could not clear scene geometry at any scale",
    }


REOBSERVE_MOVE_PRESETS: dict[str, dict[str, float | str]] = {
    "orbit": {
        "mode": "orbit",
        "orbit_degrees": 80.0,
        "travel_ratio": 0.0,
        "lift_ratio": 0.12,
        "truck_ratio": 0.0,
    },
    "cinematic_arc": {
        "mode": "cinematic_arc",
        "orbit_degrees": 70.0,
        "travel_ratio": 0.32,
        "lift_ratio": 0.18,
        "truck_ratio": 0.16,
    },
    "reveal_right": {
        "mode": "truck",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.70,
        "lift_ratio": 0.08,
        "truck_ratio": 0.0,
    },
    "locked": {
        "mode": "locked",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.0,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "push_in": {
        "mode": "dolly",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.28,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "pull_back": {
        "mode": "dolly",
        "orbit_degrees": 0.0,
        "travel_ratio": -0.22,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "track_left": {
        "mode": "truck",
        "orbit_degrees": 0.0,
        "travel_ratio": -0.35,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "track_right": {
        "mode": "truck",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.35,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "arc_left": {
        "mode": "orbit",
        "orbit_degrees": -40.0,
        "travel_ratio": 0.0,
        "lift_ratio": 0.08,
        "truck_ratio": 0.0,
    },
    "arc_right": {
        "mode": "orbit",
        "orbit_degrees": 40.0,
        "travel_ratio": 0.0,
        "lift_ratio": 0.08,
        "truck_ratio": 0.0,
    },
    "push_in_with_arc": {
        "mode": "cinematic_arc",
        "orbit_degrees": 25.0,
        "travel_ratio": 0.22,
        "lift_ratio": 0.08,
        "truck_ratio": 0.10,
    },
    "follow": {
        "mode": "cinematic_arc",
        "orbit_degrees": 35.0,
        "travel_ratio": 0.18,
        "lift_ratio": 0.10,
        "truck_ratio": 0.08,
    },
    "dolly": {
        "mode": "dolly",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.28,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "truck": {
        "mode": "truck",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.35,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
    "crane": {
        "mode": "crane",
        "orbit_degrees": 0.0,
        "travel_ratio": 0.22,
        "lift_ratio": 0.0,
        "truck_ratio": 0.0,
    },
}


def _look_at_camera(position: AnyArray, focus: AnyArray, preferred_up: AnyArray) -> AnyArray:
    forward = _normalize(focus - position)
    right = np.cross(forward, preferred_up)
    if np.linalg.norm(right) <= 1e-6:
        fallback = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(forward, fallback)
    if np.linalg.norm(right) <= 1e-6:
        right = np.cross(forward, np.asarray([1.0, 0.0, 0.0]))
    right = _normalize(right)
    down = _normalize(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack((right, down, forward), axis=1)
    c2w[:3, 3] = position
    return c2w


def _rotate_axis(vector: AnyArray, axis: AnyArray, angle: float) -> AnyArray:
    axis = _normalize(axis)
    rotated: AnyArray = (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
    )
    return rotated


def _normalize(vector: AnyArray) -> AnyArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    normalized: AnyArray = np.asarray(vector, dtype=np.float64) / norm
    return normalized
