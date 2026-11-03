from __future__ import annotations

import json
import math
import os
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atas.config import ROOT


@dataclass(frozen=True)
class GeometryJob:
    job_id: str
    kind: str
    payload: dict[str, Any]


class GeometryWaitUncertain(RuntimeError):
    """Polling ended, but the remote job may still be running."""

    recon_job_may_still_be_running = True


class GeometryJobFailed(RuntimeError):
    """The geometry service reported a terminal failed state."""

    def __init__(self, message, *, result=None):
        super().__init__(message)
        result=result or {}
        self.error_code=str(result.get('error_code') or '')
        self.failure_category=str(result.get('failure_category') or '')
        self.retryable=result.get('retryable') is True
        self.resource_details=result.get('resource_details') or {}


DOLLY_SCALE = 0.25
ORBIT_SCALE = 1.0
ORBIT_DEGREES = 8.0
CONTINUE_MOVE_SCALE = DOLLY_SCALE

# Depth + a peek, then the opposite of each, so a long room does not walk off L1.
# Never truck / reveal / cinematic_arc — those leave a single Vista4D cone.
_CONTINUE_CYCLE = (
    ("pull_back", 0.0, DOLLY_SCALE),
    ("orbit", ORBIT_DEGREES, ORBIT_SCALE),
    ("push_in", 0.0, DOLLY_SCALE),
    ("arc_left", ORBIT_DEGREES, ORBIT_SCALE),
)

_MOVE_ALIASES = {
    "push": "push_in",
    "push_in": "push_in",
    "pushin": "push_in",
    "dolly_in": "push_in",
    "dollyin": "push_in",
    "前推": "push_in",
    "推进": "push_in",
    "pull": "pull_back",
    "pull_back": "pull_back",
    "pullback": "pull_back",
    "dolly_out": "pull_back",
    "dollyout": "pull_back",
    "拉远": "pull_back",
    "后拉": "pull_back",
    "orbit": "orbit",
    "arc": "orbit",
    "环绕": "orbit",
    "弧形": "orbit",
    "arc_left": "arc_left",
    "orbit_left": "arc_left",
    "左弧": "arc_left",
    "向左弧": "arc_left",
    "arc_right": "arc_right",
    "orbit_right": "arc_right",
    "右弧": "arc_right",
    "向右弧": "arc_right",
    "push_in_with_arc": "push_in_with_arc",
    "push_with_arc": "push_in_with_arc",
    "locked": "locked",
    "lock": "locked",
    "static": "locked",
    "fixed": "locked",
    "固定": "locked",
    "固定机位": "locked",
}
_GPU_EXECUTABLE_MOVES = frozenset(
    {"push_in", "pull_back", "orbit", "arc_left", "arc_right", "push_in_with_arc", "locked"}
)


def _normalise_move_name(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in _MOVE_ALIASES:
        return _MOVE_ALIASES[raw]
    # Preserve composite labels without turning an unrelated action such as a
    # truck, crane, or follow into a dolly/orbit.
    if ("push" in raw or "推进" in raw or "前推" in raw) and (
        "arc" in raw or "orbit" in raw or "弧" in raw or "环绕" in raw
    ):
        return "push_in_with_arc"
    if "pull_back" in raw or "pullback" in raw or "拉远" in raw or "后拉" in raw:
        return "pull_back"
    if "push_in" in raw or "dolly_in" in raw or "前推" in raw or "推进" in raw:
        return "push_in"
    if "arc_left" in raw or "orbit_left" in raw or "向左弧" in raw:
        return "arc_left"
    if "arc_right" in raw or "orbit_right" in raw or "向右弧" in raw:
        return "arc_right"
    if "orbit" in raw or raw == "arc" or "环绕" in raw or "弧形" in raw:
        return "orbit"
    if raw in {"locked", "lock", "static", "fixed"} or "固定机位" in raw:
        return "locked"
    return ""


def _vec3(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        try:
            return (float(value["x"]), float(value["y"]), float(value["z"]))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    return None


def _camera_path_move_name(camera_path: list[dict[str, Any]] | tuple) -> str:
    """Recover a named semantic from a dummy-room screenplay path.

    The coordinates never become Vista4D world coordinates. They are only a
    fallback semantic source when the screenplay omitted ``camera_move_name``.
    """

    points = [item for item in camera_path if isinstance(item, dict)]
    for item in points:
        for key in ("camera_move_name", "camera_movement", "move", "action", "type"):
            move = _normalise_move_name(item.get(key))
            if move:
                return move
    if len(points) < 2:
        return "locked" if points else ""
    start = _vec3(points[0].get("position"))
    end = _vec3(points[-1].get("position"))
    look = _vec3(points[0].get("look_at"))
    if start is None or end is None:
        return ""
    displacement = tuple(end[i] - start[i] for i in range(3))
    distance = math.sqrt(sum(value * value for value in displacement))
    if distance <= 1e-6:
        return "locked"
    if look is None:
        return ""
    forward = tuple(look[i] - start[i] for i in range(3))
    forward_norm = math.sqrt(sum(value * value for value in forward))
    if forward_norm <= 1e-6:
        return ""
    axial = sum(displacement[i] * forward[i] for i in range(3)) / forward_norm
    # Lateral paths need their own GPU primitive. Never silently call one an
    # orbit merely to keep the pipeline moving.
    if abs(axial) / distance >= 0.75:
        return "push_in" if axial > 0.0 else "pull_back"
    return ""


def compile_camera_intent(
    *,
    camera_move_name: str,
    camera_movement_degrees: float,
    camera_path: list[dict[str, Any]] | tuple = (),
    breath_index: int = 0,
    geometry_shots_in_scene: int = 2,
    scene_revisit: bool = False,
) -> dict[str, Any]:
    """Compile screenplay semantics to the GPU relative named-move API.

    Explicit screenplay intent wins over the old shot-index cycle. The cycle
    is now only an audited compatibility fallback when no intent was supplied.
    Unsupported explicit semantics are rejected, not silently substituted.
    """

    requested_name = str(camera_move_name or "").strip()
    move = _normalise_move_name(requested_name)
    source = "screenplay_camera_move_name" if move else ""
    if requested_name and not move:
        raise ValueError(
            f"unsupported screenplay camera move {requested_name!r}; "
            "refusing to replace it with a different movement semantic"
        )
    if not move and camera_path:
        move = _camera_path_move_name(camera_path)
        source = "screenplay_camera_path" if move else ""
        if not move:
            raise ValueError(
                "camera_path cannot be represented by the current GPU named-move API; "
                "refusing a semantic substitution"
            )
    fallback_used = not move
    fallback_reason = "screenplay supplied no camera movement or path" if fallback_used else None
    if fallback_used:
        fallback = continue_camera_plan(
            breath_index,
            geometry_shots_in_scene=geometry_shots_in_scene,
            scene_revisit=scene_revisit,
        )
        move = str(fallback["camera_move_name"])
        degrees = float(fallback["camera_movement_degrees"])
        scale = float(fallback["named_move_scale"])
        source = "legacy_continue_cycle_fallback"
    else:
        if move not in _GPU_EXECUTABLE_MOVES:
            raise ValueError(f"camera move {move!r} is not executable by the GPU API")
        requested_degrees = abs(float(camera_movement_degrees or 0.0))
        is_orbit = move in {"orbit", "arc_left", "arc_right", "push_in_with_arc"}
        degrees = requested_degrees if is_orbit and requested_degrees > 0.0 else (
            ORBIT_DEGREES if is_orbit else 0.0
        )
        scale = 0.0 if move == "locked" else (
            ORBIT_SCALE if move in {"orbit", "arc_left", "arc_right"} else DOLLY_SCALE
        )
    return {
        "camera_move_name": move,
        "camera_movement_degrees": degrees,
        "named_move_scale": scale,
        "camera_intent_source": source,
        "camera_intent_fallback_used": fallback_used,
        "camera_intent_fallback_reason": fallback_reason,
        "camera_semantic_preserved": not fallback_used,
    }


def continue_camera_plan(
    breath_index: int,
    *,
    geometry_shots_in_scene: int = 2,
    scene_revisit: bool = False,
) -> dict[str, float | str]:
    """Conservative continue/revisit move for GPU render.

    One geometry shot in the room: a small orbit so packing C shows occlusion
    instead of another dolly. Several shots: pull / right peek / push / left
    peek. Revisit keeps that beat so a return does not stack two yaws.
    """

    if int(geometry_shots_in_scene) <= 1:
        name, degrees, scale = ("orbit", ORBIT_DEGREES, ORBIT_SCALE)
    else:
        name, degrees, scale = _CONTINUE_CYCLE[int(breath_index) % 4]
    return {
        "camera_move_name": name,
        "camera_movement_degrees": float(degrees),
        "named_move_scale": float(scale),
    }


def render_payload(
    *,
    run_id: str,
    story_id: str,
    shot_id: str,
    scene_id: str,
    event_id: str,
    duration: float,
    fps: int,
    camera_move_name: str,
    camera_side_degrees: float,
    camera_movement_degrees: float,
    start_from_shot_id: str | None,
    start_from_role: str,
    scene_revisit: bool,
    geometry_layers: str,
    shot_size: str = "",
    camera_angle: str = "",
    shot_perspective: str = "",
    camera_path: list[dict[str, Any]] | tuple = (),
    target_entity_ids: tuple[str, ...] | list[str] = (),
    target_anchor: str = "center",
    breath_index: int = 0,
    geometry_shots_in_scene: int = 2,
    camera_relation: str = "",
    recapture: dict[str, Any] | None = None,
    semantic_camera: bool = False,
) -> dict[str, Any]:
    """Screenplay-directed move from the previous accepted shot's last L0 pose.

    ``start_from_shot_id`` / ``previous_shot_id`` is that clip (continue: the
    shot before this one; revisit: last clip in this room). Dummy YAML xyz is
    only a semantic template. A legacy room beat is used only when the plan
    has no move at all. Pivot is GPU-side (``resolve_subject_anchor``).
    """

    previous = str(start_from_shot_id or "").strip() or None
    plan = compile_camera_intent(
        camera_move_name=camera_move_name,
        camera_movement_degrees=camera_movement_degrees,
        camera_path=camera_path,
        breath_index=breath_index,
        geometry_shots_in_scene=geometry_shots_in_scene,
        scene_revisit=scene_revisit,
    ) if not semantic_camera else {
        "camera_move_name": str(camera_move_name or "locked"),
        "camera_movement_degrees": float(camera_movement_degrees or 0), "named_move_scale": 1.,
        "camera_intent_source": "visual_planner_pending", "camera_intent_fallback_used": False,
        "camera_intent_fallback_reason": None, "camera_semantic_preserved": True,
    }
    gpu_move = str(plan["camera_move_name"])
    return {
        "type": "render",
        "run_id": run_id,
        "story_id": story_id,
        "shot_id": shot_id,
        "scene_id": scene_id,
        "event_id": event_id,
        "duration": duration,
        "fps": fps,
        "camera_move_name": gpu_move,
        "screenplay_camera_move_name": str(camera_move_name or ""),
        "screenplay_camera_side_degrees": float(camera_side_degrees or 0.0),
        "screenplay_camera_movement_degrees": float(camera_movement_degrees or 0.0),
        "camera_side_degrees": float(camera_side_degrees or 0.0) if recapture else 0.0,
        "camera_movement_degrees": float(plan["camera_movement_degrees"]),
        "previous_shot_id": previous,
        "start_from_shot_id": previous,
        "start_from_role": "source_event" if recapture else start_from_role,
        "scene_revisit": scene_revisit,
        "geometry_layers": geometry_layers,
        "camera_relation": str(camera_relation or "continuous"),
        "shot_size": shot_size,
        "camera_angle": camera_angle,
        "shot_perspective": shot_perspective,
        "camera_path_frame": "relative_named_move",
        "camera_path_template": list(camera_path),
        "named_move_scale": 1.0 if recapture else float(plan["named_move_scale"]),
        "camera_intent_source": plan["camera_intent_source"],
        "camera_intent_fallback_used": bool(plan["camera_intent_fallback_used"]),
        "camera_intent_fallback_reason": plan["camera_intent_fallback_reason"],
        "camera_semantic_preserved": bool(plan["camera_semantic_preserved"]),
        "camera_safety_policy": "shrink_or_reject_same_semantic",
        "allow_camera_semantic_substitution": False,
        "start_from_camera": "source_time_start" if recapture else "last",
        "target_entity_ids": [str(item) for item in target_entity_ids if str(item).strip()],
        "target_anchor": str(target_anchor or "center").strip() or "center",
        "breath_index": int(breath_index),
        **(recapture or {}),
    }


def named_move_scale(shot_size: str = "") -> float:
    """Default dolly scale; orbit shots use ``continue_camera_plan`` instead."""

    return DOLLY_SCALE


def render_chain_status(job: dict[str, Any]) -> dict[str, Any]:
    """Fields the GPU returns on a succeeded render. Nested ``report`` or top-level."""

    report = job.get("report")
    if not isinstance(report, dict):
        report = {}
    precheck = report.get("camera_precheck")
    if not isinstance(precheck, dict):
        precheck = job.get("camera_precheck") if isinstance(job.get("camera_precheck"), dict) else {}

    def pick(*keys: str) -> Any:
        for source in (job, report, precheck):
            for key in keys:
                if key in source and source[key] is not None:
                    return source[key]
        return None

    raw_fallback = pick("chained_fallback_to_home")
    if isinstance(raw_fallback, str):
        fallback = raw_fallback.strip().lower() in {"1", "true", "yes"}
    else:
        fallback = bool(raw_fallback)
    role = str(pick("start_from_role") or "").strip()
    return {
        "chained_fallback_to_home": fallback,
        "start_from_role": role or None,
        "start_from_shot_id": pick("start_from_shot_id"),
        "executed_camera_move_name": pick("camera_move_name"),
        "precheck_motion_scale": precheck.get("motion_scale"),
        "mean_coverage": pick("mean_coverage"),
    }


def chain_silently_broke(status: dict[str, Any]) -> bool:
    """Home start for shot_02 is ``scene_home`` and must not abort.

    ``chained_fallback_to_home`` on any other role means last_camera was
    missing and the pose jumped to that scene's Home.
    """

    if not status.get("chained_fallback_to_home"):
        return False
    return str(status.get("start_from_role") or "") != "scene_home"


class GeometryClient:
    """HTTP client for the GPU machine. This host never listens."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("GEOMETRY_SERVICE_URL") or ""
        ).rstrip("/")
        self._endpoint_file = None if base_url is not None else Path(os.environ.get(
            "GEOMETRY_ENDPOINT_FILE", str(ROOT / "geometry_endpoint.json")))
        self._refresh_endpoint()
        self.token = token or os.environ.get("GEOMETRY_API_TOKEN") or ""
        flag = os.environ.get("GEOMETRY_ENABLED", "false").strip().lower()
        self.enabled = enabled if enabled is not None else flag in {"1", "true", "yes"}

    def _refresh_endpoint(self) -> None:
        """Reload an operator-published endpoint without losing durable job IDs.

        This is not automatic quick-tunnel discovery. The local control file must
        be updated by an authenticated operator/publisher after a tunnel rotation.
        """
        path = self._endpoint_file
        if path is None or not path.is_file():
            return
        from urllib.parse import urlsplit
        value = json.loads(path.read_text()).get("url", "")
        url = urlsplit(value)
        local_http = url.scheme == "http" and url.hostname in {"127.0.0.1", "::1"}
        if ((url.scheme != "https" and not local_http) or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("invalid geometry endpoint control file")
        self.base_url = value.rstrip("/")

    def _headers(self, content_type: str | None = "application/json") -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 60,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._refresh_endpoint()
        if not self.base_url:
            raise RuntimeError("GEOMETRY_SERVICE_URL is empty")
        body = data
        if body is None and payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers or self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"geometry HTTP {exc.code}: {detail}") from exc
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            # A proxy/tunnel error page: the service is unreachable, not wrong.
            raise ConnectionError(f"geometry returned a non-JSON body: {raw[:200]!r}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("geometry returned a non-object JSON body")
        return decoded

    def healthz(self) -> dict[str, Any]:
        return self._request("GET", "/healthz")

    def readyz(self) -> dict[str, Any]:
        return self._request("GET", "/readyz")

    def camera_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        # This read-only operation may scan a cold scene. A short client timeout
        # can abandon still-running server work; allow one bounded longer call.
        timeout = float(os.environ.get("AVWM_CAMERA_CONTEXT_TIMEOUT_SECONDS", "300"))
        if not math.isfinite(timeout) or not 60 <= timeout <= 600:
            raise ValueError("camera context timeout must be finite and between 60 and 600 seconds")
        return self._request("POST", "/v1/camera-context", payload, timeout=timeout)

    def supports_render(self) -> bool:
        if not self.enabled:
            return False
        try:
            payload = self.readyz()
        except Exception:
            return False
        return bool(payload.get("render") or payload.get("details", {}).get("render"))

    def recon(
        self,
        *,
        run_id: str,
        story_id: str,
        shot_id: str,
        scene_id: str,
        event_id: str,
        duration: float,
        video: Path,
        dynamic_keywords: tuple[str, ...] = (),
        entity_registry: list[dict[str, Any]] | None = None,
        publication_mode: str = "",
        review_lineage: dict[str, Any] | None = None,
        world_time_start: float | None = None,
    ) -> GeometryJob:
        payload = {
            "type": "recon",
            "run_id": run_id,
            "story_id": story_id,
            "shot_id": shot_id,
            "scene_id": scene_id,
            "event_id": event_id,
            "duration": duration,
            "video": str(video),
            "dynamic_keywords": list(dynamic_keywords),
            "entity_registry": entity_registry or [],
            "retry_failed": True,
            "publication_mode": publication_mode,
            "review_lineage": review_lineage or {},
        }
        if world_time_start is not None:
            payload["world_time_start"] = float(world_time_start)
        if not self.enabled:
            return GeometryJob(job_id="dry-recon", kind="recon", payload=payload)
        response = self._submit_with_video("/v1/jobs", payload, video)
        return GeometryJob(str(response.get("job_id") or ""), "recon", payload)

    def render(self, payload: dict[str, Any]) -> GeometryJob:
        if not self.enabled:
            return GeometryJob(job_id="dry-render", kind="render", payload=payload)
        response = self._request("POST", "/v1/jobs", {**payload,"retry_failed":True})
        return GeometryJob(str(response.get("job_id") or ""), "render", payload)

    def wait(self, job_id: str, *, interval: float = 5.0, timeout: float | None = None,
             service_grace: float = 120.0) -> dict[str, Any]:
        """Poll a job to completion.

        Queueing for a GPU is normal operation when several stories share the
        service, so it is waited out like any running job; only an unreachable
        service (beyond ``service_grace``) or the overall timeout ends the wait
        early, leaving the job id intact for a later resume.
        """
        if timeout is None:
            timeout = float(os.environ.get("ATAS_GEOMETRY_WAIT_SECONDS", str(4 * 3600)))
        if job_id.startswith("dry-"):
            return {"status": "skipped", "job_id": job_id}
        deadline = time.monotonic() + timeout
        unavailable_since = None

        def unavailable(exc):
            nonlocal unavailable_since
            now=time.monotonic()
            if unavailable_since is None:unavailable_since=now
            if now >= deadline or now-unavailable_since >= service_grace:
                # The same remote job may still exist. Let the outer scheduler
                # advance independent scenes, then resume this exact job ID.
                raise GeometryWaitUncertain(
                    f"geometry job {job_id} polling unavailable; preserving its ID for deferred recovery"
                ) from exc
            time.sleep(max(0.,min(max(interval,1.),deadline-now,
                                 service_grace-(now-unavailable_since))))

        while True:
            try:
                job = self._request("GET", f"/v1/jobs/{job_id}")
            except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
                # ConnectionError: a proxy/tunnel error page instead of JSON.
                unavailable(exc)
                continue
            except RuntimeError as exc:
                detail = str(exc)
                if "HTTP 404" in detail:
                    raise GeometryWaitUncertain(
                        f"geometry job {job_id} temporarily missing; preserving its ID for reconciliation"
                    ) from exc
                if any(token in detail for token in ("HTTP 502", "HTTP 503", "HTTP 504", "HTTP 530", "HTTP 520", "HTTP 522", "HTTP 524")):
                    unavailable(exc)
                    continue
                raise
            unavailable_since = None
            status = str(job.get("status") or "")
            if status in {"succeeded", "failed", "skipped"}:
                if status == "failed":
                    raise GeometryJobFailed(
                        job.get("error") or f"geometry job {job_id} failed", result=job
                    )
                return job
            if time.monotonic() >= deadline:
                raise GeometryWaitUncertain(f"geometry job {job_id} timed out")
            time.sleep(interval)

    def fetch_memory_input(self, job: dict[str, Any], dest: Path) -> Path | None:
        self._refresh_endpoint()
        dest.parent.mkdir(parents=True, exist_ok=True)
        job_id = str(job.get("job_id") or "")
        # A service filesystem path belongs to the GPU host. Prefer its
        # authenticated artifact route without ever stat-ing that path on the client machine.
        if job_id and self.base_url:
            receipt = dest.with_name(dest.name + '.artifact.json')
            from atas.review.film_review import file_hash
            from atas.runner.recon_ledger import write_json_atomic
            try:
                cached = json.loads(receipt.read_text())
                if (cached.get('job_id') == job_id and cached.get('sha256') == file_hash(dest)):
                    return dest
            except (OSError, ValueError):
                pass
            try:
                started = time.perf_counter()
                self._download(
                    f"{self.base_url}/v1/jobs/{job_id}/artifacts/memory_input.mp4",
                    dest,
                )
                write_json_atomic(receipt, {'job_id':job_id,'sha256':file_hash(dest),
                    'bytes':dest.stat().st_size,'download_seconds':time.perf_counter()-started})
                return dest
            except RuntimeError:
                return None
        candidate = Path(str(job.get("memory_input") or ""))
        try:
            local = candidate.is_file()
        except OSError:
            local = False
        if local:
            if candidate.resolve() != dest.resolve():
                dest.write_bytes(candidate.read_bytes())
            return dest
        # Never send the geometry bearer token to an arbitrary artifact host.
        url = str(job.get("memory_input_url") or job.get("memory_input") or "")
        if url.startswith(self.base_url + "/") and self.base_url:
            self._download(url, dest)
            return dest
        return None

    def _submit_with_video(
        self, path: str, payload: dict[str, Any], video: Path
    ) -> dict[str, Any]:
        if not video.is_file():
            raise RuntimeError(f"recon video missing: {video}")
        boundary = uuid.uuid4().hex
        chunks: list[bytes] = []
        for key, value in payload.items():
            encoded = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            chunks.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                    f"{encoded}\r\n"
                ).encode("utf-8")
            )
        filename = video.name
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
                "Content-Type: video/mp4\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(video.read_bytes())
        chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(chunks)
        headers = self._headers(f"multipart/form-data; boundary={boundary}")
        # The service cannot read a client-local path. On an uncertain upload
        # response, resume with the same content; server deduplication retains
        # the original job. Never replace this with an invalid JSON recon.
        return self._request(
            "POST", path, None, timeout=600, data=body, headers=headers
        )

    def _download(self, url: str, dest: Path) -> None:
        headers = self._headers(None)
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                partial=dest.with_name(dest.name+".partial")
                # Bounded buffers, early disk writes and observable partial
                # growth; never hold the whole video in memory while waiting.
                with partial.open('wb') as output:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        output.write(chunk)
                import cv2
                capture=cv2.VideoCapture(str(partial))
                count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); first_ok,_=capture.read()
                capture.set(cv2.CAP_PROP_POS_FRAMES,max(0,count-1)); last_ok,_=capture.read()
                capture.release()
                if count<=0 or not first_ok or not last_ok:
                    raise RuntimeError("geometry download is not a complete decodable MP4")
                partial.replace(dest)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"geometry download HTTP {exc.code}: {detail}") from exc
