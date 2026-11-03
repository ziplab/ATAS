from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Literal

from atas.providers.wan.journal import DurableWanClient as Wan27Client
from atas.runner.recon_ledger import write_json_atomic
from atas.providers.settings import AVWMSettings
from atas.providers.wan.types import V2VRequest
Lane = Literal["prime", "video"]
GeometryControlMode = Literal["reference", "video_edit"]


class Wan30VideoEditClient(Wan27Client):
    """Experimental Wan3 video-edit payload; never selected by production default."""

    def _provider_output_duration(self, request, *, model, media):
        return -1

    def build_payload(self, request, references=None, duration=None, *, media_urls=None):
        payload = super().build_payload(
            request,
            references=references,
            duration=-1,
            media_urls=media_urls,
        )
        payload["parameters"]["ratio"] = "adaptive"
        payload["parameters"]["duration"] = -1
        return payload

    def _preflight(self, request, references, duration, payload):
        # Base validation needs the authored duration for its 2-15s range and
        # media compatibility checks. Provider-facing duration remains -1 so
        # video editing follows the reference video's length.
        super()._preflight(
            request,
            references,
            math.ceil(request.shot.duration),
            payload,
        )
        media = payload.get("input", {}).get("media", [])
        media_types = [item.get("type") for item in media if isinstance(item, dict)]
        if "reference_video" not in media_types:
            raise RuntimeError("video_edit requires provider reference_video")
        if "first_frame" in media_types or "last_frame" in media_types:
            raise RuntimeError("video_edit forbids strict first_frame/last_frame media")
        if payload.get("parameters", {}).get("ratio") != "adaptive":
            raise RuntimeError("video_edit requires ratio=adaptive")
        if payload.get("parameters", {}).get("duration") != -1:
            raise RuntimeError("video_edit requires duration=-1")


def _capacity_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "http 429",
            "throttl",
            "rate quota",
            "allocationquota",
            "concurrent",
            "too many requests",
        )
    )


def _inspection_error(exc: BaseException) -> bool:
    text = str(exc)
    lowered = text.lower()
    return (
        "DataInspectionFailed" in text
        or "green net" in lowered
        or "inappropriate content" in lowered
    )


class Wan30Router:
    """Fill prime first; overflow to wan3.0-video; refill prime as soon as a slot frees."""

    def __init__(self) -> None:
        self.prime_cap = max(0, int(os.environ.get("WAN30_PRIME_SLOTS", "5")))
        self.video_cap = max(0, int(os.environ.get("WAN30_VIDEO_SLOTS", "5")))
        self.prime_inflight = 0
        self.video_inflight = 0
        self._cooldown_until = {"prime": 0.0, "video": 0.0}
        self._cond = asyncio.Condition()

    def _open(self, lane: Lane) -> bool:
        if time.monotonic() < self._cooldown_until[lane]:
            return False
        if lane == "prime":
            return self.prime_inflight < self.prime_cap
        return self.video_inflight < self.video_cap

    async def acquire(self) -> Lane:
        async with self._cond:
            while True:
                if self._open("prime"):
                    self.prime_inflight += 1
                    return "prime"
                if self._open("video"):
                    self.video_inflight += 1
                    return "video"
                try:
                    await asyncio.wait_for(self._cond.wait(),timeout=1.)
                except asyncio.TimeoutError:
                    pass

    async def acquire_lane(self,lane:Lane)->Lane:
        async with self._cond:
            while True:
                count=self.prime_inflight if lane=="prime" else self.video_inflight
                cap=self.prime_cap if lane=="prime" else self.video_cap
                if count<max(1,cap):
                    if lane=="prime": self.prime_inflight+=1
                    else: self.video_inflight+=1
                    return lane
                try: await asyncio.wait_for(self._cond.wait(),timeout=1.)
                except asyncio.TimeoutError: pass

    async def release(self, lane: Lane) -> None:
        async with self._cond:
            if lane == "prime":
                self.prime_inflight = max(0, self.prime_inflight - 1)
            else:
                self.video_inflight = max(0, self.video_inflight - 1)
            self._cond.notify_all()

    def cooldown(self, lane: Lane, seconds: float = 20.0) -> None:
        self._cooldown_until[lane] = time.monotonic() + seconds

    def snapshot(self) -> dict[str, int]:
        return {
            "prime_inflight": self.prime_inflight,
            "video_inflight": self.video_inflight,
            "prime_cap": self.prime_cap,
            "video_cap": self.video_cap,
        }


_ROUTER: Wan30Router | None = None


def get_router() -> Wan30Router:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = Wan30Router()
    return _ROUTER


def model_for_lane(lane: Lane) -> str:
    if lane == "prime":
        return os.environ.get("WAN30_PRIME_MODEL", "wan3.0-video-prime")
    return os.environ.get("WAN30_VIDEO_MODEL", "wan3.0-video")


async def submit_on_lane(
    request: V2VRequest,
    lane: Lane,
    *,
    generate_audio: bool | None = None,
    geometry_control_mode: GeometryControlMode = "reference",
):
    if geometry_control_mode not in {"reference", "video_edit"}:
        raise ValueError(f"unknown geometry_control_mode={geometry_control_mode!r}")
    model = model_for_lane(lane)
    update: dict[str, object] = {
        "model_i2v": model,
        "model_r2v": model,
        "model_t2v": model,
    }
    if geometry_control_mode == "video_edit":
        update["ratio"] = "adaptive"
    if generate_audio is not None:
        update["generate_audio"] = generate_audio
    cfg = AVWMSettings().backends.wan27.model_copy(update=update)
    client_type = Wan30VideoEditClient if geometry_control_mode == "video_edit" else Wan27Client
    return await client_type(cfg).submit(request)


async def generate_routed(
    request: V2VRequest,
    *,
    geometry_control_mode: GeometryControlMode = "reference",
):
    return await _generate_routed(request, geometry_control_mode=geometry_control_mode)


async def _generate_routed(request, *, geometry_control_mode='reference'):
    router = get_router()
    last_error: BaseException | None = None
    mute = False
    mute_tries = 0
    route_file=request.output_dir/"wan_route_state.json"
    route=json.loads(route_file.read_text()) if route_file.is_file() else {}
    pinned=route.get("lane") if route.get("status")=="inflight" else None
    for _ in range(6):
        if pinned and route.get("model") != model_for_lane(pinned):
            raise RuntimeError("in-flight Wan model configuration changed; restore original model to resume")
        lane = await router.acquire_lane(pinned) if pinned else await router.acquire()
        model = model_for_lane(lane)
        write_json_atomic(route_file,{"lane":lane,"model":model,"status":"inflight"})
        try:
            result = await submit_on_lane(
                request,
                lane,
                generate_audio=False if mute else None,
                geometry_control_mode=geometry_control_mode,
            )
        except Exception as exc:  # noqa: BLE001
            await router.release(lane)
            if _capacity_error(exc):
                pinned=None
                write_json_atomic(route_file,{"lane":lane,"status":"rejected_capacity"})
                router.cooldown(lane)
                last_error = exc
                continue
            if _inspection_error(exc) and mute_tries < 2:
                if not mute:
                    print(
                        "Wan green-net failed on soundtrack; retrying muted",
                        flush=True,
                    )
                mute = True
                mute_tries += 1
                last_error = exc
                continue
            raise
        await router.release(lane)
        write_json_atomic(route_file,{"lane":lane,"model":model,"status":"succeeded"})
        lane_path = request.output_dir / "wan_lane.json"
        lane_path.write_text(
            json.dumps(
                {
                    "lane": lane,
                    "model": model,
                    "audio": False if mute else True,
                    "geometry_control_mode": geometry_control_mode,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result.artifact_path, lane, model
    raise RuntimeError(f"Wan 3.0 lanes exhausted: {last_error}") from last_error
