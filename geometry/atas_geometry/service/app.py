from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from atas_geometry.service.config import Settings
from atas_geometry.service.gpu_pool import GpuPool
from atas_geometry.service.jobs import GeometryRuntime
from atas_geometry.world.recapture import identifier
import hashlib

UPLOAD_CHUNK = 8 * 1024 * 1024


def _auth(request: Request, settings: Settings) -> None:
    if not settings.api_token:
        return
    header = request.headers.get("authorization") or ""
    if header != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _form_value(form: Any, key: str) -> str:
    value = form.get(key)
    if value is None:
        return ""
    return str(value)


async def _create_job(
    request: Request, settings: Settings, runtime: GeometryRuntime
) -> JSONResponse:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        kind = _form_value(form, "type") or "recon"
        payload: dict[str, Any] = {
            "type": kind,
            "run_id": _form_value(form, "run_id"),
            "story_id": _form_value(form, "story_id"),
            "shot_id": _form_value(form, "shot_id"),
            "scene_id": _form_value(form, "scene_id"),
            "event_id": _form_value(form, "event_id"),
            "duration": float(_form_value(form, "duration") or 0.0),
        }
        for key in ("run_id","story_id","shot_id","scene_id","event_id"):
            payload[key]=identifier(payload[key],key)
        registry=_form_value(form,"entity_registry")
        payload["entity_registry"]=json.loads(registry) if registry else []
        payload["retry_failed"]=_form_value(form,"retry_failed").lower() in {"true","1"}
        payload["publication_mode"]=_form_value(form,"publication_mode")
        if payload["publication_mode"] not in {"","reviewed_candidate_v1"}:
            raise ValueError("unknown publication mode")
        payload["review_lineage"]=json.loads(_form_value(form,"review_lineage") or "{}")
        world_time = _form_value(form, "world_time_start")
        if world_time:
            payload["world_time_start"] = float(world_time)
        raw_keywords = _form_value(form, "dynamic_keywords")
        if raw_keywords:
            decoded = json.loads(raw_keywords)
            if not isinstance(decoded,list) or not all(isinstance(x,str) for x in decoded):
                raise ValueError("dynamic_keywords must be a list of short nouns")
            payload["dynamic_keywords"] = decoded
        else:
            payload["dynamic_keywords"] = []
        extra = _form_value(form, "num_frames")
        if extra:
            payload["num_frames"] = int(extra)
        video = form.get("video")
        if video is None:
            raise HTTPException(status_code=400, detail="missing video part")
        job_dir = settings.hot_root / "jobs" / "incoming"
        job_dir.mkdir(parents=True, exist_ok=True)
        # Must include story_id: many stories recon shot_01 at once, and the
        # client always names the part shot_01_result.mp4.
        dest = job_dir / (
            f"{payload['run_id'] or 'run'}_{payload['story_id'] or 'story'}_"
            f"{payload['shot_id'] or 'clip'}_{uuid.uuid4().hex[:10]}.mp4"
        )
        digest=hashlib.sha256()
        with dest.open("wb") as handle:
            while True:
                chunk = await video.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
        payload["video_path"] = str(dest)
        payload["video_sha256"] = digest.hexdigest()
        from starlette.concurrency import run_in_threadpool
        job = await run_in_threadpool(runtime.submit, "recon", payload)
        return JSONResponse({"job_id": job["job_id"]})
    body = await request.json()
    kind = str(body.get("type") or "render")
    if kind != "render":
        raise HTTPException(status_code=400, detail="JSON jobs must be type=render")
    for key in ("run_id","story_id","shot_id","scene_id"):
        body[key]=identifier(body.get(key),key)
    from starlette.concurrency import run_in_threadpool
    job = await run_in_threadpool(runtime.submit, "render", body)
    return JSONResponse({"job_id": job["job_id"]})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.hot_root.mkdir(parents=True, exist_ok=True)
    pool = GpuPool(
        max_gpus=settings.max_gpus,
        configured=settings.gpu_ids,
        min_free_mib=settings.gpu_min_free_mib,
        poll_seconds=settings.gpu_poll_seconds,
        exclude_gpu0=settings.exclude_gpu0,
    )
    runtime = GeometryRuntime(settings, pool)
    app = FastAPI(title="GPU geometry service", version="0.1.0")
    app.state.settings = settings
    app.state.runtime = runtime

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        from atas_geometry.service.preflight import dependency_report
        report = dependency_report(settings)
        snap = pool.snapshot()
        body = {"ready": report["ok"], "dependencies": report["checks"],
                "render": True, "recapture": True, "semantic_camera": True,
                "reviewed_publication": True, "fusion_mode": "anchored", "event_memory": True,
                "resource_admission": {"recon_min_free_mib": max(settings.gpu_min_free_mib, settings.recon_min_free_mib),
                                       "render_min_free_mib": 2048},
                "details": snap}
        return JSONResponse(body, status_code=200 if report["ok"] else 503)

    @app.post("/v1/jobs")
    async def create_job(request: Request) -> JSONResponse:
        _auth(request, settings)
        try:
            return await _create_job(request, settings, runtime)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/v1/camera-context")
    async def get_camera_context(request: Request) -> dict[str, Any]:
        _auth(request, settings)
        from starlette.concurrency import run_in_threadpool
        try:
            payload = await request.json()
            return await run_in_threadpool(runtime.call, "camera_context", payload)
        except (ValueError, FileNotFoundError, KeyError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        _auth(request, settings)
        job = runtime.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job

    @app.post("/v1/world-publication")
    async def post_world_publication(request: Request):
        _auth(request,settings)
        from starlette.concurrency import run_in_threadpool
        payload=await request.json()
        def publish():
            for key in ("run_id","story_id","scene_id","shot_id","candidate_job_id"):
                identifier(payload.get(key),key)
            with runtime.scene_lock(payload["run_id"],payload["story_id"],payload["scene_id"]):
                return runtime.call("publish_candidate",payload)
        try:
            return await run_in_threadpool(publish)
        except (ValueError,FileNotFoundError,KeyError,RuntimeError,TimeoutError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/v1/world-rebase")
    async def post_world_rebase(request: Request):
        _auth(request, settings)
        from starlette.concurrency import run_in_threadpool
        from atas_geometry.world.publication_recovery import submit_rebase
        try:
            return await run_in_threadpool(submit_rebase, runtime, await request.json())
        except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/artifacts/memory_input.mp4")
    def memory_input(job_id: str, request: Request) -> FileResponse:
        _auth(request, settings)
        job = runtime.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        path = Path(str(job.get("memory_input") or runtime.store.artifact_path(job_id)))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not ready")
        return FileResponse(path, media_type="video/mp4", filename="memory_input.mp4")

    return app
