from __future__ import annotations

import json
import threading
import time
import traceback
import os
import dataclasses
import uuid
from pathlib import Path
from typing import Any

from atas_geometry.memory.camera_query import CameraQuery
from atas_geometry.service.config import Settings
from atas_geometry.memory.fuse import fuse_l0_to_l1
from atas_geometry.service.gpu_pool import GpuPool
from atas_geometry.service.paths import (
    l0_dir,
    l1_path,
    last_camera_path,
    remember_home_shot,
    resolve_home_shot,
    scene_key,
)
from atas_geometry.reconstruct.vista4d import probe_frame_count, run_recon_and_seg
from atas_geometry.service.durable import JobStore as DurableJobStore, atomic_json, file_hash
from atas_geometry.service.paths import scene_dir
from atas_geometry.world.writeback import writeback_world
JobStore = DurableJobStore


def _recon_phases(log_path: Path) -> list[dict[str, Any]]:
    """Per-phase seconds and self-reported GPU peak from the Vista4D recon log."""
    phases = []
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if line.startswith("AVWM_RECON_PHASE "):
                phases.append(json.loads(line[len("AVWM_RECON_PHASE "):]))
    except (OSError, ValueError):
        pass
    return phases


class GeometryRuntime:
    def __init__(self, settings: Settings, pool: GpuPool) -> None:
        self.settings = settings
        self.pool = pool
        self.store = JobStore(settings.hot_root)
        self._stop = threading.Event()
        self._queue: list[str] = self.store.pending()
        self._cv = threading.Condition()
        self._scene_locks: dict[str, threading.Lock] = {}
        self._scene_guard = threading.Lock()
        self._reconcile_prior_workers()
        if settings.archive_root is not None:
            from atas_geometry.world.archive import reconcile_archives
            threading.Thread(target=reconcile_archives, args=(settings.hot_root, settings.archive_root),
                             name="geometry-archive-reconcile", daemon=True).start()
        for index in range(settings.max_gpus):
            thread = threading.Thread(
                target=self._loop, name=f"geometry-worker-{index}", daemon=True
            )
            thread.start()

    def _reconcile_prior_workers(self):
        from atas_geometry.service.worker_protocol import reconcile_lease, lease_running
        for path in (self.settings.hot_root / "jobs").glob("*/worker_lease.json"):
            lease = json.loads(path.read_text())
            if not lease.get("active"):
                continue
            try:
                reconcile_lease(path.parent)
            except RuntimeError:
                gpu = lease.get("gpu_index")
                if gpu is None or not lease_running(lease):
                    continue
                self.pool.quarantine(int(gpu))
                self._reap_gpu_lease(int(gpu), lease, path)

    def _reap_gpu_lease(self, gpu, lease, path):
        from atas_geometry.service.worker_protocol import lease_running
        def release_after_exit():
            while lease_running(lease) and not self._stop.wait(2):
                pass
            if not lease_running(lease):
                self.pool.release(gpu)
                atomic_json(path, {**lease, "active": False, "reclaimed_at": time.time()})
        threading.Thread(target=release_after_exit, name="geometry-lease-reaper", daemon=True).start()

    def scene_lock(self, run_id: str, story_id: str, scene_id: str) -> threading.Lock:
        key = scene_key(run_id, story_id, scene_id)
        with self._scene_guard:
            if key not in self._scene_locks:
                self._scene_locks[key] = threading.RLock()
            return self._scene_locks[key]

    def submit(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        if kind == "recon":
            from atas_geometry.reconstruct.recon_revision import reconstruction_contract
            payload["reconstruction_contract"] = reconstruction_contract(self.settings, payload)
            # Invalidate derived registration results only; dense RGB-D/SAM
            # cache keys deliberately remain unchanged.
            payload["registration_contract"] = "fixed-initial-world-and-local-repair-v2"
        if kind == "render":
            if payload.get("geometry_layers")=="l1+l2":
                payload["dynamic_render_contract"]="source-pixel-footprint-v1"
            from atas_geometry.world.world_state import load_world
            state=load_world(scene_dir(self.settings.hot_root,payload["run_id"],payload["story_id"],payload["scene_id"]))
            payload["world_version"]=(state or {}).get("version")
        if kind == "recon" and "video_sha256" not in payload:
            payload["video_sha256"] = file_hash(payload["video_path"])
        job = self.store.create(kind, payload)
        if payload.get("retry_failed"):
            job=self.store.retry(job["job_id"])
        with self._cv:
            if job["status"] == "queued" and job["job_id"] not in self._queue:
                self._queue.append(job["job_id"])
                self._cv.notify()
        return job

    def _claim(self):
        """Pop the first queued job whose scene is free, holding that scene's lock.

        A worker never blocks on a busy scene: it would hold no GPU yet keep jobs
        of other scenes waiting behind it while cards sit idle."""
        for position, job_id in enumerate(self._queue):
            job = self.store.get(job_id) or {}
            payload = job.get("payload") or {}
            try:
                lock = self.scene_lock(payload["run_id"], payload["story_id"], payload["scene_id"])
            except KeyError:
                return self._queue.pop(position), None
            if lock.acquire(blocking=False):
                return self._queue.pop(position), lock
        return None, None

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                job_id, lock = self._claim()
                while job_id is None and not self._stop.is_set():
                    self._cv.wait(timeout=1.0)
                    job_id, lock = self._claim()
                if self._stop.is_set():
                    if lock is not None:
                        lock.release()
                    return
            try:
                self._run(job_id)
            except Exception as exc:
                try:
                    self._record_failure(job_id, exc)
                except Exception:
                    traceback.print_exc()
            finally:
                if lock is not None:
                    lock.release()
                with self._cv:
                    self._cv.notify_all()

    def _record_failure(self, job_id: str, exc: Exception) -> None:
        from atas_geometry.service.resource_errors import error_fields
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-8000:],
            **error_fields(exc),
        }
        camera_report = getattr(exc, "report", None)
        if isinstance(camera_report, dict):
            failure["report"] = camera_report
            failure["camera_rejected"] = bool(camera_report.get("camera_rejected"))
        self.store.update(job_id, **failure)
        # One fresh owned process can recover a crashed/hung native runtime or
        # a busy card. Deterministic geometry/quality failures are not retried.
        previous = self.store.get(job_id) or {}
        resource_codes = {"cuda_out_of_memory", "cuda_linalg_initialization_failed",
                          "cuda_native_runtime_failed", "gpu_capacity_unavailable"}
        if failure.get("error_code") in resource_codes:
            if int(previous.get("resource_recovery_count", 0)) >= 1:
                # Memory/capacity shortages stay retryable (the client may resubmit
                # once cards free up); a native crash twice in a fresh process is not.
                self.store.update(job_id, recovery_exhausted=True,
                                  **({"retryable": False} if failure.get("error_code") == "cuda_native_runtime_failed" else {}))
                return
            # A new attempt must reacquire adequate current resources;
            # never immediately relaunch on the already-full card.
            self.store.update(job_id, status="queued", stage="waiting_gpu", resource_recovery_count=1,
                              recovery_reason="cuda_resource_readmission", last_resource_failure=failure,
                              error=None)
            self._requeue(job_id)
        elif isinstance(exc, TimeoutError) and int(previous.get("worker_recovery_count", 0)) < 1:
            self.store.update(job_id, status="queued", worker_recovery_count=1,
                              recovery_reason="isolated_worker_timeout", error=None)
            self._requeue(job_id)

    def _requeue(self, job_id: str) -> None:
        with self._cv:
            self._queue.append(job_id)
            self._cv.notify()

    def _run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        payload=job["payload"]
        with self.scene_lock(payload["run_id"],payload["story_id"],payload["scene_id"]):
            # Duplicate submission may race with a worker claiming the queue.
            if self.store.get(job_id)["status"] != "queued": return
            job = self.store.update(job_id, status="running", error=None, error_code=None,
                                    failure_category=None, retryable=False, resource_details=None)
            if job["kind"] not in {"recon", "render"}:
                raise RuntimeError(f"unknown job kind {job['kind']}")
            self._run_isolated(job)

    def _worker_env(self):
        from atas_geometry import GEOMETRY_ROOT as repo
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
                   PYTHONPATH=str(repo))
        return env

    def _run_isolated(self, job):
        from atas_geometry.service.worker_protocol import run_isolated, reconcile_lease, ENGINE_REVISION
        jid = job["job_id"]
        folder = self.settings.hot_root / "jobs" / jid
        reconcile_lease(folder)
        required_memory = 2048 if job["kind"] == "render" else max(
            self.settings.gpu_min_free_mib, self.settings.recon_min_free_mib)
        queue_timeout = max(30., float(os.getenv("GEOMETRY_GPU_QUEUE_TIMEOUT_SECONDS", "1800")))
        self.store.update(jid, stage="waiting_gpu", engine_revision=ENGINE_REVISION,
                          gpu_index=None, gpu_lease_acquired=False,
                          resource_admission={"min_free_mib": required_memory, "queue_timeout_seconds": queue_timeout})
        gpu = self.pool.acquire(min_free_mib=required_memory,
                                stop=self._stop, timeout_seconds=queue_timeout)
        try:
            values = dataclasses.asdict(self.settings)
            values.update(api_token="", archive_root=None)
            values = {k: str(v) if isinstance(v, Path) else v for k, v in values.items()}
            request = folder / "worker_request.json"
            atomic_json(request, {"settings": values, "job": job})
            self.store.update(jid, gpu_index=gpu, gpu_lease_acquired=True, stage="worker_starting")
            def progress(state):
                fields = {k: v for k, v in state.items() if k not in {
                    "job_id", "payload", "kind", "created_at", "request_fingerprint", "recovery_count"}}
                # Only expose terminal results after the owned process exited.
                if fields.get("status") in {"succeeded", "failed"}:
                    fields.pop("status")
                self.store.update(jid, **fields)
            deadline = self.settings.recon_deadline_seconds if job["kind"] == "recon" else self.settings.render_deadline_seconds
            result = run_isolated([str(self.settings.vista4d_python), "-m", "atas_geometry.service.isolated_worker",
                                   "--request", str(request), "--gpu", str(gpu)],
                                  folder=folder, env=self._worker_env(), deadline_seconds=deadline,
                                  heartbeat_timeout=self.settings.worker_heartbeat_timeout, on_state=progress,
                                  lease_metadata={"gpu_index": gpu, "job_id": jid})
            self.store.update(jid, **{k: v for k, v in result.items() if k not in {
                "job_id", "payload", "kind", "created_at", "request_fingerprint", "recovery_count"}})
            if (result.get("status") == "failed" and result.get("retryable") is True
                    and result.get("error_code") in {"cuda_out_of_memory", "cuda_linalg_initialization_failed", "cuda_native_runtime_failed"}):
                from atas_geometry.service.resource_errors import ResourceUnavailableError
                details = {**(result.get("resource_details") or {}), "min_free_mib": required_memory}
                raise ResourceUnavailableError(result["error_code"], result.get("error", "CUDA memory exhausted"), **details)
            if (result.get("memory_manifest") or {}).get("publication_status") == "committed":
                self._maybe_archive(None, job["payload"]["run_id"], job["payload"]["story_id"])
        finally:
            from atas_geometry.service.worker_protocol import lease_running
            lease_path = folder / "worker_lease.json"
            lease = json.loads(lease_path.read_text()) if lease_path.exists() else {}
            if lease.get("active") and lease_running(lease):
                self.store.update(jid, worker_quarantined=True)
                self._reap_gpu_lease(gpu, lease, lease_path)
            else:
                self.pool.release(gpu)
                self.store.update(jid, gpu_lease_acquired=False)

    def call(self, operation, payload):
        """Small API metadata/projection jobs are isolated too: native decoders
        must never be able to stall the HTTP interpreter."""
        from atas_geometry.service.worker_protocol import run_isolated
        folder = self.settings.hot_root / "rpc" / uuid.uuid4().hex
        request = folder / "worker_request.json"
        atomic_json(request, {"rpc": operation, "root": str(self.settings.hot_root), "payload": payload})
        state = run_isolated([str(self.settings.vista4d_python), "-m", "atas_geometry.service.isolated_worker",
                              "--request", str(request)], folder=folder, env=self._worker_env(),
                             deadline_seconds=self.settings.rpc_deadline_seconds,
                             heartbeat_timeout=self.settings.worker_heartbeat_timeout)
        if state["status"] != "succeeded":
            raise RuntimeError(state.get("error", "geometry RPC failed"))
        result = state["result"]
        if operation == "publish_candidate" and result.get("publication_status") == "committed":
            self._maybe_archive(None, payload["run_id"], payload["story_id"])
        return result

    def _run_recon(self, job: dict[str, Any]) -> None:
        payload = job["payload"]
        if not payload.get("reconstruction_contract"):
            from atas_geometry.reconstruct.recon_revision import reconstruction_contract
            payload["reconstruction_contract"] = reconstruction_contract(self.settings, payload)
        job_id = job["job_id"]
        run_id = str(payload["run_id"])
        story_id = str(payload["story_id"])
        shot_id = str(payload["shot_id"])
        scene_id = str(payload["scene_id"])
        duration = float(payload.get("duration") or 4.0)
        video = Path(payload["video_path"])
        if not payload.get("video_sha256"):
            payload["video_sha256"] = file_hash(video)
        keywords = [str(item) for item in payload.get("dynamic_keywords") or []]
        from atas_geometry.reconstruct.recon_revision import accepted_reconstruction, revision_l0
        trusted_l0 = accepted_reconstruction(self.settings.hot_root, payload)
        l0 = revision_l0(self.settings.hot_root, payload)
        if trusted_l0 is not None:
            l0 = trusted_l0
        dest_l1 = l1_path(self.settings.hot_root, run_id, story_id, scene_id)
        log_path = self.settings.hot_root / "jobs" / job_id / "recon.log"
        gpu = self.pool.acquire(stop=self._stop)
        timings: dict[str, float] = {}
        try:
            self.store.update(job_id, gpu_index=gpu, stage="vista4d_recon")
            frames = probe_frame_count(self.settings.vista4d_python, video)
            if frames > self.settings.max_num_frames:
                raise ValueError("video exceeds configured dense frame limit; refusing silent truncation")
            if payload.get("num_frames") and int(payload["num_frames"]) != frames:
                raise ValueError("production recon preserves every frame; num_frames must match input")
            num_frames = frames
            t0 = time.perf_counter()
            from atas_geometry.reconstruct.recon_revision import dense_manifest, quarantine_incomplete
            l0_ready = trusted_l0 is not None or dense_manifest(l0, payload, frames)
            if l0_ready:
                metadata=l0/"observation.json"
                if metadata.is_file():
                    old=json.loads(metadata.read_text())
                    if old.get("original_video_sha256") and old["original_video_sha256"] != payload["video_sha256"]:
                        raise ValueError("existing L0 belongs to a different video; use a new shot id")
                timings["vista4d_recon_s"] = 0.0
                timings["reused_l0"] = True
            else:
                quarantine_incomplete(l0)
                run_recon_and_seg(
                    python_bin=self.settings.vista4d_python,
                    repo_dir=self.settings.vista4d_repo,
                    video=video,
                    output_dir=l0,
                    keywords=keywords,
                    gpu_index=gpu,
                    log_path=log_path,
                    pi3_model_id=self.settings.pi3_model_id,
                    sam3_checkpoint=self.settings.sam3_checkpoint,
                    width=self.settings.recon_width,
                    height=self.settings.recon_height,
                    num_frames=num_frames,
                    pi3_pixel_limit=self.settings.pi3_pixel_limit,
                    pi3_vo_chunk_size=self.settings.pi3_vo_chunk_size,
                    pi3_vo_overlap=self.settings.pi3_vo_overlap,
                    low_vram=self.settings.low_vram,
                )
                if not dense_manifest(l0, payload, frames, write=True):
                    raise RuntimeError("dense reconstruction incomplete; refusing L0 cache/publication")
                timings["vista4d_recon_s"] = time.perf_counter() - t0
                timings["recon_phases"] = _recon_phases(log_path)
            self.store.update(job_id, stage="fuse_l1")
            t1 = time.perf_counter()
            with self.scene_lock(run_id, story_id, scene_id):
                scene = scene_dir(self.settings.hot_root, run_id, story_id, scene_id)
                home_fuse = None
                initial_world_report = None
                if not dest_l1.is_file():
                    # First accepted observation of a scene: it defines the
                    # scene coordinate frame and becomes a Home candidate.
                    version_dir = scene / ".world_versions" / job_id
                    from atas_geometry.reconstruct.initial_world import run_initial_world
                    self.store.update(job_id, stage="initial_world_alignment")
                    tick = time.perf_counter()
                    l0, initial_world_report = run_initial_world(
                        l0, version_dir / "initial_world", settings=self.settings, gpu_index=gpu)
                    timings["initial_world_alignment_s"] = time.perf_counter() - tick
                    self.store.update(job_id, stage="fuse_l1")
                    candidate_l1 = version_dir / "home_static.npz"
                    candidate_l1.parent.mkdir(parents=True, exist_ok=True)
                    if not candidate_l1.is_file():
                        fuse_l0_to_l1(l0, candidate_l1, duration=duration, shot_id=shot_id, gpu_index=gpu)
                    home_fuse = {"home": True, "candidate_l1": str(candidate_l1)}
                    home = shot_id
                else:
                    home = resolve_home_shot(self.settings.hot_root, run_id, story_id, scene_id)
                # Restart after Home L1 creation but before commit is still Home.
                if home == shot_id and not (scene / "world_state.json").exists():
                    home_fuse = home_fuse or {"home": True}
                fuse_info = writeback_world(
                    l0, scene, payload={**payload, "duration": duration}, job_id=job_id,
                    settings=self.settings, gpu_index=gpu,
                    home_l0=l0_dir(self.settings.hot_root, run_id, story_id, home), home_fuse=home_fuse)
                if initial_world_report is not None:
                    fuse_info["initial_world_alignment"] = initial_world_report
            timings["fuse_l1_s"] = time.perf_counter() - t1
            if fuse_info.get("accepted_l0"):
                l0=Path(fuse_info["accepted_l0"])
                dest_l1=Path(fuse_info["l1_npz"])
            if fuse_info.get("home"):
                remember_home_shot(
                    self.settings.hot_root, run_id, story_id, scene_id, shot_id
                )
        finally:
            self.pool.release(gpu)
        cameras = str(l0 / "cameras.npz")
        result = {
            "status": "succeeded",
            "job_id": job_id,
            "kind": "recon",
            "gpu_index": gpu,
            "cameras_npz": cameras,
            "l1_point_count": fuse_info["l1_point_count"],
            "l0_dir": str(l0),
            "l1_npz": str(dest_l1),
            "timings": timings,
            "fuse": fuse_info,
            "num_frames": num_frames,
        }
        receipt = l0 / "dense_complete.json"
        original = json.loads(receipt.read_text()) if receipt.is_file() else {}
        metadata = json.loads((l0 / "observation.json").read_text()) if (l0 / "observation.json").is_file() else {}
        original_contract = original.get("reconstruction_contract") or metadata.get("source_reconstruction_contract")
        result["reconstruction_provenance"] = {
            "source": "committed_observation" if trusted_l0 is not None else "dense_reconstruction",
            "requested_contract_applied": trusted_l0 is None,
            "original_contract_digest": original.get("contract_digest") or metadata.get("source_reconstruction_contract_digest"),
            "original_reconstruction_contract": original_contract,
            "original_contract_available": bool(original_contract),
        }
        manifest_payload = payload
        if trusted_l0 is not None:
            from atas_geometry.reconstruct.recon_revision import accepted_provenance
            provenance = accepted_provenance(self.settings.hot_root, payload, trusted_l0)
            result["reconstruction_provenance"].update(provenance)
            fuse_info.update(candidate_job_id=provenance["original_candidate_job_id"],
                             publication_status="committed")
            manifest_payload = {**payload, "event_id": provenance["original_event_id"],
                                "review_lineage": provenance["original_review_lineage"]}
        from atas_geometry.world.memory_manifest import observation_manifest
        result["memory_manifest"] = observation_manifest(l0, dest_l1, payload=manifest_payload,
            world_version=fuse_info.get("world_version"),
            publication_status=fuse_info.get("publication_status", "committed" if (
                fuse_info.get("already_committed") or "commit_published" in fuse_info) else "unpublished"),
            candidate_job_id=fuse_info.get("candidate_job_id"))
        self.store.update(job_id, **{key: value for key, value in result.items() if key != "job_id"})
        self._write_job_json(job_id, result)

    def _run_render(self, job: dict[str, Any]) -> None:
        if job["payload"].get("camera_contract") != "semantic-camera-v1":
            raise ValueError("render requires camera_contract=semantic-camera-v1")
        self._run_semantic_camera(job)

    def _run_semantic_camera(self, job: dict[str, Any]) -> None:
        from atas_geometry.render.camera_runtime import render_semantic_camera
        payload, job_id = job["payload"], job["job_id"]
        out = self.store.artifact_path(job_id)
        gpu = self.pool.acquire(min_free_mib=2048, stop=self._stop)
        try:
            self.store.update(job_id, stage="semantic_camera_render", gpu_index=gpu)
            with self.scene_lock(payload["run_id"], payload["story_id"], payload["scene_id"]):
                report = render_semantic_camera(self.settings.hot_root, payload, out,
                    self.settings.render_width, self.settings.render_height, gpu)
                self._persist_chain_camera(payload["run_id"], payload["story_id"], payload["shot_id"], out)
        finally:
            self.pool.release(gpu)
        result = {"status": "succeeded", "kind": "render", "report": report,
            "memory_input": str(out), "memory_input_url": f"/v1/jobs/{job_id}/artifacts/memory_input.mp4"}
        self._write_job_json(job_id, {"job_id": job_id, **result})
        self.store.update(job_id, **result)

    def _persist_chain_camera(
        self, run_id: str, story_id: str, shot_id: str, out_mp4: Path
    ) -> None:
        """Save this render's last-frame camera as the next shot's chain start.

        The pose is one we generated in the scene L1 frame, so the next shot
        chains off it directly -- no recon of the Wan output, no alignment.
        """
        query_path = out_mp4.with_name("camera_query.json")
        if not query_path.is_file():
            return
        query = CameraQuery.load(query_path)
        dest = last_camera_path(self.settings.hot_root, run_id, story_id, shot_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(dest,query.cameras[-1].to_json_dict())

    def _write_job_json(self, job_id: str, result: dict[str, Any]) -> None:
        path = self.settings.hot_root / "jobs" / job_id / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path,result)

    def _maybe_archive(self, story_parent: Path, run_id: str, story_id: str) -> None:
        if self.settings.archive_root is None:
            return
        if os.getenv("GEOMETRY_ARCHIVE_POLICY", "commit") == "story_end":
            return  # The batch supervisor archives explicitly once the story and its pending world jobs finish.
        from atas_geometry.world.archive import archive_story
        # Publication succeeds independently; archival has its own durable
        # receipt and retries on the next commit or service startup.
        threading.Thread(target=archive_story,
            args=(self.settings.hot_root, self.settings.archive_root, run_id, story_id),
            name="geometry-archive", daemon=True).start()
