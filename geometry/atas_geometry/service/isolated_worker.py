"""One owned process per geometry operation; the HTTP process never computes."""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from pathlib import Path

from atas_geometry.service.durable import atomic_json
class WorkerStore:
    def __init__(self, folder, job):
        self.folder, self.job = Path(folder), dict(job)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.update(job["job_id"], status="running", stage="worker_starting")
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self):
        while not self.stop.wait(5):
            with self.lock:
                atomic_json(self.folder / "worker_state.json", {**self.job, "heartbeat_at": time.time()})

    def update(self, jid, **fields):
        with self.lock:
            self.job.update(fields, updated_at=time.time(), heartbeat_at=time.time())
            atomic_json(self.folder / "worker_state.json", self.job)
            return dict(self.job)

    def get(self, jid):
        return dict(self.job)

    def artifact_path(self, jid):
        return self.folder / "memory_input.mp4"


class FixedPool:
    def __init__(self, gpu):
        self.gpu = gpu

    def acquire(self, **kwargs):
        return self.gpu

    def release(self, gpu):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    folder = args.request.parent
    job = request.get("job", {"job_id": folder.name})
    store = WorkerStore(folder, job)
    try:
        # A small thread budget also prevents native libraries oversubscribing
        # all CPUs when independent stories use several workers.
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=4)
        import cv2
        cv2.setNumThreads(4)
        if request.get("rpc"):
            from importlib import import_module
            allowed = {
                "camera_context": ("atas_geometry.render.camera_context", "camera_context"),
                "publish_candidate": ("atas_geometry.world.reviewed_world", "publish_candidate"),
            }
            module, function = allowed[request["rpc"]]
            store.update(job["job_id"], stage=request["rpc"])
            result = getattr(import_module(module), function)(
                Path(request["root"]), request["payload"])
            store.update(job["job_id"], status="succeeded", result=result)
        else:
            from atas_geometry.service.config import Settings
            from atas_geometry.service.jobs import GeometryRuntime
            values = request["settings"]
            for key in ("hot_root", "archive_root", "vista4d_repo", "vista4d_python", "sam3_checkpoint"):
                if values[key] is not None:
                    values[key] = Path(values[key])
            values["gpu_ids"] = tuple(values["gpu_ids"])
            runtime = GeometryRuntime.__new__(GeometryRuntime)
            runtime.settings = Settings(**values)
            runtime.pool, runtime.store = FixedPool(args.gpu), store
            runtime._stop = threading.Event()
            runtime._scene_locks, runtime._scene_guard = {}, threading.Lock()
            # Parent owns this scene and has reconciled its previous worker.
            job["payload"]["recover_interrupted_registration"] = True
            if job["kind"] == "recon":
                runtime._run_recon(job)
            else:
                runtime._run_render(job)
    except BaseException as exc:
        from atas_geometry.service.resource_errors import error_fields
        fields = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-8000:]}
        resource_fields = error_fields(exc, physical_gpu_index=args.gpu,
                                       stage=store.job.get("stage", "starting"))
        fields.update(resource_fields)
        if isinstance(getattr(exc, "report", None), dict):
            fields.update(report=exc.report, camera_rejected=bool(exc.report.get("camera_rejected")))
        store.update(job["job_id"], **fields)
    finally:
        store.stop.set()


if __name__ == "__main__":
    main()
