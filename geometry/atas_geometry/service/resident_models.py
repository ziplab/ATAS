"""Per-GPU serial model process; weights are reused across jobs and offloaded to host memory when idle.

Only the model loading lifecycle changes; models, precision, segmentation and fusion are unchanged.
The compute process exits when its client disappears, so a timed-out job cannot keep writing
to disk after its GPU lease is released.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import inspect
import json
import os
from pathlib import Path
import runpy
import select
import socket
import subprocess
import sys
import threading
import time
import traceback

ALLOWED = {"scripts.preprocess.recon_and_seg_single", "atas_geometry.register.anchored_fusion"}


def precision_state():
    import torch
    return (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.is_grad_enabled())


def restore_precision(state):
    import torch
    enabled, dtype, matmul, cudnn, grad = state
    torch.set_autocast_enabled('cuda', enabled)
    torch.set_autocast_dtype('cuda', dtype)
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.set_grad_enabled(grad)
    torch.clear_autocast_cache()


def resident_command(command, *, repo, vista_repo, log_path):
    if os.getenv("GEOMETRY_RESIDENT_MODELS", "0") != "1":
        return command
    module = command[2]
    if command[1] != "-m" or module not in ALLOWED:
        raise ValueError("unsupported resident module")
    return [command[0], str(Path(repo) / "atas_geometry/service/resident_models.py"),
            "client", "--vista-repo", str(vista_repo), "--module", module,
            "--task-log", str(Path(log_path).resolve()),
            "--", *command[3:]]


class ModelCache:
    def __init__(self):
        self.entries = {}
        self.precisions = {}

    @staticmethod
    def weights(model):
        return getattr(model, "model", model)

    def offload(self, except_kind=None):
        for kind, (_, model) in self.entries.items():
            if kind != except_kind:
                self.weights(model).to("cpu")
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def wrap(self, kind, loader):
        signature = inspect.signature(loader)
        def cached(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = repr(sorted((k, str(v)) for k, v in bound.arguments.items()))
            self.offload(except_kind=kind)
            if kind == 'sam3':
                import torch
                torch.backends.cuda.matmul.allow_tf32 = self.sam_tf32[0]
                torch.backends.cudnn.allow_tf32 = self.sam_tf32[1]
            hit = kind in self.entries and self.entries[kind][0] == key
            if not hit:
                self.entries.pop(kind, None)
                model = loader(*args, **kwargs)
                self.entries[kind] = (key, model)
                self.precisions[kind] = precision_state()
            else:
                model = self.entries[kind][1]
                self.weights(model).to(bound.arguments.get("device", "cuda"))
                restore_precision(self.precisions[kind])
            print("ATAS_MODEL_CACHE " + json.dumps({"kind": kind, "hit": hit}), flush=True)
            return model
        return cached


def install_cache():
    baseline = precision_state()
    from utils.recon_and_seg import recon_pi3, seg_sam3_official
    cache = ModelCache()
    sam_state = precision_state()
    cache.sam_tf32 = sam_state[2:4]
    recon_pi3.init_pi3 = cache.wrap("pi3", recon_pi3.init_pi3)
    seg_sam3_official.init_sam3_video = cache.wrap("sam3", seg_sam3_official.init_sam3_video)
    cache.baseline = baseline
    restore_precision(baseline)
    return cache


def location():
    card = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not (card.isdigit() or (card.startswith("GPU-") and all(c.isalnum() or c == "-" for c in card))):
        raise ValueError("resident worker requires one explicit physical GPU")
    root = Path(f"/var/tmp/atas-resident-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise PermissionError("resident directory must be private")
    return root, root / f"gpu{card}.sock"


def code_fingerprint(vista_repo):
    """Content hash of the code a resident worker executes. A worker outlives the
    service process, so a restart with new code must not reuse an old worker."""
    import hashlib
    from atas_geometry import GEOMETRY_ROOT as repo
    vista = Path(vista_repo)
    # All Python under the package and the bundled model code (Pi3X, SAM3, scripts).
    files = [*sorted((repo / "atas_geometry").rglob("*.py")), *sorted(vista.rglob("*.py"))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def wait_for_exit(root, endpoint, timeout=60):
    """The old worker holds its server lock until it has exited."""
    limit = time.monotonic() + timeout
    with (root / (endpoint.stem + ".server.lock")).open("a") as owner:
        while time.monotonic() < limit:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(owner, fcntl.LOCK_UN)
                return
            except BlockingIOError:
                time.sleep(.2)
    raise TimeoutError("stale resident worker did not exit")


def connect(path):
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.connect(str(path))
        return sock
    except OSError:
        sock.close()
        return None


def client(args):
    root, endpoint = location()
    code = code_fingerprint(args.vista_repo)
    # Different callers never interleave tasks on one card, even outside API.
    with (root / (endpoint.stem + ".client.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(2):
            if _submit(args, root, endpoint, code):
                return
            wait_for_exit(root, endpoint)       # stale worker: replace it once
        raise RuntimeError("resident worker kept reporting stale code")


def _submit(args, root, endpoint, code):
    """Run one task on the card's worker; False when the worker runs older code."""
    sock = connect(endpoint)
    if sock is None:
        with (root / (endpoint.stem + ".log")).open("ab") as log:
            process = subprocess.Popen([sys.executable, __file__, "server",
                "--vista-repo", args.vista_repo], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        limit = time.monotonic() + 90
        while sock is None and time.monotonic() < limit:
            if process.poll() is not None:
                raise RuntimeError("resident model server failed to start")
            time.sleep(.2)
            sock = connect(endpoint)
        if sock is None:
            raise TimeoutError("resident startup timeout")
    try:
        log_path = args.task_log
        if not log_path.startswith("/") or not Path(log_path).is_file():
            raise ValueError("resident client stdout must be a task log file")
        payload = {"module": args.module, "args": args.rest,
                   "cwd": os.getcwd(), "log": log_path, "code": code}
        sock.sendall(json.dumps(payload).encode() + b"\n")
        sock.settimeout(1200)
        response = json.loads(sock.makefile("rb").readline())
        if response.get("stale"):
            return False
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return True
    finally:
        sock.close()


def server(args):
    root, endpoint = location()
    with (root / (endpoint.stem + ".server.lock")).open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        endpoint.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(endpoint)); listener.listen(1)
        from atas_geometry import GEOMETRY_ROOT as repo
        sys.path[:0] = [str(repo), args.vista_repo]
        cache = None
        code = code_fingerprint(args.vista_repo)
        while True:
            conn, _ = listener.accept()
            request = json.loads(conn.makefile("rb").readline())
            if request.get("code") != code:
                conn.sendall(b'{"ok":false,"stale":true,"error":"resident worker runs older code"}\n')
                conn.close(); listener.close(); endpoint.unlink(missing_ok=True)
                return
            if request["module"] not in ALLOWED:
                conn.sendall(b'{"ok":false,"error":"module not allowed"}\n')
                conn.close(); continue
            finished = threading.Event()
            def monitor(conn=conn, finished=finished):
                while not finished.wait(.1):
                    readable, _, _ = select.select([conn], [], [], 0)
                    if readable and not conn.recv(1, socket.MSG_PEEK):
                        os._exit(75)
            guard = threading.Thread(target=monitor, daemon=True)
            guard.start()
            old_cwd, old_argv = os.getcwd(), sys.argv
            saved = [os.dup(1), os.dup(2)]
            ok, error = False, ""
            try:
                sys.stdout.flush(); sys.stderr.flush()
                with open(request["log"], "a") as output:
                    os.dup2(output.fileno(), 1); os.dup2(output.fileno(), 2)
                    os.chdir(request["cwd"])
                    if cache is None:
                        from threadpoolctl import threadpool_limits
                        threadpool_limits(limits=4)
                        import torch, cv2
                        torch.set_num_threads(4); cv2.setNumThreads(4)
                        cache = install_cache()
                    import torch
                    # SAM may enable process-wide autocast; each job must start with the same precision state as a fresh process.
                    restore_precision(cache.baseline)
                    sys.argv = [request["module"], *request["args"]]
                    print("ATAS_RESIDENT_BEGIN " + str(os.getpid()), flush=True)
                    runpy.run_module(request["module"], run_name="__main__")
                    restore_precision(cache.baseline)
                    cache.offload()
                    # SAM closes every per-video session in its existing finally.
                    for kind, (_, model) in cache.entries.items():
                        if kind == "sam3" and model._ALL_INFERENCE_STATES:
                            raise RuntimeError("SAM session leaked across tasks")
                    print("ATAS_RESIDENT_END " + str(os.getpid()), flush=True)
                    ok = True
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                sys.stdout.flush(); sys.stderr.flush()
                os.dup2(saved[0], 1); os.dup2(saved[1], 2)
                for fd in saved: os.close(fd)
                os.chdir(old_cwd); sys.argv = old_argv
                finished.set()
                guard.join(timeout=1)
            conn.sendall(json.dumps({"ok": ok, "error": error}).encode() + b"\n")
            conn.close()
            if not ok:
                # Never reuse possibly corrupted CUDA/model state after error.
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["client", "server"])
    parser.add_argument("--vista-repo", required=True)
    parser.add_argument("--module", choices=sorted(ALLOWED))
    parser.add_argument("--task-log")
    args, rest = parser.parse_known_args()
    args.rest = rest[1:] if rest[:1] == ["--"] else rest
    (client if args.mode == "client" else server)(args)
