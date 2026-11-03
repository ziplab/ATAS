"""Cheap dependency checks so readiness reflects whether jobs can actually run."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from atas_geometry.service.config import Settings
def _writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-check-"):
            pass
        return True, str(path)
    except OSError as exc:
        return False, f"{path}: {exc}"


def dependency_report(settings: Settings) -> dict:
    checks = {}
    checks["hot_root_writable"] = _writable(settings.hot_root)
    recon_script = settings.vista4d_repo / "scripts" / "preprocess" / "recon_and_seg_single.py"
    checks["backbones"] = (recon_script.is_file(), str(recon_script))
    python = Path(settings.vista4d_python)
    checks["python"] = (python.is_file() and os.access(python, os.X_OK), str(python))
    pi3 = Path(settings.pi3_model_id)
    # A Hugging Face model id is allowed; a local path must exist.
    checks["pi3_model"] = (pi3.exists() or ("/" in settings.pi3_model_id and not settings.pi3_model_id.startswith("/")),
                           settings.pi3_model_id)
    checks["sam3_checkpoint"] = (settings.sam3_checkpoint.is_file(), str(settings.sam3_checkpoint))
    checks["nvidia_smi"] = (shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "not found")
    return {"ok": all(ok for ok, _ in checks.values()),
            "checks": {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()}}
