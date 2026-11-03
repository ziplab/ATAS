from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


INTENT_FILE = "geometry_recon_intent.json"
RESULT_FILE = "geometry_recon_result.json"
STATE_FILE = "geometry_recon_state.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Replace one ledger file atomically so a killed client cannot truncate it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def result_is_complete(result: dict[str, Any] | None) -> bool:
    """A recon is resumable only after GPU writeback returned its full contract."""

    if not isinstance(result, dict) or result.get("status") != "succeeded":
        return False
    required_strings = ("job_id", "cameras_npz", "l0_dir", "l1_npz")
    if any(not str(result.get(key) or "").strip() for key in required_strings):
        return False
    if not isinstance(result.get("l1_point_count"), int):
        return False
    if not isinstance(result.get("fuse"), dict):
        return False
    return True


def completed_result(shot_dir: Path) -> dict[str, Any] | None:
    """Return a durable success, including one written just before a client crash."""

    result = _read_object(shot_dir / RESULT_FILE)
    return result if result_is_complete(result) else None


def resumable_job_id(shot_dir: Path) -> str | None:
    state = _read_object(shot_dir / STATE_FILE)
    if not state or state.get("status") not in {"submitted", "waiting"}:
        return None
    job_id = str(state.get("job_id") or "").strip()
    return job_id or None


def _attempt(shot_dir: Path) -> int:
    state = _read_object(shot_dir / STATE_FILE) or {}
    try:
        return max(int(state.get("attempt") or 0), 0)
    except (TypeError, ValueError):
        return 0


def record_intent(shot_dir: Path, intent: dict[str, Any]) -> None:
    write_json_atomic(shot_dir / INTENT_FILE, intent)


def record_submitted(shot_dir: Path, *, job_id: str) -> None:
    write_json_atomic(
        shot_dir / STATE_FILE,
        {
            "schema_version": 1,
            "status": "waiting",
            "attempt": _attempt(shot_dir) + 1,
            "job_id": job_id,
            "updated_at": _now(),
        },
    )


def record_failed(shot_dir: Path, *, error: BaseException, job_id: str | None) -> None:
    write_json_atomic(
        shot_dir / STATE_FILE,
        {
            "schema_version": 1,
            "status": "failed",
            "attempt": max(_attempt(shot_dir), 1),
            "job_id": job_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at": _now(),
        },
    )


def record_wait_uncertain(shot_dir: Path, *, error: BaseException, job_id: str) -> None:
    """Preserve a remote job id when polling failed but execution may continue."""

    write_json_atomic(
        shot_dir / STATE_FILE,
        {
            "schema_version": 1,
            "status": "waiting",
            "attempt": max(_attempt(shot_dir), 1),
            "job_id": job_id,
            "last_poll_error_type": type(error).__name__,
            "last_poll_error": str(error),
            "updated_at": _now(),
        },
    )


def _writeback_summary(result: dict[str, Any]) -> dict[str, Any]:
    fuse = dict(result["fuse"])
    after = int(result["l1_point_count"])
    merged = fuse.get("merged")
    novel = fuse.get("novel_points")
    before: int | None = None
    if merged is False:
        before = after
    elif merged is True and isinstance(novel, int):
        before = max(after - novel, 0)
    fusion_status = "merged" if merged is True else "skipped" if merged is False else "completed"
    if fuse.get("publication_status")=="pending_review": fusion_status="candidate_pending_review"

    job_id = str(result["job_id"])
    return {
        "l0": {
            "directory": str(result["l0_dir"]),
            "cameras_npz": str(result["cameras_npz"]),
            "num_frames": result.get("num_frames"),
        },
        "fusion": {
            "status": fusion_status,
            "merged": merged,
            "report": fuse,
        },
        "l1": {
            "npz": str(result["l1_npz"]),
            "point_count_before": before,
            "point_count_after": after,
            # Versioned worlds expose the authoritative commit pointer; keep
            # the job ID only as a compatibility fallback for older services.
            "version": fuse.get("world_version") if fuse.get("publication_status")=="pending_review" else fuse.get("world_version") or job_id,
        },
    }


def record_succeeded(shot_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    if not result_is_complete(result):
        raise RuntimeError("geometry recon succeeded without a complete writeback result")
    summary = _writeback_summary(result)
    # Result first: if the process dies before state replacement, resume can
    # still recognize the complete durable result and avoid duplicate recon.
    write_json_atomic(shot_dir / RESULT_FILE, result)
    write_json_atomic(
        shot_dir / STATE_FILE,
        {
            "schema_version": 1,
            "status": "succeeded",
            "attempt": max(_attempt(shot_dir), 1),
            "job_id": str(result["job_id"]),
            "completed_at": _now(),
            "writeback": summary,
        },
    )
    return summary


def ensure_recon_writeback(
    shot_dir: Path,
    *,
    submit: Callable[[], str],
    wait: Callable[[str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Finish one recon exactly as far as the geometry service permits.

    A complete local result wins. Otherwise an interrupted in-flight job is
    waited again; failed attempts are not reused and therefore submit anew.
    The boolean return value says whether existing work was resumed.
    """

    durable = completed_result(shot_dir)
    if durable is not None:
        return durable, record_succeeded(shot_dir, durable), True

    job_id = resumable_job_id(shot_dir)
    resumed = job_id is not None
    try:
        if job_id is None:
            job_id = str(submit() or "").strip()
            if not job_id:
                raise RuntimeError("geometry recon returned an empty job_id")
            record_submitted(shot_dir, job_id=job_id)
        result = wait(job_id)
        summary = record_succeeded(shot_dir, result)
        return result, summary, resumed
    except Exception as exc:
        # Unknown completion is never a terminal remote failure. Keep this job ID
        # for the world-only collector rather than submitting another paid film.
        if job_id and getattr(exc, "recon_job_may_still_be_running", False):
            record_wait_uncertain(shot_dir, error=exc, job_id=job_id)
        else:
            record_failed(shot_dir, error=exc, job_id=job_id)
        raise
