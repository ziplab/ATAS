"""Bounded process ownership and file-based progress; no geometry imports."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from atas_geometry.service.durable import atomic_json
ENGINE_REVISION = "resource-admission-v3-20260916"


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def owned_group_members(lease):
    """Prove a spawn-created session even after its leader has been reaped.

    Return None when ownership cannot be proven, otherwise live member PIDs.
    Other processes cannot join this session from another session. The boot,
    session, group, UID, and monotonic process-start stamps prevent PID reuse.
    """
    pid = int(lease.get("pid", 0))
    if pid <= 1:
        return None
    modern = lease.get("owner_contract") == "isolated_geometry_session_v1"
    if modern:
        if lease.get("boot_id") != boot_id():
            return []
        if (lease.get("pgid") != pid
                or lease.get("session_id") != pid or not lease.get("process_start")):
            return None
        current_start = process_start(pid)
        if current_start is not None and current_start != lease["process_start"]:
            # An old inactive lease can outlive its entire session. If that
            # numeric leader PID was reused, it is categorically not ours.
            return []
    else:
        # Pre-upgrade leases are recoverable only while their original leader
        # is still alive and directly identifiable. Never infer a dead owner.
        if process_start(pid) != lease.get("process_start"):
            return None
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if b"atas_geometry.service.isolated_worker" not in cmd or os.getsid(pid) != pid or os.getpgid(pid) != pid:
                return None
        except OSError:
            return None
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pid:
                continue
            if (int(fields[3]) != pid or int(fields[19]) < int(lease["process_start"])
                    or (modern and entry.stat().st_uid != lease.get("uid"))):
                return None
            if fields[0] != "Z":
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError):
            # An inaccessible process cannot be certified as belonging to us.
            continue
    return members

def process_start(pid):
    try:
        # comm may contain spaces, so split after the final closing parenthesis.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def lease_running(lease):
    """A zombie has exited; a D-state worker has not, even after SIGKILL."""
    if lease.get("owner_contract") == "isolated_geometry_session_v1":
        members = owned_group_members(lease)
        # An unprovable active lease must never be silently freed.
        return members is None or bool(members)
    pid = int(lease.get("pid", 0))
    if process_start(pid) != lease.get("process_start"):
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def stop_owned(lease):
    """Only terminate a verified worker process group, never an unrelated PID."""
    pid = int(lease.get("pid", 0))
    members = owned_group_members(lease)
    if members is None:
        return False
    if not members:
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
        limit = time.monotonic() + 3
        while lease_running(lease) and time.monotonic() < limit:
            time.sleep(.05)
        # Killing this owned group also reaps nested Vista/Pi3 workers.
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True
    except (OSError, ValueError):
        return False


def reconcile_lease(folder):
    path = Path(folder) / "worker_lease.json"
    if not path.exists():
        return
    lease = json.loads(path.read_text())
    pid = int(lease.get("pid", 0))
    if lease_running(lease):
        if not stop_owned(lease):
            raise RuntimeError("worker lease is live but ownership cannot be verified")
        if lease_running(lease):
            raise RuntimeError("owned worker has not exited; lease remains quarantined")
    atomic_json(path, {**lease, "reconciled_at": time.time(), "active": False})


def run_isolated(command, *, folder, env, deadline_seconds, heartbeat_timeout=90,
                 on_state=None, process_factory=subprocess.Popen, lease_metadata=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    reconcile_lease(folder)
    state_path = folder / "worker_state.json"
    if state_path.exists():
        # Old partial state is diagnostic only; never adopt its terminal result.
        state_path.replace(folder / f"worker_state.previous.{time.time_ns()}.json")
    started = time.monotonic()
    with (folder / "worker.log").open("a") as output:
        process = process_factory(command, stdout=output, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=True)
        lease = {**(lease_metadata or {}), "pid": process.pid, "process_start": process_start(process.pid),
                 "active": True, "created_at": time.time(), "deadline_seconds": deadline_seconds,
                 "engine_revision": ENGINE_REVISION}
        if "atas_geometry.service.isolated_worker" in command:
            lease.update(owner_contract="isolated_geometry_session_v1", pgid=process.pid,
                         session_id=process.pid, uid=os.getuid(), boot_id=boot_id())
        atomic_json(folder / "worker_lease.json", lease)
        last_stamp = None
        state = {}
        try:
            while True:
                if state_path.exists():
                    stamp = state_path.stat().st_mtime_ns
                    if stamp != last_stamp:
                        state = json.loads(state_path.read_text())
                        last_stamp = stamp
                        if on_state:
                            on_state(state)
                code = process.poll()
                if code is not None:
                    if state_path.exists():
                        state = json.loads(state_path.read_text())
                        if on_state:
                            on_state(state)
                    if code or state.get("status") not in {"succeeded", "failed"}:
                        raise RuntimeError(f"isolated geometry worker exited {code} without a valid result")
                    return state
                elapsed = time.monotonic() - started
                heartbeat = state.get("heartbeat_at", lease["created_at"])
                if elapsed > deadline_seconds:
                    raise TimeoutError(f"geometry worker exceeded {deadline_seconds:g}s hard deadline at {state.get('stage', 'starting')}")
                if time.time() - heartbeat > heartbeat_timeout:
                    raise TimeoutError(f"geometry worker heartbeat expired at {state.get('stage', 'starting')}")
                time.sleep(.25)
        finally:
            # The Python leader may have exited while a nested Vista worker
            # is still running. Always reconcile the whole owned group.
            if process.poll() is None or lease_running(lease):
                verified = stop_owned(lease)
                limit = time.monotonic() + 10
                while verified and lease_running(lease) and time.monotonic() < limit:
                    process.poll()
                    time.sleep(.05)
                process.poll()
                if not verified or lease_running(lease) or process.poll() is None:
                    atomic_json(folder / "worker_lease.json", {**lease, "active": True,
                        "quarantined": True, "termination_failed_at": time.time()})
                    raise RuntimeError("owned worker group did not exit; GPU lease quarantined")
            atomic_json(folder / "worker_lease.json", {**lease, "active": False,
                                                       "finished_at": time.time()})
