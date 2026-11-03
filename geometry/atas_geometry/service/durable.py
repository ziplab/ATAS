"""Atomic local state and restart-safe, content-idempotent geometry jobs."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024**2),b""): digest.update(block)
    return digest.hexdigest()


def fingerprint(kind, payload):
    data = {k:v for k,v in payload.items() if k not in {
        "video_path", "video", "request_id", "retry_failed"}}
    return hashlib.sha256(json.dumps([kind,data],sort_keys=True,separators=(",",":")).encode()).hexdigest()


class JobStore:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self._lock = threading.RLock(); self._jobs = {}; self._keys = {}
        for folder in sorted((self.root/"jobs").glob("*")):
            p = folder/"job.json"
            if not p.is_file(): p = folder/"result.json"
            if not p.is_file(): continue
            try:
                job = json.loads(p.read_text())
                jid = job.get("job_id",folder.name)
                if not isinstance(jid,str) or not re.fullmatch(r"[\w-]+",jid):
                    raise ValueError("invalid persisted job id")
            except (OSError, ValueError) as exc:
                # One damaged record must not keep the whole service down.
                self._quarantine(folder, exc)
                continue
            job["job_id"] = jid
            if job.get("status") in {"running","queued"}:
                if "payload" not in job: raise ValueError("pending job lacks recoverable payload")
                attempts = int(job.get("recovery_count",0))+1
                job.update(status="queued" if attempts <= 3 else "failed",recovery_count=attempts,
                           recovery_reason="service_restart",updated_at=time.time())
                if attempts > 3: job["error"] = "restart recovery limit reached; explicit retry required"
                atomic_json(folder/"job.json",job)
            self._jobs[jid] = job
            if job.get("request_fingerprint"): self._keys[job["request_fingerprint"]] = jid

    def _quarantine(self, folder, exc):
        target = self.root/"jobs_corrupt"/folder.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            folder.rename(target)
        except OSError:
            pass
        print(f"geometry: quarantined unreadable job record {folder.name}: {exc}", flush=True)

    def pending(self):
        return [j["job_id"] for j in sorted(self._jobs.values(),key=lambda x:x.get("created_at",0)) if j.get("status")=="queued"]

    def create(self, kind, payload):
        payload = copy.deepcopy(payload)
        key = fingerprint(kind,payload)
        with self._lock:
            if key in self._keys:
                return copy.deepcopy(self._jobs[self._keys[key]])
            jid = f"{kind}_{uuid.uuid4().hex[:12]}"
            job = dict(job_id=jid,kind=kind,payload=payload,status="queued",created_at=time.time(),
                       request_fingerprint=key,recovery_count=0)
            atomic_json(self.root/"jobs"/jid/"job.json",job)
            self._jobs[jid]=job; self._keys[key]=jid
            return copy.deepcopy(job)

    def get(self,jid):
        with self._lock: return copy.deepcopy(self._jobs.get(jid))

    def update(self,jid,**fields):
        fields.pop("job_id",None)
        with self._lock:
            updated={**self._jobs[jid],**fields,"updated_at":time.time()}
            atomic_json(self.root/"jobs"/jid/"job.json",updated)
            self._jobs[jid]=updated
            return copy.deepcopy(updated)

    def retry(self,jid):
        with self._lock:
            job=self._jobs[jid]
            if job["status"] != "failed": return copy.deepcopy(job)
            attempts=int(job.get("retry_count",0))
            if attempts>=3: return copy.deepcopy(job)
            return self.update(jid,status="queued",retry_count=attempts+1,recovery_count=0,error=None)

    def artifact_path(self,jid):
        if not re.fullmatch(r"[\w-]+",jid): raise ValueError("invalid job id")
        return self.root/"jobs"/jid/"memory_input.mp4"
