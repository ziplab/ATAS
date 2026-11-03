"""Close paid-submit crash windows without ever guessing that POST failed."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from atas.providers.wan.client import Wan27Client
from atas.runner.recon_ledger import write_json_atomic
class SubmissionUncertain(RuntimeError):
    """Human/provider reconciliation required; never automatically re-submit."""


def _hash_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024**2),b""): h.update(block)
    return h.hexdigest()


class JournaledTransport:
    def __init__(self,transport,client,request):
        self.transport=transport; self.client=client; self.request=request
        self.root=request.output_dir/"wan_submissions"; self.root.mkdir(parents=True,exist_ok=True)
        self.media=[{"role":r.role.value,"sha256":_hash_file(r.path)} for r in request.condition.references]

    def __getattr__(self,key): return getattr(self.transport,key)

    def request_json(self,method,url,**kwargs):
        if method.upper()!="POST" or url!=self.client._url(self.client.config.create_path):
            return self.transport.request_json(method,url,**kwargs)
        payload=kwargs.get("payload") or {}
        key=hashlib.sha256(json.dumps([payload.get("model"),payload.get("parameters"),
            payload.get("input",{}).get("prompt"),
            [item.get("type") for item in payload.get("input",{}).get("media",[])],
            self.media],sort_keys=True).encode()).hexdigest()
        state_file=self.root/f"{key}.json"
        with (self.root/f"{key}.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            old=json.loads(state_file.read_text()) if state_file.exists() else {}
            if old.get("status")=="submitted" and self._ended_without_output(old.get("task_id")):
                # The provider itself reported this task as failed (archived by the
                # client): nothing was produced, so one new submission is allowed.
                state_file.rename(self.root/f"{key}.{old['task_id']}.ended.json")
                old={}
            if old.get("status")=="submitted": return old["created_response"]
            if old.get("status") in {"submitting","uncertain"}:
                raise SubmissionUncertain("Wan POST outcome unknown; saved intent blocks duplicate paid submission. Reconcile provider task before retry.")
            write_json_atomic(state_file,{"status":"submitting","signature":key})
            try:
                response=self.transport.request_json(method,url,**kwargs)
                task_id=self.client._task_id(response)
            except Exception as exc:
                definite=bool(re.search(r"HTTP\s+(400|401|403|404|422|429)\b",str(exc),re.I))
                write_json_atomic(state_file,{"status":"rejected" if definite else "uncertain","signature":key,
                    "error_type":type(exc).__name__})
                raise
            write_json_atomic(state_file,{"status":"submitted","signature":key,"task_id":task_id,"created_response":response})
            return response

    def _ended_without_output(self,task_id):
        if not isinstance(task_id,str):return False
        return any(self.request.output_dir.glob(f"wan27/*/state.*.{task_id}.json"))

    def download(self,url,destination,**kwargs):
        import cv2
        destination=Path(destination)
        temporary=destination.with_name(destination.name+".partial")
        self.transport.download(url,temporary,**kwargs)
        video=cv2.VideoCapture(str(temporary))
        count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ok,_=video.read()
        if count>0: video.set(cv2.CAP_PROP_POS_FRAMES,count-1)
        last,_=video.read(); video.release()
        if not ok or not last or count<1: raise RuntimeError("downloaded Wan video is incomplete; retry download, not generation")
        os.replace(temporary,destination)


class DurableWanClient(Wan27Client):
    @staticmethod
    def _write_json(path,value): write_json_atomic(path,value)

    async def submit(self,request):
        self.transport=JournaledTransport(self.transport,self,request)
        return await super().submit(request)

    def _latest_succeeded_run(self,*args,**kwargs):
        # A task budget must not substitute an unrelated older result.
        return None

    def _find_resumable_task(self,*args,**kwargs):
        # Exact request state and our content journal handle resumption. A
        # prompt/seed-only fallback can incorrectly reuse different references.
        return None
