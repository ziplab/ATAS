import contextlib
import threading
from types import SimpleNamespace
import pytest
from atas_geometry.world import publication_recovery as p
from atas_geometry.service.resource_errors import native_resource_code, ResourceUnavailableError
def test_native_crash_is_distinct_from_geometry_error():
    assert native_resource_code('CUDA error: an illegal instruction was encountered')=='cuda_native_runtime_failed'
    assert native_resource_code('internal_reconstruction_inconsistent') is None

def test_native_crash_recovery_exhausts_after_one_fresh_process(tmp_path):
    from atas_geometry.service.jobs import GeometryRuntime
    from atas_geometry.service.durable import JobStore
    r=GeometryRuntime.__new__(GeometryRuntime);r.store=JobStore(tmp_path)
    j=r.store.create('recon',{'run_id':'r','story_id':'s','scene_id':'room'})
    r._stop=threading.Event();r._cv=threading.Condition();r._queue=[j['job_id']];r._scene_guard=threading.Lock();r._scene_locks={}
    calls=[]
    def run(jid):
        calls.append(jid)
        if len(calls)==2:r._stop.set()
        raise ResourceUnavailableError('cuda_native_runtime_failed','native crash')
    r._run=run;r._loop()
    row=r.store.get(j['job_id'])
    assert len(calls)==2 and not row['retryable'] and row['recovery_exhausted']

def fixture(monkeypatch,tmp_path):
    source=tmp_path/'film.mp4';source.write_bytes(b'immutable film')
    data=dict(run_id='r',story_id='s',scene_id='room',shot_id='shot',job_id='old',parent_version='v1',source_sha256=p.file_hash(source))
    payload={k:data[k] for k in ('run_id','story_id','scene_id','shot_id')}
    payload.update(video_path=str(source),video_sha256=data['source_sha256'],review_lineage={'render':'original'})
    monkeypatch.setattr(p,'candidate_source',lambda *a:(tmp_path,data,tmp_path/'l0'))
    monkeypatch.setattr(p,'load_world',lambda *a:{'version':'v2'})
    seen=[]
    r=SimpleNamespace(settings=SimpleNamespace(hot_root=tmp_path),
        store=SimpleNamespace(get=lambda jid:{'kind':'recon','payload':payload}),
        scene_lock=lambda *a:contextlib.nullcontext(),submit=lambda kind,body:(seen.append(body) or {'job_id':'new'}))
    return r,seen,source

def test_rebase_preserves_source_lineage_and_creates_new_parent_key(monkeypatch,tmp_path):
    r,seen,_=fixture(monkeypatch,tmp_path)
    assert p.submit_rebase(r,{})['job_id']=='new'
    assert seen[0]['rebase_parent_version']=='v2'
    assert seen[0]['review_lineage']=={'render':'original'}
    assert seen[0]['video_sha256']==p.file_hash(seen[0]['video_path'])

def test_rebase_refuses_tampered_film(monkeypatch,tmp_path):
    r,seen,source=fixture(monkeypatch,tmp_path);source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):p.submit_rebase(r,{})
    assert not seen
