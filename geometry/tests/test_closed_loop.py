import json
from pathlib import Path
import numpy as np
import pytest
from atas_geometry.service.durable import JobStore, atomic_json
from atas_geometry.world.world_state import commit_world, load_world, current_l1
from atas_geometry.service.paths import l0_dir, l1_path
def test_job_survives_restart_and_deduplicates_upload_paths(tmp_path):
    store=JobStore(tmp_path)
    payload={"video_path":"/a.mp4","video_sha256":"same","scene_id":"room"}
    job=store.create("recon",payload)
    store.update(job["job_id"],status="running",stage="fusing")
    resumed=JobStore(tmp_path)
    assert resumed.pending()==[job["job_id"]]
    assert resumed.create("recon",{**payload,"video_path":"/b.mp4"})["job_id"]==job["job_id"]
    resumed.update(job["job_id"],status="succeeded",l1_point_count=42)
    assert JobStore(tmp_path).get(job["job_id"])["l1_point_count"]==42


def test_corrupt_job_record_is_quarantined_not_fatal(tmp_path):
    good=JobStore(tmp_path).create("recon",{"video_sha256":"x"})
    bad=tmp_path/"jobs"/"broken"; bad.mkdir()
    (bad/"job.json").write_text("{not json")
    store=JobStore(tmp_path)
    assert store.get(good["job_id"]) is not None
    assert (tmp_path/"jobs_corrupt"/"broken").is_dir()


def test_failed_job_requires_explicit_bounded_retry(tmp_path):
    store=JobStore(tmp_path); job=store.create("render",{"scene":"r"})
    jid=job["job_id"]
    for _ in range(3):
        store.update(jid,status="failed"); assert store.retry(jid)["status"]=="queued"
    store.update(jid,status="failed"); assert store.retry(jid)["status"]=="failed"


def test_legacy_result_is_readable_after_restart(tmp_path):
    atomic_json(tmp_path/"jobs/render_old/result.json",{"job_id":"render_old","status":"succeeded","kind":"render"})
    assert JobStore(tmp_path).get("render_old")["status"]=="succeeded"


def version(scene,name,shot):
    folder=scene/".world_versions"/name; obs=folder/"l0"
    obs.mkdir(parents=True)
    atomic_json(obs/"observation.json",{"registered":True,"shot_id":shot})
    np.savez(folder/"l1.npz",xyz=np.ones((3,3)))
    return obs,folder/"l1.npz"


def test_world_pointer_publishes_l1_and_l0_together(tmp_path):
    scene=tmp_path/"run/story/scenes/room"
    obs,cloud=version(scene,"v1","s1")
    state,published=commit_world(scene,version="v1",l1=cloud,observation=obs,shot_id="s1",source_sha256="a",expected_version=None)
    assert published
    assert l1_path(tmp_path,"run","story","room")==cloud
    assert l0_dir(tmp_path,"run","story","s1")==obs
    _,published=commit_world(scene,version="v1",l1=cloud,observation=obs,shot_id="s1",source_sha256="a",expected_version=None)
    assert not published
    with pytest.raises(ValueError,match="different video"):
        commit_world(scene,version="v2",l1=cloud,observation=obs,shot_id="s1",source_sha256="b",expected_version="v1")


def test_crash_before_pointer_keeps_previous_world(tmp_path,monkeypatch):
    import atas_geometry.world.world_state as module
    scene=tmp_path/"scene"; obs,cloud=version(scene,"v1","s1")
    commit_world(scene,version="v1",l1=cloud,observation=obs,shot_id="s1",source_sha256="a",expected_version=None)
    obs2,cloud2=version(scene,"v2","s2")
    original=module.atomic_json
    def crash(path,value):
        if Path(path).name=="world_state.json": raise OSError("injected crash")
        original(path,value)
    monkeypatch.setattr(module,"atomic_json",crash)
    with pytest.raises(OSError):
        commit_world(scene,version="v2",l1=cloud2,observation=obs2,shot_id="s2",source_sha256="b",expected_version="v1")
    assert current_l1(scene)==cloud and load_world(scene)["version"]=="v1"
    monkeypatch.setattr(module,"atomic_json",original)
    commit_world(scene,version="v2",l1=cloud2,observation=obs2,shot_id="s2",source_sha256="b",expected_version="v1")
    assert current_l1(scene)==cloud2


def test_stale_world_commit_rejected(tmp_path):
    scene=tmp_path/"scene"; obs,cloud=version(scene,"v1","s1")
    commit_world(scene,version="v1",l1=cloud,observation=obs,shot_id="s1",source_sha256="a",expected_version=None)
    obs2,cloud2=version(scene,"v2","s2")
    with pytest.raises(RuntimeError,match="world changed"):
        commit_world(scene,version="v2",l1=cloud2,observation=obs2,shot_id="s2",source_sha256="b",expected_version="wrong")


def test_partial_snapshot_resumes_without_changing_raw_metadata(tmp_path):
    from atas_geometry.world.writeback import _snapshot_l0
    source=tmp_path/"raw"; target=tmp_path/"revision"
    source.mkdir(); (source/"video.mp4").write_bytes(b"video")
    atomic_json(source/"observation.json",{"registered":False})
    target.mkdir(); import os
    os.link(source/"video.mp4",target/"video.mp4")
    _snapshot_l0(source,target)
    assert (target/"observation.json").is_file()
    atomic_json(target/"observation.json",{"registered":True})
    assert not json.loads((source/"observation.json").read_text())["registered"]
