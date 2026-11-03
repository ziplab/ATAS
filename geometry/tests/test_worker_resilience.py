import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atas_geometry.service.durable import atomic_json
from atas_geometry.service.worker_protocol import run_isolated, stop_owned, process_start
from atas_geometry.reconstruct.recon_revision import revision_l0, dense_manifest
def test_hung_owned_worker_has_deadline_and_is_reaped(tmp_path):
    command = [sys.executable, "-c", "while True: pass", "atas_geometry.service.isolated_worker"]
    with pytest.raises(TimeoutError, match="hard deadline"):
        run_isolated(command, folder=tmp_path, env=os.environ.copy(), deadline_seconds=.3)
    lease = json.loads((tmp_path / "worker_lease.json").read_text())
    assert lease["active"] is False
    assert process_start(lease["pid"]) is None


def test_recovery_never_kills_unowned_process(tmp_path):
    assert not stop_owned({"pid": os.getpid(), "process_start": process_start(os.getpid())})
    assert not stop_owned({"pid": os.getpid(), "process_start": "wrong-start"})


def test_modern_lease_does_not_signal_a_reused_leader_pid(monkeypatch):
    from atas_geometry.service.worker_protocol import owned_group_members, boot_id
    pid = os.getpid()
    lease = dict(pid=pid, pgid=pid, session_id=pid, uid=os.getuid(),
        owner_contract="isolated_geometry_session_v1", boot_id=boot_id(), process_start="0")
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("reused PID must never be signaled"))
    assert owned_group_members(lease) == [] and stop_owned(lease)


@pytest.mark.parametrize("leader_succeeds", [False, True])
def test_exited_leader_does_not_leave_nested_worker_or_release_a_live_group(tmp_path, leader_succeeds):
    from atas_geometry.service.worker_protocol import owned_group_members
    # This is a real nested process, not a monkeypatched Popen. It inherits the
    # session created for the fake owned worker and survives its leader's exit.
    code = """
import json,os,subprocess,sys
from pathlib import Path
folder=Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
(folder/'nested_pid').write_text(str(child.pid))
if sys.argv[2]=='success':
    (folder/'worker_state.json').write_text(json.dumps({'status':'succeeded','result':'done'}))
    os._exit(0)
os._exit(17)
"""
    command = [sys.executable, "-c", code, str(tmp_path),
               "success" if leader_succeeds else "crash", "atas_geometry.service.isolated_worker"]
    if leader_succeeds:
        assert run_isolated(command, folder=tmp_path, env=os.environ.copy(), deadline_seconds=5)["result"] == "done"
    else:
        with pytest.raises(RuntimeError, match="exited 17"):
            run_isolated(command, folder=tmp_path, env=os.environ.copy(), deadline_seconds=5)
    lease = json.loads((tmp_path / "worker_lease.json").read_text())
    assert lease["owner_contract"] == "isolated_geometry_session_v1"
    assert not lease["active"] and owned_group_members(lease) == []
    child = int((tmp_path / "nested_pid").read_text())
    stat = Path(f"/proc/{child}/stat")
    assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_live_unverified_lease_cannot_be_reconciled(tmp_path):
    from atas_geometry.service.worker_protocol import reconcile_lease
    atomic_json(tmp_path / "worker_lease.json", {"pid": os.getpid(),
        "process_start": process_start(os.getpid()), "active": True})
    with pytest.raises(RuntimeError, match="ownership"):
        reconcile_lease(tmp_path)
    assert json.loads((tmp_path / "worker_lease.json").read_text())["active"]


def test_real_isolated_rpc_returns_structured_failure_without_gpu(tmp_path):
    request = tmp_path / "worker_request.json"
    atomic_json(request, {"rpc": "camera_context", "root": str(tmp_path), "payload": {}})
    state = run_isolated([sys.executable, "-m", "atas_geometry.service.isolated_worker", "--request", str(request)],
        folder=tmp_path, env=os.environ.copy(), deadline_seconds=60)
    assert state["status"] == "failed" and state["stage"] == "camera_context"
    assert "run_id" in state["error"]


def test_prior_terminal_worker_state_is_not_adopted(tmp_path):
    atomic_json(tmp_path / "worker_state.json", {"status": "succeeded", "result": "old"})
    with pytest.raises(RuntimeError, match="without a valid result"):
        run_isolated([sys.executable, "-c", "pass"], folder=tmp_path,
                     env=os.environ.copy(), deadline_seconds=3)


def test_keywords_change_reconstruction_revision_without_touching_old(tmp_path):
    payload = dict(run_id="r", story_id="s", shot_id="q", video_sha256="a" * 64,
                   reconstruction_contract={"dynamic_keywords": ["cat"], "model": "pi3x"})
    a = revision_l0(tmp_path, payload)
    a.mkdir(parents=True)
    (a / "sentinel").write_text("immutable")
    b = revision_l0(tmp_path, {**payload, "reconstruction_contract": {"dynamic_keywords": ["dog"], "model": "pi3x"}})
    assert a != b and not b.exists() and (a / "sentinel").read_text() == "immutable"


def dense_case(tmp_path):
    p = dict(video_sha256="a" * 64, reconstruction_contract={"keywords": ["cat"]})
    np.savez(tmp_path / "cameras.npz", cam_c2w=np.zeros((2, 4, 4)), intrinsics=np.zeros((2, 4)))
    (tmp_path / "video.mp4").write_bytes(b"video")
    for folder, ext in (("depths", "exr"), ("dynamic_mask", "png"), ("sky_mask", "png")):
        (tmp_path / folder).mkdir()
        for i in range(2):
            (tmp_path / folder / f"{i:05d}.{ext}").write_bytes(b"frame")
    atomic_json(tmp_path / "instance_tracks.json", {"frame_count": 2, "tracks": {}})
    return p


def test_camera_file_alone_is_not_a_complete_cache(tmp_path):
    p = dense_case(tmp_path)
    assert not dense_manifest(tmp_path, p, 2)
    assert dense_manifest(tmp_path, p, 2, write=True)
    assert dense_manifest(tmp_path, p, 2)
    (tmp_path / "dynamic_mask/00001.png").unlink()
    assert not dense_manifest(tmp_path, p, 2)


def test_complete_manifest_rejects_changed_segmentation_contract(tmp_path):
    p = dense_case(tmp_path)
    assert dense_manifest(tmp_path, p, 2, write=True)
    assert not dense_manifest(tmp_path, {**p, "reconstruction_contract": {"keywords": ["dog"]}}, 2)


def test_published_observation_is_reused_even_when_cache_contract_changes(tmp_path):
    from atas_geometry.reconstruct.recon_revision import accepted_reconstruction
    from atas_geometry.world.world_state import commit_world
    scene = tmp_path / "run/story/scenes/room"
    l0 = scene / ".world_versions/v1/l0"
    l0.mkdir(parents=True)
    atomic_json(l0 / "observation.json", {"scene_id": "room", "registered": True,
        "event_id": "original_event", "original_video_sha256": "a"*64})
    l1 = l0.parent / "l1.npz"
    np.savez(l1, xyz=np.ones((1, 3)))
    commit_world(scene, version="v1", l1=l1, observation=l0, shot_id="s1",
                 source_sha256="a"*64, expected_version=None)
    payload = dict(run_id="run", story_id="story", scene_id="room", shot_id="s1",
                   event_id="original_event", video_sha256="a"*64, reconstruction_contract={"new": True})
    assert accepted_reconstruction(tmp_path, payload) == l0
    with pytest.raises(ValueError, match="immutable"):
        accepted_reconstruction(tmp_path, {**payload, "video_sha256": "b"*64})
    with pytest.raises(ValueError, match="event_id is immutable"):
        accepted_reconstruction(tmp_path, {**payload, "event_id": "new_event"})
    from atas_geometry.reconstruct.recon_revision import accepted_provenance
    atomic_json(l0.parent / "candidate.json", {"lineage": {"camera": "original_hash"}})
    original = accepted_provenance(tmp_path,
        {**payload, "review_lineage": {"camera": "new_hash"}}, l0)
    assert original["original_review_lineage"] == {"camera": "original_hash"}
    assert original["original_event_id"] == "original_event" and original["original_candidate_job_id"] == "v1"
    assert (l0 / "observation.json").is_file()


def test_interrupted_registration_recovers_once_but_never_retries_rejection(tmp_path):
    from atas_geometry.register.registration_recovery import run_bounded
    ref = tmp_path / "old"
    attempts = []
    def crash(r, folder):
        attempts.append(1)
        folder.mkdir()
        (folder / "partial").write_text("keep")
        raise TimeoutError("worker killed")
    with pytest.raises(TimeoutError):
        run_bounded(tmp_path, stamp={}, references=[ref], execute=crash)
    def complete(r, folder):
        attempts.append(1)
        atomic_json(folder / "fusion_report.json", {"reference_l0": str(r),
            "reconstruction": {"geometry_accepted": False}, "cross_shot_validation": {"accepted": True}})
    _, ledger = run_bounded(tmp_path, stamp={}, references=[ref], execute=complete, recover_interrupted=True)
    assert ledger["status"] == "exhausted" and len(attempts) == 2
    assert list(tmp_path.glob("interrupted_*/partial"))
    run_bounded(tmp_path, stamp={}, references=[ref], execute=lambda *a: pytest.fail("repeat rejected input"),
                recover_interrupted=True)


def test_interrupted_registration_recovery_is_bounded(tmp_path):
    from atas_geometry.register.registration_recovery import run_bounded
    def crash(*a):
        raise TimeoutError("worker killed")
    for _ in range(2):
        with pytest.raises(TimeoutError):
            run_bounded(tmp_path, stamp={}, references=[tmp_path / "old"], execute=crash,
                        recover_interrupted=True)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        run_bounded(tmp_path, stamp={}, references=[tmp_path / "old"], execute=crash,
                    recover_interrupted=True)


def test_archive_second_commit_contains_both_observations(tmp_path):
    from atas_geometry.world.archive import archive_story
    from atas_geometry.world.world_state import commit_world
    hot, archive = tmp_path / "hot", tmp_path / "archive"
    scene = hot / "run/story/scenes/room"
    for i in (1, 2):
        folder = scene / ".world_versions" / f"v{i}"
        obs = folder / "l0"
        obs.mkdir(parents=True)
        atomic_json(obs / "observation.json", {"registered": True, "shot_id": f"s{i}"})
        np.savez(folder / "l1.npz", xyz=np.ones((i, 3)))
        commit_world(scene, version=f"v{i}", l1=folder / "l1.npz", observation=obs,
                     shot_id=f"s{i}", source_sha256=str(i), expected_version=None if i == 1 else "v1")
        archive_story(hot, archive, "run", "story")
    copied = archive / "run/story/scenes/room"
    state = json.loads((copied / "world_state.json").read_text())
    assert state["version"] == "v2" and set(state["observations"]) == {"s1", "s2"}
    assert all((copied / row["l0"] / "observation.json").is_file() for row in state["observations"].values())
    assert (copied / ".world_versions/v1/l1.npz").exists()


def test_health_remains_responsive_during_owned_worker_spin(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import atas_geometry.service.app as module
    from atas_geometry.service.config import Settings
    monkeypatch.setenv("GEOMETRY_HOT_ROOT", str(tmp_path / "hot"))
    monkeypatch.setenv("GEOMETRY_ARCHIVE_ROOT", "")
    class Pool:
        def __init__(self, **kw): pass
    monkeypatch.setattr(module, "GpuPool", Pool)
    app = module.create_app(Settings.from_env())
    errors = []
    def spin():
        try:
            run_isolated([sys.executable, "-c", "while True: pass", "atas_geometry.service.isolated_worker"],
                         folder=tmp_path / "spin", env=os.environ.copy(), deadline_seconds=.5)
        except TimeoutError:
            errors.append("reaped")
    thread = threading.Thread(target=spin)
    thread.start()
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
    thread.join(timeout=6)
    app.state.runtime._stop.set()
    assert errors == ["reaped"]


def test_optional_context_depth_is_unknown_not_a_focal_failure():
    from test_cinematic_camera import plan
    from atas_geometry.render.cinematic_camera import ground_subjects, focal_geometry
    h, w = 40, 60
    y, x = np.indices((h, w))
    depth = np.full((h, w), 3.)
    xyz = np.column_stack(((x.ravel()+.5-w/2)/40*3, (y.ravel()+.5-h/2)/40*3, depth.ravel()))
    observation = SimpleNamespace(xyz=xyz, foreground=np.ones(h*w))
    recon = SimpleNamespace(depth_at=lambda _: depth, unproject=lambda _: observation,
                            exclude_sky=True, sky_files=[])
    intent = plan()
    intent["subjects"].append({**intent["subjects"][0], "label": "bowl", "kind": "static",
                               "part": "object", "framing_role": "context"})
    points, focus, dynamics, audits = ground_subjects(recon, 0, intent)
    assert len(audits) == 2 and audits[1]["points"] == 0 and audits[1]["status"] == "unknown"
    groups = np.split(points, np.cumsum([a["points"] for a in audits])[:-1])
    focal, target = focal_geometry(points, intent, groups)
    assert len(groups) == 2 and len(groups[1]) == 0 and len(focal) == len(points)
    intent["subjects"][1]["framing_role"] = "focal"
    with pytest.raises(ValueError, match="seed depth"):
        ground_subjects(recon, 0, intent)


def test_memory_manifest_exposes_time_local_dynamic_without_dense_transfer(tmp_path):
    from atas_geometry.world.memory_manifest import observation_manifest
    dense_case(tmp_path)
    atomic_json(tmp_path / "observation.json", {"source_fps": 30, "original_frame_count": 2,
        "event_id": "event", "registered": True, "coordinate_space": "scene_world"})
    result = observation_manifest(tmp_path, tmp_path / "l1.npz", payload={"shot_id": "s1"})
    assert result["event_id"] == "event" and result["publication_status"] == "pending_review"
    assert observation_manifest(tmp_path, tmp_path / "l1.npz",
        payload={"shot_id": "s1", "event_id": "wrong_request"})["event_id"] == "event"
    assert result["l2"]["frame_count"] == 2 and result["l2"]["complete_per_frame"]
    assert result["l2"]["representation"] == "time_local_rgbd_masks"
    assert result["l2"]["registered"] and result["l2"]["time_end_exclusive"] == 2/30


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_home_incremental_index_is_exactly_equal_to_previous_set(tmp_path, monkeypatch, device_name):
    import torch
    import atas_geometry.memory.fuse as module
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU verification requires an explicitly assigned visible GPU")
    device = torch.device(device_name)
    rng = np.random.default_rng(41)
    base = rng.uniform(-2, 2, (5000, 3)).astype(np.float32)
    frames = []
    for i in range(5):
        points = np.concatenate((base, rng.uniform(-2-i*.1, 2+i*.1, (3000, 3)))).astype(np.float32)
        color = rng.integers(0, 256, (len(points), 3), dtype=np.uint8)
        frames.append((points, color))
    monkeypatch.setattr(module, "_static_frame_points", lambda recon, i: frames[i])
    voxel = .02
    origin = np.floor(np.percentile(frames[0][0].astype(np.float64), 1., axis=0)/voxel)*voxel
    seen, expected_xyz, expected_rgb = set(), [], []
    for points, color in frames:
        xyz, rgb, keys = module._unique_voxels(points, color, origin, voxel, device)
        indices = [i for i, k in enumerate(keys.tolist()) if int(k) not in seen]
        expected_xyz.append(xyz[indices]); expected_rgb.append(rgb[indices])
        seen.update(int(keys[i]) for i in indices)
    xyz, rgb, actual_origin, count = module._accumulate_home_voxels(None, voxel=voxel,
        frames=list(range(len(frames))), device=device)
    np.testing.assert_array_equal(xyz, np.concatenate(expected_xyz))
    np.testing.assert_array_equal(rgb, np.concatenate(expected_rgb))
    np.testing.assert_array_equal(actual_origin, origin)
    assert count == sum(len(xyz) for xyz, _ in frames)
