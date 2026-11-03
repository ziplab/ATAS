from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from atas_geometry.world.recapture import event_window, record_observation, load_source, SceneEventRecon
from atas_geometry.memory.vista4d_layers import ReconStore
def payload(**extra):
    return dict(run_id="run", story_id="story", scene_id="room", shot_id="new",
                source_shot_id="source", source_event_id="event", event_id="event",
                duration=2., **extra)


def test_event_window():
    assert event_window(payload(source_time_start=1., source_time_end=3.), 4.) == (1., 3., 2.)
    for changes in ({"source_time_start": -1}, {"source_time_end": 5},
                    {"source_time_start": float("nan")}, {"source_time_end": 3}):
        with pytest.raises(ValueError):
            event_window(payload(**changes), 4.)


def test_registration_and_source_identity(tmp_path):
    l0 = tmp_path / "run/story/shots/source/l0"
    l0.mkdir(parents=True)
    src = {**payload(), "shot_id": "source", "duration": 2.}
    record_observation(l0, src, {"home": True})
    source, meta, _ = load_source(tmp_path, payload())
    assert source == l0 and meta["raw_to_scene"]["scale"] == 1
    for field, value in (("event_id", "other"), ("scene_id", "other"), ("source_shot_id", "../source")):
        with pytest.raises(ValueError):
            load_source(tmp_path, {**payload(), field: value})
    record_observation(l0, src, {"merged": False})
    with pytest.raises(ValueError, match="not registered"):
        load_source(tmp_path, payload())


def test_legacy_similarity_record(tmp_path):
    raw = np.eye(4)[None]
    raw[0, :3, 3] = [1, 2, 3]
    aligned = raw.copy()
    aligned[0, :3, 3] = 2 * raw[0, :3, 3] + [5, 0, 0]
    np.savez(tmp_path / "cameras_pi3_raw.npz", cam_c2w=raw)
    np.savez(tmp_path / "cameras.npz", cam_c2w=aligned)
    record_observation(tmp_path, payload(), {"merged": True, "align_scale": 2.})
    transform = json.loads((tmp_path / "observation.json").read_text())["raw_to_scene"]
    np.testing.assert_allclose(transform["translation"], [5, 0, 0])


def test_unproject_scales_raw_depth_once(tmp_path, monkeypatch):
    @dataclass
    class Batch:
        xyz: np.ndarray
    def initialize(self, *args, **kwargs):
        self.cam_c2w = np.eye(4)[None]
        self.cam_c2w[0, :3, 3] = [1, 0, 0]
        self.count = 1
        self.dynamic_files = [Path("mask")]
    def unproject(self, index):
        np.testing.assert_allclose(self.cam_c2w[0, :3, 3], [1, 0, 0])
        return Batch(np.array([[1, 0, 2.]]))
    monkeypatch.setattr(ReconStore, "__init__", initialize)
    monkeypatch.setattr(ReconStore, "unproject", unproject)
    meta = {"duration": 2., "raw_to_scene": {"scale": 2., "rotation": np.eye(3).tolist(), "translation": [5, 0, 0]}}
    recon = SceneEventRecon(tmp_path, meta)
    np.testing.assert_allclose(recon.unproject(0).xyz, [[7, 0, 4]])
    np.testing.assert_allclose(recon.cam_c2w[0, :3, 3], [7, 0, 0])


@pytest.mark.skipif("ATAS_TEST_GPU" not in os.environ, reason="explicit idle GPU required")
def test_time_local_dynamic_single_depth_buffer():
    import torch
    from atas_geometry.render.render import render_l1_gpu
    from atas_geometry.memory.observation import CameraCalibration
    camera = CameraCalibration("test", np.eye(4), np.array([[8., 0, 8], [0, 8, 8], [0, 0, 1]]), (16, 16))
    static = SimpleNamespace(xyz=np.array([[0., 0, 2]]), rgb=np.array([[255, 0, 0]], dtype=np.uint8))
    dynamic = [(np.array([[0., 0, z]]), np.array([[0, 255, 0]], dtype=np.uint8)) for z in (1., 3.)]
    sink = []
    frames, report = render_l1_gpu(static, [camera, camera], width=16, height=16,
        device=torch.device("cuda:" + os.environ["ATAS_TEST_GPU"]), dynamic_frames=dynamic, frame_sink=sink.append)
    assert frames == [] and len(sink) == 2
    np.testing.assert_array_equal(sink[0][8, 8], [0, 255, 0])
    np.testing.assert_array_equal(sink[1][8, 8], [255, 0, 0])
    np.testing.assert_array_equal(sink[0][0, 0], [127, 127, 127])
