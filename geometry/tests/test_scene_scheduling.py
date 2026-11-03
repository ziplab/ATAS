import json
import threading
from types import SimpleNamespace

from atas_geometry.service.jobs import GeometryRuntime
from atas_geometry.service import paths


def runtime_with(jobs):
    runtime = GeometryRuntime.__new__(GeometryRuntime)
    runtime._scene_guard, runtime._scene_locks = threading.Lock(), {}
    runtime._queue = list(jobs)
    runtime.store = SimpleNamespace(get=lambda job_id: {"payload": jobs[job_id]})
    return runtime


def test_a_busy_scene_does_not_block_jobs_of_other_scenes():
    jobs = {"a2": {"run_id": "r", "story_id": "s", "scene_id": "A"},
            "b1": {"run_id": "r", "story_id": "s", "scene_id": "B"}}
    runtime = runtime_with(jobs)
    busy = runtime.scene_lock("r", "s", "A")
    holder = threading.Thread(target=busy.acquire)   # another worker is running scene A
    holder.start(); holder.join()
    job_id, lock = runtime._claim()
    assert job_id == "b1" and runtime._queue == ["a2"]
    lock.release()


def test_home_shot_is_recovered_from_the_committed_world(tmp_path):
    scene = paths.scene_dir(tmp_path, "r", "s", "A")
    scene.mkdir(parents=True)
    (scene / "world_state.json").write_text(json.dumps({"observations": {"shot_1": {}, "shot_2": {}}}))
    assert paths.resolve_home_shot(tmp_path, "r", "s", "A") == "shot_1"
    assert paths.home_shot_marker(tmp_path, "r", "s", "A").read_text().strip() == "shot_1"
