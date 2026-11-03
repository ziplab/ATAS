import threading
from pathlib import Path

import pytest

from atas_geometry.service import gpu_pool as g
from atas_geometry.service.resource_errors import ResourceUnavailableError, log_resource_error, error_fields
def cards(free=30000):
    return [g.GPUInfo(i, "test", 32607, free, 0) for i in range(1, 8)]


def test_candidate_pool_is_not_truncated_to_concurrency_limit(monkeypatch):
    monkeypatch.setattr(g, "query_gpus", lambda: cards())
    pool = g.GpuPool(configured=tuple(range(1, 8)), max_gpus=3, poll_seconds=.001)
    assert pool.indices == tuple(range(1, 8))
    for expected in (1, 2, 3):
        assert pool.acquire(stop=threading.Event()) == expected
    with pytest.raises(ResourceUnavailableError) as error:
        pool.acquire(stop=threading.Event(), timeout_seconds=.01)
    assert error.value.error_code == "gpu_capacity_unavailable" and len(pool._busy) == 3
    pool.release(1)
    assert pool.acquire(stop=threading.Event()) == 4


def test_insufficient_vram_never_leases_card(monkeypatch):
    monkeypatch.setattr(g, "query_gpus", lambda: cards(15000))
    pool = g.GpuPool(configured=tuple(range(1, 8)), poll_seconds=.001)
    with pytest.raises(ResourceUnavailableError) as error:
        pool.acquire(min_free_mib=24576, stop=threading.Event(), timeout_seconds=.01)
    assert pool._busy == set()
    assert error.value.fields()["retryable"] is True
    assert error.value.resource_details["min_free_mib"] == 24576


def test_waits_for_resources_then_chooses_suitable_physical_card(monkeypatch):
    calls = []
    def snapshot():
        calls.append(1)
        rows = cards(5000)
        if len(calls) >= 3:
            rows[-1] = g.GPUInfo(7, "test", 32607, 30000, 0)
        return rows
    monkeypatch.setattr(g, "query_gpus", snapshot)
    pool = g.GpuPool(configured=tuple(range(1, 8)), poll_seconds=.001)
    assert pool.acquire(min_free_mib=24576, stop=threading.Event(), timeout_seconds=1) == 7
    assert pool._busy == {7}


def test_configuration_keeps_seven_candidates_and_three_workers(monkeypatch):
    from atas_geometry.service.config import Settings
    monkeypatch.setenv("GEOMETRY_GPU_IDS", "1,2,3,4,5,6,7")
    monkeypatch.setenv("GEOMETRY_MAX_GPUS", "3")
    monkeypatch.setenv("GEOMETRY_RECON_MIN_FREE_MIB", "15000")
    settings = Settings.from_env()
    assert settings.gpu_ids == tuple(range(1, 8)) and settings.max_gpus == 3
    assert settings.recon_min_free_mib == 15000  # explicit setting honoured; no hidden floor


def test_cuda_oom_log_preserves_physical_mapping_and_ignores_old_attempt(tmp_path):
    path = tmp_path / "recon.log"
    path.write_text("torch.OutOfMemoryError: CUDA out of memory. GPU 0 is full.\n")
    error = log_resource_error(path, physical_gpu_index=7, stage="recon")
    assert error.error_code == "cuda_out_of_memory"
    assert error.resource_details["physical_gpu_index"] == 7
    assert error.resource_details["logical_gpu_index"] == 0
    offset = path.stat().st_size
    with path.open("a") as f: f.write("geometry registration rejected\n")
    assert log_resource_error(path, physical_gpu_index=7, stage="recon", start_offset=offset) is None
    assert error_fields(ValueError("no reliable geometry matches")) == {}


@pytest.mark.parametrize("message,code",[
    ("torch.OutOfMemoryError: CUDA out of memory. GPU 0 is full.","cuda_out_of_memory"),
    ("RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`","cuda_linalg_initialization_failed"),
])
def test_vista_failure_is_structured_resource_not_generic_exit(tmp_path, monkeypatch,message,code):
    from atas_geometry.reconstruct import vista4d as v
    captured = {}
    class Process:
        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)
            self.stdout = iter([message+"\n"])
        def wait(self): return 1
    monkeypatch.setattr(v.subprocess, "Popen", Process)
    with pytest.raises(ResourceUnavailableError) as error:
        v.run_recon_and_seg(python_bin=Path("python"), repo_dir=tmp_path, video=tmp_path/"video.mp4",
            output_dir=tmp_path/"l0", keywords=["cat"], gpu_index=7, log_path=tmp_path/"recon.log",
            pi3_model_id="pi3x", sam3_checkpoint=tmp_path/"sam.pt", width=1280, height=720,
            num_frames=150, pi3_pixel_limit=255000, pi3_vo_chunk_size=24, pi3_vo_overlap=6, low_vram=True)
    assert error.value.fields()["retryable"]
    assert error.value.error_code==code
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert captured["command"][captured["command"].index("--num_frames")+1] == "150"


@pytest.mark.parametrize("code,retries", [("cuda_out_of_memory", 1), ("cuda_linalg_initialization_failed", 1), ("gpu_capacity_unavailable", 1)])
def test_native_resource_failure_gets_one_fresh_resource_requeue(tmp_path, code, retries):
    from atas_geometry.service.jobs import GeometryRuntime
    from atas_geometry.service.durable import JobStore
    runtime = GeometryRuntime.__new__(GeometryRuntime)
    runtime.store = JobStore(tmp_path)
    job = runtime.store.create("recon", {"run_id": "r", "story_id": "s", "scene_id": "room"})
    runtime._stop = threading.Event(); runtime._cv = threading.Condition(); runtime._queue = [job["job_id"]]
    runtime._scene_guard = threading.Lock(); runtime._scene_locks = {}
    calls = []
    def run(jid):
        calls.append(jid)
        if len(calls) == 2:
            runtime._stop.set()
        raise ResourceUnavailableError(code, "resource unavailable", physical_gpu_index=1)
    runtime._run = run
    runtime._loop()
    result = runtime.store.get(job["job_id"])
    assert len(calls) == retries+1 and result.get("resource_recovery_count", 0) == retries
    assert result["status"] == "failed" and result["error_code"] == code and result["retryable"]
    assert result["recovery_exhausted"] is True


def test_cusolver_creation_failure_is_retryable_but_not_claimed_oom(tmp_path):
    path=tmp_path/"recon.log"
    message="RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`"
    path.write_text(message+"\n")
    error=log_resource_error(path,physical_gpu_index=2,stage="vista4d_recon")
    assert error.error_code=="cuda_linalg_initialization_failed"
    assert error.fields()["retryable"] is True
    assert error.resource_details["physical_gpu_index"]==2
    assert error.resource_details["logical_gpu_index"]==0
    assert "memory exhausted" not in str(error)
    assert error_fields(RuntimeError(message))["error_code"]==error.error_code
    offset=path.stat().st_size
    with path.open("a") as f:f.write("SVD numerical convergence failed\n")
    assert log_resource_error(path,physical_gpu_index=2,stage="recon",start_offset=offset) is None


@pytest.mark.parametrize("message",[
    "SVD failed to converge: input matrix contained non-finite values",
    "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR during factorization",
    "cross_shot_static_alignment_failed",
])
def test_numeric_geometry_errors_do_not_become_resource_retries(message):
    assert error_fields(RuntimeError(message))=={}
