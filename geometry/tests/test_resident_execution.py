import json
import pytest
from atas_geometry.service.device_mapping import gpu_selector, device_index
from atas_geometry.service.resident_models import ModelCache, resident_command
def test_physical_uuid_mapping_skips_unhealthy_card(monkeypatch):
    monkeypatch.setenv('GEOMETRY_GPU_UUIDS', json.dumps({'3':'GPU-three','7':'GPU-seven'}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-three,GPU-seven')
    assert gpu_selector(7)=='GPU-seven'
    assert device_index(7)==1
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-seven')
    assert device_index(7)==0
    with pytest.raises(RuntimeError): device_index(3)


def test_mapping_disabled_is_backwards_compatible(monkeypatch):
    monkeypatch.delenv('GEOMETRY_GPU_UUIDS', raising=False)
    assert device_index(3)==3
    assert gpu_selector(3)=='3'


def test_cache_reuses_same_model_but_not_different_configuration(monkeypatch):
    class Model:
        def to(self, device): self.device=device; return self
    calls=[]
    def load(model_id='a',device='cuda'):
        calls.append(model_id); return Model()
    cache=ModelCache(); monkeypatch.setattr(cache,'offload',lambda **kw:None)
    wrapped=cache.wrap('pi3',load)
    a=wrapped(); assert wrapped(model_id='a') is a
    assert wrapped(model_id='b') is not a
    assert calls==['a','b']


def test_resident_is_explicit_opt_in(monkeypatch,tmp_path):
    cmd=['python','-m','atas_geometry.register.anchored_fusion','--output','a']
    monkeypatch.delenv('GEOMETRY_RESIDENT_MODELS',raising=False)
    assert resident_command(cmd,repo=tmp_path,vista_repo=tmp_path,log_path=tmp_path/'log')==cmd
    monkeypatch.setenv('GEOMETRY_RESIDENT_MODELS','1')
    converted=resident_command(cmd,repo=tmp_path,vista_repo=tmp_path,log_path=tmp_path/'log')
    assert converted[-3:]==['--','--output','a']


def test_end_archive_policy_does_not_copy_on_every_commit(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from atas_geometry.service.jobs import GeometryRuntime
    monkeypatch.setenv('GEOMETRY_ARCHIVE_POLICY','story_end')
    runtime=GeometryRuntime.__new__(GeometryRuntime)
    runtime.settings=SimpleNamespace(archive_root=tmp_path)
    runtime._maybe_archive(None,'run','story')
    assert not list(tmp_path.iterdir())


def test_model_cache_restores_each_models_precision_context(monkeypatch):
    import torch
    from atas_geometry.service.resident_models import precision_state, restore_precision
    baseline=precision_state()
    class Model:
        def to(self, device): return self
    cache=ModelCache();cache.sam_tf32=(True,True)
    monkeypatch.setattr(cache,'offload',lambda **kw:None)
    def pi3(): return Model()
    def sam():
        torch.set_autocast_enabled('cuda',True)
        torch.set_autocast_dtype('cuda',torch.bfloat16)
        return Model()
    p,s=cache.wrap('pi3',pi3),cache.wrap('sam3',sam)
    try:
        torch.set_autocast_enabled('cuda',False)
        p();s()
        assert torch.is_autocast_enabled('cuda')
        p()
        assert not torch.is_autocast_enabled('cuda')
        s()
        assert torch.is_autocast_enabled('cuda')
    finally: restore_precision(baseline)


def test_worker_code_fingerprint_follows_code_changes(tmp_path):
    from atas_geometry.service.resident_models import code_fingerprint
    (tmp_path / "utils").mkdir(); (tmp_path / "scripts").mkdir()
    module = tmp_path / "utils" / "recon.py"
    module.write_text("A = 1\n")
    before = code_fingerprint(tmp_path)
    assert code_fingerprint(tmp_path) == before
    module.write_text("A = 2\n")
    assert code_fingerprint(tmp_path) != before
