import pytest
from atas_geometry.reconstruct import vista4d as module
@pytest.mark.parametrize('pixels,chunk,low,expected',[
    (255000,48,True,(160000,24)),
    (255000,48,False,(255000,48)),
    (120000,16,True,(120000,16)),
    (160000,24,True,(160000,24)),
])
def test_effective_settings(pixels,chunk,low,expected):
    assert module.effective_pi3_settings(pixels,chunk,low)==expected

@pytest.mark.parametrize('receipt,expected',[
    ({},(160000,24)),
    ({'dense_complete.json':{'reconstruction_contract':{'pi3_pixel_limit':'120000',
        'pi3_vo_chunk_size':'16','low_vram':'True'}}},(120000,16)),
    ({'reconstruction_effective.json':{'pi3_pixel_limit':180000,'pi3_vo_chunk_size':32}},(180000,32)),
])
def test_initial_reference_uses_actual_source_settings(tmp_path,monkeypatch,receipt,expected):
    import json,subprocess
    from types import SimpleNamespace
    from atas_geometry.reconstruct import initial_world
    source=tmp_path/'source';source.mkdir()
    folder=tmp_path/'repair'
    for name,data in receipt.items():
        (source/name).write_text(json.dumps(data))
    settings=SimpleNamespace(pi3_pixel_limit=255000,pi3_vo_chunk_size=48,low_vram=True,
        vista4d_python='python',vista4d_repo=tmp_path,pi3_model_id='model')
    def run(command,**kwargs):
        assert int(command[command.index('--pixel-limit')+1])==expected[0]
        assert int(command[command.index('--first-window')+1])==expected[1]
        (folder/'initial_world.json').write_text(json.dumps(dict(status='unchanged',l0=str(source))))
    monkeypatch.setattr(subprocess,'run',run)
    assert initial_world.run_initial_world(source,folder,settings=settings,gpu_index=6)[0]==source


def test_single_pass_falls_back_to_windows_for_long_clips(monkeypatch):
    monkeypatch.delenv('GEOMETRY_PI3_MAX_SINGLE_PASS_FRAMES', raising=False)
    assert module.effective_pi3_settings(255000, 0, False, num_frames=120) == (255000, 0)
    assert module.effective_pi3_settings(255000, 0, False, num_frames=300) == (255000, 48)
    assert module.effective_pi3_settings(255000, 0, True, num_frames=300) == (160000, 24)
    monkeypatch.setenv('GEOMETRY_PI3_MAX_SINGLE_PASS_FRAMES', '400')
    assert module.effective_pi3_settings(255000, 0, False, num_frames=300) == (255000, 0)


def test_single_pass_falls_back_to_windows_for_long_clips(monkeypatch):
    monkeypatch.delenv('GEOMETRY_PI3_MAX_SINGLE_PASS_FRAMES', raising=False)
    assert module.effective_pi3_settings(255000, 0, False, num_frames=120) == (255000, 0)
    assert module.effective_pi3_settings(255000, 0, False, num_frames=300) == (255000, 48)
    assert module.effective_pi3_settings(255000, 0, True, num_frames=300) == (160000, 24)
    monkeypatch.setenv('GEOMETRY_PI3_MAX_SINGLE_PASS_FRAMES', '400')
    assert module.effective_pi3_settings(255000, 0, False, num_frames=300) == (255000, 0)
