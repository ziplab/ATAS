import json
import imageio.v2 as imageio
import numpy as np
import pytest
from atas_geometry.render.camera_context import camera_context
from types import SimpleNamespace

def test_historical_frame_selection_is_addressed_and_bounds_checked(tmp_path):
    source=tmp_path/'run/story/shots/source/l0';source.mkdir(parents=True)
    with imageio.get_writer(str(source/'video.mp4'),fps=6,macro_block_size=1) as writer:
        for i in range(6):writer.append_data(np.full((16,16,3),i*20,dtype=np.uint8))
    np.savez(source/'cameras.npz',cam_c2w=np.repeat(np.eye(4)[None],6,axis=0))
    scene=tmp_path/'run/story/scenes/room';scene.mkdir(parents=True)
    (scene/'l1.npz').touch();(scene/'home_shot_id.txt').write_text('source')
    (source/'observation.json').write_text(json.dumps(dict(run_id='run',story_id='story',scene_id='room',shot_id='source',registered=True)))
    p=dict(run_id='run',story_id='story',scene_id='room',shot_id='target')
    a=camera_context(tmp_path,p);b=camera_context(tmp_path,{**p,'camera_grounding_frame_index':2})
    assert a['frame_index']==5 and b['frame_index']==2 and a['context_id']!=b['context_id']
    for index in (-1,6,True):
        with pytest.raises(ValueError,match='frame index'):camera_context(tmp_path,{**p,'camera_grounding_frame_index':index})

def test_continuous_endpoint_requires_committed_same_scene_observation(tmp_path,monkeypatch):
    from atas_geometry.render import camera_runtime as r
    from atas_geometry.world import world_state
    source=tmp_path/'observation';source.mkdir()
    meta=source/'observation.json'
    meta.write_text(json.dumps({'registered':True,'scene_id':'room'}))
    monkeypatch.setattr(world_state,'committed_observation',lambda *args:source)
    monkeypatch.setattr(r,'SceneEventRecon',lambda *args:SimpleNamespace(count=150))
    seen=[]
    monkeypatch.setattr(r,'_reference_camera_at',lambda recon,w,h,index:(seen.append(index) or 'registered_camera'))
    assert r.registered_endpoint_camera(tmp_path,'run','story','room','shot',1280,720)=='registered_camera'
    assert seen==[149]
    assert r.registered_endpoint_camera(tmp_path,'run','story','other_room','shot',1280,720) is None
    monkeypatch.setattr(world_state,'committed_observation',lambda *args:None)
    assert r.registered_endpoint_camera(tmp_path,'run','story','room','shot',1280,720) is None
