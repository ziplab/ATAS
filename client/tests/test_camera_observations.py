import json
from types import SimpleNamespace
import numpy as np
import pytest
from atas.shot.camera_planner import plan_camera, observation_indices
def test_recapture_does_not_change_event_time():
    assert observation_indices({'frame_index':10,'recon_frame_count':150},{'render_mode':'recapture'})==[10]
    assert observation_indices({'frame_index':149,'recon_frame_count':150},{})==[149,74,0]

def test_selected_historical_frame_is_bound_to_gpu_context_and_cached(tmp_path,monkeypatch):
    from atas.shot import camera_planner as c
    from atas.providers import vlm
    source=tmp_path/'story_story/source';source.mkdir(parents=True)
    (source/'result.mp4').write_bytes(b'film');out=tmp_path/'target';out.mkdir()
    base=dict(context_id='last',source_shot_id='source',frame_index=9,recon_frame_count=10)
    calls=[]
    def ctx(intent):
        n=intent.get('camera_grounding_frame_index',9)
        return {**base,'frame_index':n,'context_id':str(n)}
    monkeypatch.setattr(c,'reference_frame',lambda v,d:(np.full((8,8,3),d['frame_index'],dtype=np.uint8),d['frame_index']))
    monkeypatch.setattr('atas.world.identity.appearance_pairs',lambda s:[])
    monkeypatch.setattr('atas.request.planning_context',lambda s:{})
    response=dict(schema_version=1,status='ok',reference_observation_index=1,
        subjects=[dict(label='Alice',kind='dynamic',part='face',visible=True,confidence=.95,bbox=[.1,.1,.8,.8],anchor_uv=[.4,.4])],
        framing=dict(side_degrees=0,elevation_degrees=0,screen_position=[.5,.5],vertical_fraction=.6,horizontal_fraction=.6),
        motion=dict(name='locked',degrees=0,travel_ratio=0))
    def chat(*args,images,**kwargs):
        calls.append([int(im[0,0,0]) for im in images]);return json.dumps(response)
    monkeypatch.setattr(vlm,'chat_with_retry',chat)
    shot=SimpleNamespace(story_id='story',shot_id='target',entity_ids=['Alice'],prompt='Alice',raw={},
        shot_perspective='',shot_size='close_up',camera_angle='',camera_move_name='locked',camera_side_degrees=0,camera_movement_degrees=0)
    g=SimpleNamespace(camera_context=ctx)
    a=plan_camera(g,{'camera_relation':'continuous'},shot,tmp_path,out)
    b=plan_camera(g,{'camera_relation':'continuous'},shot,tmp_path,out)
    assert a==b and calls==[[9,4,0]]
    assert a['camera_grounding_frame_index']==4
    assert a['camera_plan']['context']['context_id']=='4'
    assert a['camera_relation']=='continuous'

def test_missing_evidence_never_retries_same_images_even_on_resume(tmp_path):
    from atas.runner.ai_recovery import corrected_json, AIRecoveryExhausted
    from atas.shot.camera_planner import _parse
    calls=[]
    def generate(history):
        calls.append(1)
        return json.dumps(dict(schema_version=1,status='unresolved',reason='actor absent in all three frames'))
    for _ in range(2):
        with pytest.raises(AIRecoveryExhausted):corrected_json(tmp_path,'key',generate,_parse)
    assert calls==[1]

def test_planned_prop_is_not_observed_or_allowed_in_recapture():
    from atas.shot.camera_planner import validate_for_shot
    shot=SimpleNamespace(raw={},entity_ids=['Alice'],camera_move_name='locked',prompt='Alice holds a wooden palette')
    plan=dict(motion={'name':'locked'},subjects=[dict(label='Alice',kind='dynamic',framing_role='focal')],
        planned_props=[dict(label='wooden palette',anchor_label='Alice',screenplay_evidence='holds a wooden palette',
                            placement='in her visible hand',status='planned_unobserved')])
    assert validate_for_shot(plan,shot,{} ) is plan
    with pytest.raises(ValueError,match='recapture'):validate_for_shot(plan,shot,{'render_mode':'recapture'})
    plan['planned_props'][0]['label']='Alice'
    with pytest.raises(ValueError,match='impersonate'):validate_for_shot(plan,shot,{})
