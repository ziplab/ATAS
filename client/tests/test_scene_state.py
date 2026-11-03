import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from atas.world import scene_state as s
from atas.request import load_plan
def shot(tmp_path):
    p=tmp_path/'story.yaml';p.write_text('shots:\n  - id: shot2\n    location_id: room\n    duration: 4\n    start: 7\n    prompt: Close-up Alice talking to Bob.\n    entity_ids: [Alice]\n    camera_move_name: pull_back\n')
    return load_plan(p).shots[0]


def prior():
    return {'entities':{'Bob':{'label':'Bob','entity_id':'bob','presence':'present',
        'source_shot_id':'shot1','last_observed_time':7,'location':'beside the bed'}}}


def test_nonfocal_actor_remains_in_scene_and_references(tmp_path):
    current=s.prepare(shot(tmp_path),prior(),{'Alice':'/Alice.jpg','Bob':'/Bob.jpg'})
    assert current.entity_ids==('Alice',)
    cast=s.cast_shot(current)
    assert set(cast.entity_ids)=={'Alice','Bob'} and Path('/Bob.jpg') in cast.character_reference_images
    assert s.context(current)['entities']['Bob']['presence']=='present'
    assert 'beside the bed' in s.brief(current)


def test_invisible_does_not_depart_or_require_mask_binding(tmp_path):
    current=s.prepare(shot(tmp_path),prior(),{})
    review={'scene_observations':[{'label':'Alice','visibility':'visible'}, {'label':'Bob','visibility':'not_visible'}]}
    assert s.visible_cast(current,review).entity_ids==('Alice',)
    state=s.advance(current,prior(),review,{'bindings':{}},'world2')
    assert state['entities']['Bob']['presence']=='present'
    assert state['graph']['type']=='avwm_entity_scene_graph'


def test_departure_requires_observed_evidence(tmp_path):
    current=s.prepare(shot(tmp_path),prior(),{})
    rows=[{'label':label,'visibility':'not_visible','confidence':.99,'evidence':'offscreen',
        'departure_observed':label=='Bob','departure_evidence':''} for label in ['Alice','Bob']]
    with pytest.raises(ValueError,match='departure'):s.validate_observations(rows,current)
    rows[1]['departure_evidence']='walks through bedroom doorway at frame 90'
    s.validate_observations(rows,current)
    updated=s.advance(current,prior(),{'scene_observations':rows},{'bindings':{}},'v2')
    assert updated['entities']['Bob']['presence']=='departed'


def test_room_state_is_not_carried_from_another_scene(tmp_path):
    current=s.prepare(shot(tmp_path),{}, {})
    assert set(s.context(current)['entities'])=={'Alice'}


def test_event_snapshot_uses_source_members_not_future_cast():
    after=prior();after['entities']['Bob']['source_shot_id']='event1'
    state=s.event_snapshot({},after,'event1',.5)
    assert state['event_source_time']==.5 and state['entities']['Bob']['presence']=='event_member'
    assert 'end pose not reused' in state['entities']['Bob']['location']


def test_wan_prompt_does_not_remove_nonfocal_people(tmp_path):
    from atas.providers.wan.generation import _prompt
    current=s.prepare(shot(tmp_path),prior(),{})
    text=_prompt(current,(),has_geometry=True)
    assert '画面中只出现' not in text and 'Bob' in text and 'beside the bed' in text
    assert 'Alice' in text and 'not an instruction to make a group shot' in text


def test_unknown_visibility_is_not_a_world_publication_gate(tmp_path):
    current=s.prepare(shot(tmp_path),prior(),{})
    visible=s.visible_cast(current,{'scene_observations':[{'label':'Bob','visibility':'uncertain'}]})
    assert not visible.entity_ids


def test_observed_character_needs_no_track_binding_and_failed_action_is_not_completed(tmp_path):
    current=s.prepare(shot(tmp_path),prior(),{})
    review={'decision':'warning','checks':[{'criterion':'action','status':'fail'},
        {'criterion':'event','status':'pass'}], 'scene_observations':[
        {'label':'Alice','visibility':'visible','location':'bedside','evidence':'frame 3'}]}
    state=s.advance(current,prior(),review,{},'world2')
    assert state['entities']['Alice']['location']=='bedside'
    assert not state['entities']['Alice']['identity_verified']
    assert not state['scripted_event_verified']


def test_no_latest_graph_leaks_into_recapture(tmp_path):
    future=prior();future['entities']['Bob']['location']='on the moon'
    historical={'entities':{'Bob':{**prior()['entities']['Bob'],'location':'beside bed'}}}
    current=s.prepare(shot(tmp_path),future,{},recapture_state=historical)
    assert s.context(current)['entities']['Bob']['location']=='beside bed'


def test_film_prompt_receives_history_and_identity_references(tmp_path,monkeypatch):
    import numpy as np
    from atas.review.film_review import review_shot
    from PIL import Image
    current=s.prepare(shot(tmp_path),prior(),{'Bob':str(tmp_path/'Bob.png')})
    Image.new('RGB',(8,8)).save(tmp_path/'Bob.png')
    out=tmp_path/'shot2';out.mkdir();history=tmp_path/'shot1';history.mkdir()
    (history/'result.mp4').write_bytes(b'history');(out/'result.mp4').write_bytes(b'film')
    monkeypatch.setattr('atas.review.film_review.frames',lambda *a,**kw:([np.zeros((8,8,3),np.uint8)],[{'frame':0,'time_seconds':0}]))
    calls=[]
    def chat(text,**kw):
        calls.append(text)
        return json.dumps({'checks':[{'criterion':c,'status':'pass','severity':'minor','confidence':.99,'evidence':'frame0'}
            for c in ['subject','framing','motion','action','event','world_consistency']], 'summary':'ok',
            'scene_observations':[{'label':label,'visibility':'visible','confidence':.99,'evidence':'frame0'} for label in ['Alice','Bob']]})
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',chat)
    result=review_shot(current,out/'result.mp4',out,stage='film')
    assert result['input']['historical_observations'][0]['source_shot_id']=='shot1'
    assert result['input']['appearance_references'][0]['label']=='Bob'







def test_compiler_onstage_list_removes_characters_who_left_by_script():
    from types import SimpleNamespace
    from atas.world import scene_state as st
    previous={'entities':{'zhou':{'label':'zhou','presence':'present','source_shot_id':'a'},
                          'courier':{'label':'courier','presence':'present','source_shot_id':'a'}}}
    raw={'planning_provenance':{'kind':'atas-recursive-compiler-v1'},'onstage_labels':['zhou','xiaoyu']}
    shot=SimpleNamespace(raw=raw,entity_ids=('xiaoyu',),scene_id='cafe',start=20.)
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Shot:
        raw:dict; entity_ids:tuple; scene_id:str; start:float
    ctx=st.context(st.prepare(Shot(raw,('xiaoyu',),'cafe',20.),previous,{}))
    assert set(ctx['entities'])=={'zhou','xiaoyu'} and ctx['left_by_script']==['courier']
    flat=dict(raw,planning_provenance={'kind':'ai-story-planner-v1'})
    ctx=st.context(st.prepare(Shot(flat,('xiaoyu',),'cafe',20.),previous,{}))
    assert 'courier' in ctx['entities']   # other planners keep the conservative rule
