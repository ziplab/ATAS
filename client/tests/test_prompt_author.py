import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from atas.shot import prompt_author as p
from atas.world import scene_state as s
from atas.request import load_plan
from atas.providers.wan.generation import build_request
@pytest.fixture
def case(tmp_path):
    path=tmp_path/'story.yaml'
    path.write_text('shots:\n  - id: shot2\n    location_id: room\n    duration: 4\n    prompt: Alice speaks to Bob.\n    entity_ids: [Alice]\n    camera_move_name: pull_back\n')
    refs={}
    for name in ('Alice','Bob'):
        target=tmp_path/(name+'.png');Image.new('RGB',(32,24)).save(target);refs[name]=str(target)
    shot=s.prepare(load_plan(path).shots[0],{'entities':{'Bob':{'label':'Bob','presence':'present',
        'location':'bedside','entity_id':'bob','source_shot_id':'shot1'}}},refs)
    first=tmp_path/'opening.png'
    draft,_,_=build_request(shot,output_dir=tmp_path,geometry_video=None,continuity_still=None,first_frame=first)
    return shot,draft,first


def output(draft,generate=True):
    return {'status':'ready','first_frame_prompt':'Opening image: Alice on bed, Bob beside it.' if generate else '',
        'video_prompt':'A smooth pullback, Alice speaks to Bob. '+ ' '.join(r.prompt_handle for r in draft.condition.references),
        'conflicts':[],'character_decisions':[{'label':n,'treatment':'keep position'} for n in ('Alice','Bob')]}


def test_ai_prompts_pass_through_without_template_append(case,tmp_path,monkeypatch):
    shot,draft,first=case;response=output(draft);calls=[]
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',lambda *a,**k:calls.append(a[0]) or json.dumps(response))
    shot=p.author_prompts(shot,tmp_path,references=draft.condition.references)
    # Same source inputs hit the cache without a second model call.
    p.author_prompts(shot,tmp_path,references=draft.condition.references)
    Image.new('RGB',(32,24)).save(first)
    actual,_,_=build_request(shot,output_dir=tmp_path,geometry_video=None,continuity_still=None,first_frame=first,require_files=True)
    assert actual.prompt==response['video_prompt'] and len(calls)==1
    assert 'Scene continuity facts' not in actual.prompt


def test_author_receives_exact_scene_roster_and_actionable_correction(case,tmp_path,monkeypatch):
    shot,draft,_=case;calls=[]
    def chat(text,**kwargs):
        data=json.loads(text.split('\nINPUT:\n')[1]);calls.append(data)
        assert set(data['required_character_labels'])=={'Alice','Bob'}
        response=output(draft)
        if len(calls)==1:
            response['character_decisions'].append({'label':'Alice alias','treatment':'same person'})
        return json.dumps(response)
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',chat)
    authored=p.author_prompts(shot,tmp_path,references=draft.condition.references)
    assert len(calls)==2
    assert 'Alice alias' in json.dumps(calls[-1]['response_recovery'])
    assert 'required_character_labels' in json.dumps(calls[-1]['response_recovery'])
    assert len(authored.raw['_generation_prompts']['character_decisions'])==2


def test_prompt_cache_invalidates_on_visual_change(case,tmp_path,monkeypatch):
    shot,draft,_=case;calls=[]
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',lambda *a,**k:calls.append(1) or json.dumps(output(draft)))
    p.author_prompts(shot,tmp_path,references=draft.condition.references)
    Image.new('RGB',(32,24),'red').save(tmp_path/'Bob.png')
    p.author_prompts(shot,tmp_path,references=draft.condition.references)
    assert len(calls)==2


@pytest.mark.parametrize('fault',['missing_character','bad_handle','conflicting_ready','redraw_source'])
def test_author_validation(fault,case):
    _,draft,_=case;r=output(draft)
    if fault=='missing_character':r['character_decisions'].pop()
    if fault=='bad_handle':r['video_prompt']+=' @图片99'
    if fault=='conflicting_ready':r['conflicts']=[{'issue':'contradiction'}]
    with pytest.raises(ValueError):p.validate(r,['Alice','Bob'],p.video_manifest(draft.condition.references),fault!='redraw_source')


def test_persistent_conflicts_are_resolved_by_role_not_fatal(case,tmp_path,monkeypatch):
    shot,draft,_=case;r=output(draft);r.update(status='needs_revision',conflicts=[{'issue':'old pose at the door','requires':'choose'}])
    prompts=[]
    def chat(prompt,**k):
        prompts.append(prompt)
        return json.dumps(r)   # keeps reporting the conflict, also when asked to resolve it
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',chat)
    shot=p.author_prompts(shot,tmp_path,references=draft.condition.references)
    authored=shot.raw['_generation_prompts']
    assert authored['status']=='ready' and authored['resolved_conflicts']==r['conflicts']
    assert authored['conflict_resolution'].startswith('screenplay-geometry-history')
    assert sum('FINAL CONFLICT RESOLUTION' in q for q in prompts)==1 and len(prompts)==4


def test_final_reference_mapping_cannot_change(case,tmp_path,monkeypatch):
    shot,draft,_=case
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',lambda *a,**k:json.dumps(output(draft)))
    shot=p.author_prompts(shot,tmp_path,references=draft.condition.references)
    with pytest.raises(ValueError,match='packing changed'):p.submitted_video_prompt(shot,draft.condition.references[:-1])


def test_image_generation_uses_only_authored_prompt_and_bound_refs(case,tmp_path,monkeypatch):
    shot,draft,first=case;r=output(draft);calls=[]
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',lambda *a,**k:json.dumps(r))
    shot=p.author_prompts(shot,tmp_path,references=draft.condition.references)
    def generate(self,prompt,dest,**kw):
        calls.append((prompt,kw['references']));Image.new('RGB',(32,18)).save(dest)
    monkeypatch.setattr('atas.providers.gpt_image.GPTImage2Generator.__init__',lambda *a,**k:None)
    monkeypatch.setattr('atas.providers.gpt_image.GPTImage2Generator.generate',generate)
    from atas.shot.stills import generate_cold_start_frame
    generate_cold_start_frame(shot,first);generate_cold_start_frame(shot,first)
    assert len(calls)==1 and calls[0][0].startswith(r['first_frame_prompt']) and '16:9' in calls[0][0]
    assert len(calls[0][1])==2


def test_first_frame_review_does_not_decode_image_as_video(case,tmp_path,monkeypatch):
    shot,_,first=case;Image.new('RGB',(32,24)).save(first)
    from atas.review.film_review import review_shot
    checks=[{'criterion':c,'status':'not_applicable' if c in ('motion','event') else 'pass',
        'confidence':.99,'severity':'minor','evidence':'opening image'} for c in ('subject','framing','motion','action','event','world_consistency')]
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',lambda *a,**k:json.dumps({'checks':checks,'summary':'ok'}))
    result=review_shot(shot,first,tmp_path,stage='first_frame')
    assert result['decision']=='pass' and result['input']['review_frames']==[{'frame':0,'time_seconds':0.}]








def test_repair_sends_actual_base_and_preserves_identity_refs(case,tmp_path,monkeypatch):
    shot,draft,_=case
    failed=tmp_path/'previous.png';Image.new('RGB',(32,24),'red').save(failed)
    feedback={'stage':'first_frame','checks':[{'criterion':'action','status':'fail'},
        {'criterion':'subject','status':'pass'},{'criterion':'framing','status':'pass'}]}
    response=output(draft)
    response.update(first_frame_prompt='Edit Image 1: change only the expression; preserve both identities.',
        repair_plan={'change':[{'criterion':'action','instruction':'startled expression'}],
            'preserve':[{'criterion':'subject','instruction':'same Alice and Bob'},
                        {'criterion':'framing','instruction':'same viewpoint'}]})
    inputs=[];calls=[]
    def chat(text,**kw):
        inputs.append(json.loads(text.split('\nINPUT:\n')[1]));return json.dumps(response)
    monkeypatch.setattr('atas.providers.vlm.chat_with_retry',chat)
    revised=p.author_prompts(shot,tmp_path/'repair',references=draft.condition.references,
        revision_feedback=feedback,failed_first_frame=failed)
    refs=revised.raw['_generation_prompts']['image_references']
    assert refs[0]['path']==str(failed) and refs[0]['is_repair_base']
    assert {n for r in refs[1:] for n in r['people']}=={'Alice','Bob'}
    def generate(self,prompt,dest,**kw):
        calls.append(kw['references']);Image.new('RGB',(32,18),'blue').save(dest)
    monkeypatch.setattr('atas.providers.gpt_image.GPTImage2Generator.__init__',lambda *a,**k:None)
    monkeypatch.setattr('atas.providers.gpt_image.GPTImage2Generator.generate',generate)
    dest=tmp_path/'repair/opening.png';p.generate_authored_still(revised,dest)
    assert calls[0][0]==failed and len(calls[0])==3
    receipt=json.loads(dest.with_suffix('.ai_prompt_receipt.json').read_text())
    assert receipt['reference_manifest'][0]['sha256']==p.file_hash(failed)
    assert inputs[0]['revision']['mode']=='targeted_opening_edit_v1'


@pytest.mark.parametrize('fault',['missing_change','missing_preserve','missing_base','invalid_slot'])
def test_targeted_edit_must_cover_review_and_valid_base(fault):
    response={'status':'ready','first_frame_prompt':'Edit Image 1',
        'repair_plan':{'change':[{'criterion':'action','instruction':'fix'}],
                       'preserve':[{'criterion':'subject','instruction':'keep'}]}}
    if fault=='missing_change':response['repair_plan']['change']=[]
    if fault=='missing_preserve':response['repair_plan']['preserve']=[]
    if fault=='missing_base':response['first_frame_prompt']='Redraw scene'
    if fault=='invalid_slot':response['first_frame_prompt']='Edit Image 1 with Image 99'
    feedback={'checks':[{'criterion':'action','status':'fail'},{'criterion':'subject','status':'pass'}]}
    with pytest.raises(ValueError):p.validate_repair_plan(response,feedback,[{'handle':'Image 1'}])
