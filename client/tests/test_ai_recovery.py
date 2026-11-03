import json
from types import SimpleNamespace
import pytest
from atas.runner.ai_recovery import corrected_json, AIRecoveryExhausted, camera_failure_recoverable, safe_error
from atas.shot.camera_planner import validate_for_shot
def test_invalid_response_is_corrected_and_cached(tmp_path):
    calls=[]
    def generate(feedback):
        calls.append(feedback)
        return 'bad json' if len(calls)==1 else '{"ok":true}'
    assert corrected_json(tmp_path,'key',generate,json.loads)=={'ok':True}
    assert len(calls)==2 and calls[1]['previous_attempts'][0]['status']=='invalid'
    corrected_json(tmp_path,'key',generate,json.loads)
    assert len(calls)==2

def test_budget_survives_restart_and_execution_feedback(tmp_path):
    calls=[]
    def generate(feedback):calls.append(feedback);return '{}'
    def invalid(_):raise ValueError('missing geometry')
    for _ in range(2):
        with pytest.raises(AIRecoveryExhausted):corrected_json(tmp_path,'bad',generate,invalid)
    assert len(calls)==3
    corrected_json(tmp_path,'good',generate,json.loads)
    corrected_json(tmp_path,'good',generate,json.loads,feedback={'error':'seed outside mask'})
    assert calls[-1]['execution_feedback']['error']=='seed outside mask'

def test_transport_error_is_not_semantic_retry(tmp_path):
    def generate(_):raise ConnectionError('lost connection')
    with pytest.raises(ConnectionError):corrected_json(tmp_path,'key',generate,json.loads)
    assert json.loads((tmp_path/'key/attempt_00.json').read_text())['status']=='call_failed'

def test_valid_negative_audit_is_not_retried(tmp_path):
    calls=[]
    def generate(feedback):
        calls.append(feedback);return '{"decision":"hold"}'
    assert corrected_json(tmp_path,'audit',generate,json.loads)['decision']=='hold'
    assert len(calls)==1

def entrant_case():
    shot=SimpleNamespace(entity_ids=('Alice','Bob'),camera_move_name='locked',
        prompt='Alice turns as Bob enters through the doorway.',raw={'_scene_context':{'entities':{'Alice':{'source_shot_id':'previous'},'Bob':{'presence':'planned'}}}})
    plan={'subjects':[{'label':'Alice','kind':'dynamic'},{'label':'doorway','kind':'static'}],
        'motion':{'name':'locked'},'planned_entrants':[{'label':'Bob','anchor_label':'doorway',
        'screenplay_evidence':'Bob enters through the doorway','placement':'stand inside the visible doorway',
        'status':'planned_unobserved'}]}
    return shot,plan

def test_entrant_is_separate_from_observed_geometry():
    shot,plan=entrant_case();assert validate_for_shot(plan,shot,{}) is plan
    assert all(s['label']!='Bob' for s in plan['subjects'])

@pytest.mark.parametrize('fault',['historical','recapture','invented_evidence','fake_anchor','observed'])
def test_entrant_cannot_bypass_history_or_identity(fault):
    shot,plan=entrant_case();intent={}
    if fault=='historical':shot.raw['_scene_context']['entities']['Bob']['source_shot_id']='previous'
    if fault=='recapture':intent['render_mode']='recapture'
    if fault=='invented_evidence':plan['planned_entrants'][0]['screenplay_evidence']='made up arrival'
    if fault=='fake_anchor':plan['planned_entrants'][0]['anchor_label']='missing'
    if fault=='observed':plan['planned_entrants'][0]['status']='identity_verified'
    with pytest.raises(ValueError):validate_for_shot(plan,shot,intent)

def test_camera_retry_only_for_geometric_planning_failures():
    assert camera_failure_recoverable('no reliable seed depth for doorway')
    assert not camera_failure_recoverable('HTTP 401 unauthorized')
    assert not camera_failure_recoverable('world registration rejected')
    assert '<url>' in safe_error('failed https://private.example/path')
