import json
import pytest
from atas_geometry.service.durable import atomic_json
from atas_geometry.register.registration_recovery import run_bounded, diagnose
def result(ref, good):
    return {'reference_l0':str(ref),'reconstruction':{'geometry_accepted':good},
            'cross_shot_validation':{'accepted':True}}


def test_failed_reference_tries_one_distinct_alternative_and_resumes(tmp_path):
    calls=[]
    refs=[tmp_path/'old',tmp_path/'recent',tmp_path/'third']
    def execute(ref,d):
        calls.append(ref);atomic_json(d/'fusion_report.json',result(ref,ref==refs[1]))
    a,l=run_bounded(tmp_path,stamp={'parent':'v1'},references=refs,execute=execute)
    assert l['status']=='accepted' and calls==refs[:2]
    run_bounded(tmp_path,stamp={'parent':'v1'},references=refs,execute=execute)
    assert calls==refs[:2]


def test_same_reference_never_retried_to_chase_acceptance(tmp_path):
    ref=tmp_path/'old';calls=[]
    def execute(r,d):calls.append(r);atomic_json(d/'fusion_report.json',result(r,False))
    _,l=run_bounded(tmp_path,stamp={},references=[ref,ref],execute=execute)
    assert len(calls)==1 and l['status']=='exhausted'
    run_bounded(tmp_path,stamp={},references=[ref,ref],execute=execute)
    assert len(calls)==1


def test_completed_original_attempt_is_reused(tmp_path):
    ref=tmp_path/'old'
    atomic_json(tmp_path/'attempt_000/fusion_report.json',result(ref,False))
    _,l=run_bounded(tmp_path,stamp={},references=[ref],execute=lambda *a:pytest.fail('duplicate compute'))
    assert l['status']=='exhausted'


def test_uncertain_attempt_is_not_implicitly_restarted(tmp_path):
    calls=[]
    def crash(*a):calls.append(1);raise TimeoutError('worker interrupted')
    with pytest.raises(TimeoutError):run_bounded(tmp_path,stamp={},references=[tmp_path/'old'],execute=crash)
    with pytest.raises(RuntimeError,match='incomplete'):
        run_bounded(tmp_path,stamp={},references=[tmp_path/'old'],execute=crash)
    assert calls==[1]


def test_changed_inputs_and_wrong_reference_rejected(tmp_path):
    ref=tmp_path/'old'
    def execute(r,d):atomic_json(d/'fusion_report.json',result(r,True))
    run_bounded(tmp_path,stamp={'parent':'one'},references=[ref],execute=execute)
    with pytest.raises(ValueError,match='changed'):
        run_bounded(tmp_path,stamp={'parent':'two'},references=[ref],execute=execute)


def test_history_rejection_does_not_claim_new_video_is_wrong():
    d=diagnose({'reconstruction':{'geometry_accepted':False,'anchor_tolerance':.059,
        'chunks':[{'chunk':0,'accepted_geometry':False,'heldout_median':.078,
                   'heldout_within_2t':.7,'seam_median':.005,'pinhole_residual_pixels':.18}]},
        'cross_shot_validation':{'accepted':True}})
    assert d['failed_chunks']==[{'chunk':0,'checks':['heldout_history_median'],'heldout_x_limit':1.32,'seam_x_limit':0.04}]
    assert d['cross_shot_accepted'] and not d['acceptance_rules_changed']


def test_cross_shot_gate_still_required_and_no_third_attempt(tmp_path):
    calls=[]
    def execute(r,d):
        calls.append(r); data=result(r,True);data['cross_shot_validation']['accepted']=False
        atomic_json(d/'fusion_report.json',data)
    _,ledger=run_bounded(tmp_path,stamp={},references=[tmp_path/str(i) for i in range(3)],execute=execute)
    assert len(calls)==2 and ledger['status']=='exhausted'


def test_wrong_report_reference_rejected(tmp_path):
    def execute(r,d):atomic_json(d/'fusion_report.json',result(tmp_path/'wrong',True))
    with pytest.raises(ValueError,match='reference mismatch'):
        run_bounded(tmp_path,stamp={},references=[tmp_path/'old'],execute=execute)


def test_changed_completed_report_rejected(tmp_path):
    ref=tmp_path/'old'
    def execute(r,d):atomic_json(d/'fusion_report.json',result(r,False))
    run_bounded(tmp_path,stamp={},references=[ref],execute=execute)
    atomic_json(tmp_path/'attempt_000/fusion_report.json',result(ref,True))
    with pytest.raises(ValueError,match='report changed'):
        run_bounded(tmp_path,stamp={},references=[ref],execute=execute)


def test_legacy_partial_directory_not_reexecuted(tmp_path):
    (tmp_path/'attempt_000').mkdir()
    with pytest.raises(RuntimeError,match='incomplete'):
        run_bounded(tmp_path,stamp={},references=[tmp_path/'old'],execute=lambda *a:pytest.fail('unsafe rerun'))


def test_production_rejection_preserves_world_and_reuses_report(tmp_path,monkeypatch):
    import numpy as np
    import atas_geometry.world.writeback as module
    from atas_geometry.service.durable import file_hash
    scene=tmp_path/'scene'; ref=scene/'trusted';ref.mkdir(parents=True)
    source=tmp_path/'raw';source.mkdir()
    l1=scene/'l1.npz';np.savez(l1,xyz=np.ones((3,3)))
    state=scene/'world_state.json'
    atomic_json(state,{'version':'v1','l1':'l1.npz','observations':{'old':{'l0':'trusted'}}})
    folder=scene/'.world_versions/recon_test'
    atomic_json(folder/'intent.json',{'source_sha256':'video','parent_version':'v1','reference':str(ref)})
    atomic_json(folder/'attempt_000/fusion_report.json',{
        **result(ref,False),'source_l0':str(source),'original_l1_sha256':file_hash(l1)})
    before=(file_hash(l1),file_hash(state))
    monkeypatch.setattr(module,'rank_references',lambda *a:[ref])
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:pytest.fail('unnecessary GPU work'))
    for _ in range(2):
        response=module.writeback_world(source,scene,payload={'shot_id':'new','video_sha256':'video'},
            job_id='recon_test',settings=None,gpu_index=1,home_l0=ref)
        assert not response['registered'] and response['registration_recovery']['status']=='exhausted'
    assert before==(file_hash(l1),file_hash(state))
    assert not (folder/'candidate.json').exists()


def test_uncommitted_home_is_not_a_trusted_reference(tmp_path):
    from atas_geometry.world.writeback import trusted_references
    ref=tmp_path/'good';ref.mkdir()
    atomic_json(tmp_path/'world_state.json',{'observations':{'old':{'l0':'good'}}})
    assert trusted_references(tmp_path,tmp_path/'uncommitted')==[ref]


@pytest.mark.parametrize("code",["cuda_out_of_memory","cuda_linalg_initialization_failed"])
def test_resource_failures_preserve_type_and_do_not_exhaust_registration(tmp_path,code):
    from atas_geometry.service.resource_errors import ResourceUnavailableError
    ref=tmp_path/'old'; calls=[]
    state=tmp_path/'world_state.json';state.write_text('{"version":"old"}')
    before=state.read_bytes()
    def unavailable(r,folder):
        calls.append(r);folder.mkdir();(folder/'partial').write_text('keep this unpublished attempt')
        raise ResourceUnavailableError(code,'resource unavailable',physical_gpu_index=len(calls))
    for _ in range(3):
        with pytest.raises(ResourceUnavailableError) as error:
            run_bounded(tmp_path,stamp={},references=[ref],execute=unavailable,recover_interrupted=True)
        assert error.value.error_code==code and error.value.fields()['retryable']
        ledger=json.loads((tmp_path/'registration_recovery.json').read_text())
        assert ledger['status']=='waiting_resource'
        assert ledger['attempts'][0].get('interrupted_retries',0)==0
        assert len(ledger['attempts'])==1
        assert state.read_bytes()==before
    def complete(r,folder):atomic_json(folder/'fusion_report.json',result(r,True))
    _,ledger=run_bounded(tmp_path,stamp={},references=[ref],execute=complete,recover_interrupted=True)
    assert ledger['status']=='accepted'
    assert ledger['attempts'][0]['resource_interruptions']==3
    assert len(list(tmp_path.glob('resource_interrupted_*/partial')))==3
    run_bounded(tmp_path,stamp={},references=[ref],execute=lambda *a:pytest.fail('completed input repeated'))
    assert state.read_bytes()==before


def test_resource_resume_still_requires_owned_worker_reconciliation(tmp_path):
    from atas_geometry.service.resource_errors import ResourceUnavailableError
    def fail(*args):raise ResourceUnavailableError('cuda_out_of_memory','out of memory')
    with pytest.raises(ResourceUnavailableError):
        run_bounded(tmp_path,stamp={},references=[tmp_path/'old'],execute=fail)
    with pytest.raises(RuntimeError,match='reconcile'):
        run_bounded(tmp_path,stamp={},references=[tmp_path/'old'],execute=lambda *a:pytest.fail('unsafe rerun'))
