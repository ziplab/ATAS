from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
from atas_geometry.reconstruct import initial_world as m
def _spread(medians, count=150):
    step = (count - 1) // len(medians)
    return [dict(frames=[i*step, (i+1)*step if i < len(medians)-1 else count-1], count=200, median=v)
            for i, v in enumerate(medians)]


def test_drift_needs_repeated_or_gross_supported_evidence():
    assert m.assess(_spread([.3, .3, .1]), .1, 150)['drift']
    assert not m.assess(_spread([.3, .1, .1]), .1, 150)['drift']           # one mild pair
    assert m.assess(_spread([.81, .1, .1]), .1, 150)['drift']              # one gross pair (> 4 x 2t)
    assert not m.assess([dict(frames=[0, 149], count=99, median=.3)]*2, .1, 150)['drift']
    assert not m.assess(_spread([.19, .19, .19]), .1, 150)['drift']


def test_missing_or_partial_evidence_is_not_consistency():
    assert not m.assess([], .1, 150)['sufficient']
    assert not m.assess([], None, 150)['sufficient']
    assert not m.assess([dict(frames=[0, 75], count=500, median=.01)]*3, .1, 150)['sufficient']
    assert m.assess(_spread([.05, .05, .05]), .1, 150)['sufficient']


def test_valid_normal_world_is_not_reconstructed(tmp_path,monkeypatch):
    from atas_geometry.register import anchored_fusion as a
    source=tmp_path/'source';source.mkdir()
    (source/'video.mp4').write_bytes(b'video');(source/'cameras.npz').write_bytes(b'cameras')
    monkeypatch.setattr(a,'strict_store',lambda p:SimpleNamespace(count=150))
    monkeypatch.setattr(m,'temporal_evidence',lambda s:(_spread([.05,.05,.05]),.1))
    monkeypatch.setattr(a,'reconstruct_conditioned',lambda *a,**kw:pytest.fail('normal world must not reconstruct'))
    result=m.prepare_initial_world(source,tmp_path/'repair',vista_repo=tmp_path,model_id='model')
    assert result['status']=='unchanged' and result['l0']==str(source)
    monkeypatch.setattr(m,'temporal_evidence',lambda s:pytest.fail('resume repeats diagnosis'))
    assert m.prepare_initial_world(source,tmp_path/'repair',vista_repo=tmp_path,model_id='model')==result
    (source/'cameras.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):
        m.prepare_initial_world(source,tmp_path/'repair',vista_repo=tmp_path,model_id='model')


def test_candidate_cannot_drop_evidence_to_pass(monkeypatch):
    from atas_geometry.register import matching as f
    pair=dict(frames=[0,149],uv=np.zeros((200,2)),vv=np.zeros((200,2)),count=200,median=.3)
    monkeypatch.setattr(f,'_lift_uv_robust',lambda *a:(np.zeros((200,3)),np.arange(200)<120,{}))
    checks,passed=m.verify_candidate(None,[pair],.1)
    assert not passed and not checks[0]['passed']


def test_candidate_checks_fixed_coordinates_without_refitting(monkeypatch):
    from atas_geometry.register import matching as f
    pair=dict(frames=[0,149],uv=np.zeros((200,2)),vv=np.zeros((200,2)),count=200,median=.3)
    monkeypatch.setattr(f,'_lift_uv_robust',lambda s,i,uv:(np.full((200,3),.2 if i else 0.),np.ones(200,bool),{}))
    assert not m.verify_candidate(None,[pair],.1)[1]


def test_candidate_agreement_passes(monkeypatch):
    from atas_geometry.register import matching as f
    pair=dict(frames=[0,149],uv=np.zeros((200,2)),vv=np.zeros((200,2)),count=200,median=.3)
    monkeypatch.setattr(f,'_lift_uv_robust',lambda *a:(np.zeros((200,3)),np.ones(200,bool),{}))
    assert m.verify_candidate(None,[pair],.1)[1]


@pytest.mark.parametrize('verified',[True,False])
def test_repair_is_derived_full_frame_and_never_published(tmp_path,monkeypatch,verified):
    from atas_geometry.register import anchored_fusion as a
    source=tmp_path/'raw';source.mkdir()
    for name in ('video.mp4','cameras.npz','instance_tracks.json'):
        (source/name).write_bytes(b'original')
    raw=SimpleNamespace(count=150,recon_dir=source)
    monkeypatch.setattr(a,'strict_store',lambda p:raw)
    pairs=[dict(frames=[0,149],count=200,median=.3)]*2
    monkeypatch.setattr(m,'temporal_evidence',lambda s:(pairs,.1))
    calls=[]
    def reconstruct(reference,new,output,**kw):
        assert reference.count==48 and new.count==150 and raw.count==150
        assert reference is not raw
        output.mkdir();calls.append(output)
        report=dict(geometry_accepted=True,frame_count=150)
        (output/'anchored_reconstruction.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(a,'reconstruct_conditioned',reconstruct)
    monkeypatch.setattr(m,'verify_candidate',lambda *args:([dict(passed=verified)],verified))
    result=m.prepare_initial_world(source,tmp_path/'repair',vista_repo=tmp_path,model_id='m')
    assert result['published'] is False and len(calls)==1
    assert (source/'cameras.npz').read_bytes()==b'original'
    assert result['status']==('repaired' if verified else 'rejected')
    assert result['l0']==(str(calls[0]) if verified else str(source))
    if verified:
        assert (calls[0]/'instance_tracks.json').read_bytes()==b'original'
    repeated=m.prepare_initial_world(source,tmp_path/'repair',vista_repo=tmp_path,model_id='m')
    assert repeated==result and len(calls)==1


def _pairs(medians):
    return [dict(frames=[0,i+1],uv=np.zeros((200,2)),vv=np.zeros((200,2)),count=200,median=m) for i,m in enumerate(medians)]


def test_repair_is_held_to_the_same_drift_criterion_as_unrepaired_sources(monkeypatch):
    # Observed in a 330-frame clip: repair cut every pair's error, two long-gap
    # pairs stayed slightly above 1x tolerance but well below 2x.
    from atas_geometry.register import matching as f
    after = iter([.034,.0643,.040,.061,.031,.084,.049])
    def lift(store, index, uv):
        if index == 0:
            lift.offset = next(after)
            return np.zeros((200,3)), np.ones(200,bool), {}
        return np.tile([lift.offset,0,0],(200,1)), np.ones(200,bool), {}
    monkeypatch.setattr(f,'_lift_uv_robust',lift)
    checks, passed = m.verify_candidate(None,_pairs([.143,.194,.085,.248,.133,.242,.070]),.064)
    assert passed and all(c['passed'] for c in checks)


def test_repair_still_rejects_unresolved_drift(monkeypatch):
    from atas_geometry.register import matching as f
    monkeypatch.setattr(f,'_lift_uv_robust',lambda s,i,uv:(np.full((200,3),.0 if i==0 else .1),np.ones(200,bool),{}))
    # Two pairs remain beyond 2x tolerance (|(.1,.1,.1)| = .173 > .128) although improved.
    assert not m.verify_candidate(None,_pairs([.3,.3]),.064)[1]


@pytest.mark.parametrize('accepted',[True,False])
def test_unverifiable_first_clip_is_checked_by_anchored_reconstruction(tmp_path,monkeypatch,accepted):
    from atas_geometry.register import anchored_fusion as a
    source=tmp_path/'raw';source.mkdir()
    for name in ('video.mp4','cameras.npz'):
        (source/name).write_bytes(b'original')
    monkeypatch.setattr(a,'strict_store',lambda p:SimpleNamespace(count=150,recon_dir=source))
    monkeypatch.setattr(m,'temporal_evidence',lambda s:([],.1))
    def reconstruct(reference,new,output,**kw):
        output.mkdir()
        report=dict(geometry_accepted=accepted,frame_count=150)
        (output/'anchored_reconstruction.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(a,'reconstruct_conditioned',reconstruct)
    result=m.prepare_initial_world(source,tmp_path/'repair',vista_repo=tmp_path,model_id='m')
    assert result['status']==('repaired' if accepted else 'rejected')
    assert result['reason']==('verified_by_anchored_reconstruction' if accepted else 'consistency_unverifiable')


def test_consistent_pair_may_change_within_tolerance(monkeypatch):
    # Observed: four drifted pairs fixed 15-25x, one consistent pair .068 -> .092 (t=.159).
    from atas_geometry.register import matching as f
    after = iter([.021,.03,.025,.02,.092])
    def lift(store, index, uv):
        if index == 0:
            lift.offset = next(after)
            return np.zeros((200,3)), np.ones(200,bool), {}
        return np.tile([lift.offset,0,0],(200,1)), np.ones(200,bool), {}
    monkeypatch.setattr(f,'_lift_uv_robust',lift)
    assert m.verify_candidate(None,_pairs([.53,.89,.35,.47,.068]),.159)[1]
