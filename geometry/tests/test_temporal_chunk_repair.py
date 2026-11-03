import copy
import numpy as np
import pytest
from atas_geometry.reconstruct.temporal_chunk_repair import trusted_frame_indices, support_split, target_frames, repair_checks_pass
def test_only_accepted_owners_can_anchor_repair():
    owner=np.array([0]*4+[1]*4+[2]*4+[-1])
    reports=[dict(chunk=i,accepted_geometry=i!=1) for i in range(3)]
    assert trusted_frame_indices(owner,reports)==[0,1,2,3,8,9,10,11]
    targets=target_frames(dict(chunk=1,start=2,end=10),owner)
    assert targets==[4,5,6,7]
    p,t=support_split(targets,trusted_frame_indices(owner,reports))
    assert p==[0,2,8,10] and t==[1,3,9,11] and not set(p)&set(t)

def test_second_attempt_changes_evidence_but_keeps_immediate_seam_tests():
    targets=list(range(12,20));trusted=list(range(12))+list(range(20,32))
    p0,t0=support_split(targets,trusted,0);p1,t1=support_split(targets,trusted,1)
    assert p0!=p1 and set(t0)<=set(t1) and not set(p1)&set(t1)

def test_no_good_frames_cannot_repair_or_certify_itself():
    owner=np.zeros(10,int);r=[dict(chunk=0,accepted_geometry=False)]
    assert support_split(list(range(10)),trusted_frame_indices(owner,r))==([],[])
    assert support_split([2,3],[1,4])==([],[])

def good():
    return dict(heldout_count=100,heldout_median=.05,heldout_within_2t=.8,
        seam_checks=[dict(frame=2,count=101,median=.09)],pinhole_residual_pixels=1.)

def test_original_thresholds_are_not_relaxed():
    assert repair_checks_pass(good(),.05,[2])
    for k,v in [('heldout_count',99),('heldout_median',.051),('heldout_within_2t',.64),('pinhole_residual_pixels',2.01)]:
        row=good();row[k]=v;assert not repair_checks_pass(row,.05,[2])
    for k,v in [('count',100),('median',None),('median',.101)]:
        row=good();row['seam_checks'][0][k]=v;assert not repair_checks_pass(row,.05,[2])
    assert not repair_checks_pass(good(),.05,[2,4])

def test_unknown_frame_ownership_is_repaired_without_overwriting_good_overlap():
    owner=np.array([0]*4+[-1]*4+[2]*4)
    assert target_frames(dict(chunk=1,start=2,end=10),owner)==[4,5,6,7]

def test_seam_failure_cannot_be_hidden_by_other_good_seams():
    row=good();row['seam_checks']+=[dict(frame=4,count=1000,median=.11)]
    assert not repair_checks_pass(row,.05,[2,4])

def test_balanced_fit_recovers_transform_and_ignores_nonfinite_pairs():
    from atas_geometry.reconstruct.temporal_chunk_repair import balanced_anchor_fit
    from atas_geometry.register.anchored_fusion import apply_sim3
    rng=np.random.default_rng(123)
    sources=[rng.normal(size=(1000,3)),rng.normal(size=(200,3))]
    targets=[2*s+np.array([1.,2.,3.]) for s in sources]
    sources[0][0]=np.nan
    tr=balanced_anchor_fit(sources,targets,1)
    np.testing.assert_allclose(apply_sim3(sources[1],tr),targets[1],atol=1e-6)

def test_balanced_fit_requires_both_anchor_families():
    from atas_geometry.reconstruct.temporal_chunk_repair import balanced_anchor_fit
    from atas_geometry.register.anchored_fusion import AnchorFitError
    x=np.random.default_rng(1).normal(size=(100,3))
    with pytest.raises(AnchorFitError,match='missing'):
        balanced_anchor_fit([x,np.empty((0,3))],[x,np.empty((0,3))],1)

def test_retry_adds_rgb_context_without_removing_required_or_bad_frames():
    from atas_geometry.reconstruct.temporal_chunk_repair import context_frames
    targets=list(range(144,150));p=[136,138];t=[137,139,141,143]
    ids=context_frames(targets,p,t,list(range(144)),1)
    assert len(ids)==24 and set(targets+p+t)<=set(ids)
    assert ids==list(range(126,150))
    assert context_frames(targets,p,t,list(range(144)),0)==sorted(set(targets+p+t))

def test_registration_revision_does_not_recompute_raw_rgbd_or_masks():
    from atas_geometry.service.durable import fingerprint
    from atas_geometry.reconstruct.recon_revision import contract_digest
    old=dict(video_sha256='a'*64,reconstruction_contract={'schema':'dense_rgbd_and_seg_v2'})
    new={**old,'registration_contract':'trusted-neighbour-context-repair-v1'}
    assert fingerprint('recon',old)!=fingerprint('recon',new)
    assert contract_digest(old)==contract_digest(new)
