import pytest
from atas_geometry.register.partial_static import eligible_chunks, fusion_frames
def report():
    return dict(geometry_accepted=False, frame_count=6, frame_owner=[0,0,0,1,1,1],
        chunks=[dict(chunk=0,start=0,end=3,accepted_geometry=True),
                dict(chunk=1,start=3,end=6,accepted_geometry=False)])


def test_good_block_survives_bad_block():
    r=report()
    assert [f for _,f in eligible_chunks(r,6)] == [[0,1,2]]
    r['partial_static_validation']=dict(schema='partial_static_v1',accepted=True,
        frame_indices=[0,1,2],chunks=[dict(accepted=True,frames=[0,1,2])])
    assert fusion_frames(r,6)==[0,1,2]
    assert r['geometry_accepted'] is False


def test_unverified_frame_rejected():
    r=report()
    r['partial_static_validation']=dict(schema='partial_static_v1',accepted=True,
        frame_indices=[0,1,2,3],chunks=[dict(accepted=True,frames=[0,1,2,3])])
    with pytest.raises(ValueError): fusion_frames(r,6)


def test_no_evidence_no_fusion():
    assert fusion_frames(report(),6)==[]


def test_full_legacy_unchanged():
    assert fusion_frames(dict(geometry_accepted=True),6)==list(range(6))


def test_missing_ownership_rejected():
    r=report();r['frame_owner']=[]
    with pytest.raises(ValueError):eligible_chunks(r,6)


def test_cross_match_only_one_source_frame_does_not_pass(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from atas_geometry.register.matching import AnchorSet
    from atas_geometry.register.partial_static import validate_blocks
    points=np.zeros((40,3))
    match=AnchorSet(source=points,target=points.copy(),pair_ids=np.repeat([0,1],20),
        pair_reports=[dict(pair_id=0,src_index=0),dict(pair_id=1,src_index=0)],diagnostics={})
    monkeypatch.setattr('atas_geometry.register.matching.collect_sift_anchors',lambda *a,**k:match)
    r=report();r['anchor_tolerance']=.1
    assert validate_blocks(None,SimpleNamespace(count=6),r)['accepted'] is False


@pytest.mark.parametrize('bad_frame,expected', [(False,True),(True,False)])
def test_block_needs_independent_consistent_source_frames(monkeypatch,bad_frame,expected):
    import numpy as np
    from types import SimpleNamespace
    from atas_geometry.register.matching import AnchorSet
    from atas_geometry.register.partial_static import validate_blocks
    points=np.zeros((60,3));target=points.copy()
    if bad_frame:target[40:]=1
    match=AnchorSet(source=points,target=target,pair_ids=np.repeat([0,1,2],20),
        pair_reports=[dict(pair_id=i,src_index=i) for i in range(3)],diagnostics={})
    monkeypatch.setattr('atas_geometry.register.matching.collect_sift_anchors',lambda *a,**k:match)
    r=report();r['anchor_tolerance']=.1
    result=validate_blocks(None,SimpleNamespace(count=6),r)
    assert result['accepted'] is expected
    assert result['whole_clip_recapture_accepted'] is False
    assert result['frame_indices']==([0,1,2] if expected else [])


def test_correction_only_applies_to_accepted_blocks():
    from atas_geometry.register.partial_static import frame_corrections
    tr=dict(accepted=True,scale=1.,rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,.01])
    r=dict(partial_static_validation=dict(chunks=[
        dict(accepted=True,frames=[0,1],correction=tr),
        dict(accepted=False,frames=[2,3],correction=tr)]))
    assert set(frame_corrections(r))=={0,1}


def test_corrected_pose_and_depth_equal_corrected_points():
    import numpy as np
    from atas_geometry.register.anchored_fusion import apply_sim3
    angle=.03
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    scale=1.03;translation=np.array([.01,.02,-.01])
    pose=np.eye(4);pose[:3,3]=[.3,.4,.5]
    local=np.array([[.2,.3,1.],[.4,.1,2.]])
    world=local@pose[:3,:3].T+pose[:3,3]
    corrected_pose=pose.copy()
    corrected_pose[:3,3]=scale*rotation@pose[:3,3]+translation
    corrected_pose[:3,:3]=rotation@pose[:3,:3]
    np.testing.assert_allclose((local*scale)@corrected_pose[:3,:3].T+corrected_pose[:3,3],
        apply_sim3(world,(scale,rotation,translation)))


def test_registration_is_verified_on_unseen_frames():
    import numpy as np
    from atas_geometry.register.matching import AnchorSet
    from atas_geometry.register.partial_static import propose_block_registration
    rng=np.random.default_rng(4);source=rng.normal(size=(200,3));target=source+[.2,.1,.3]
    labels=np.repeat(np.arange(5),40)
    match=AnchorSet(source=source,target=target,pair_ids=labels,
        pair_reports=[dict(pair_id=i,src_index=i) for i in range(5)],diagnostics={})
    tr,evidence=propose_block_registration(match,.01)
    assert tr is not None
    assert not set(evidence['training_frames']) & set(evidence['validation_frames'])
    target[np.isin(labels,[1,3])]+=.5
    tr,evidence=propose_block_registration(match,.01)
    assert tr is None


def test_partial_temporal_support_uses_reliable_interval():
    from atas_geometry.register.partial_static import evidence_bins
    frames=list(range(24,44))
    assert set(evidence_bins(frames,150,partial=False).values())=={0}
    assert set(evidence_bins(frames,150,partial=True).values())=={0,1,2}
    assert evidence_bins([],150,partial=True)=={}
    assert len(set(evidence_bins([24],150,partial=True).values()))==1


def test_two_level_radius_is_exact():
    import numpy as np
    from scipy.spatial import cKDTree
    from atas_geometry.memory.bounded_neighbors import outside_radius
    rng=np.random.default_rng(17)
    world=rng.uniform(-1,1,(15000,3));queries=rng.uniform(-1.2,1.2,(3000,3))
    radius=.08;cell=radius/4
    _,indices=np.unique(np.floor(world/cell).astype(int),axis=0,return_index=True)
    full=cKDTree(world);coarse=cKDTree(world[indices])
    expected=outside_radius(full,queries,radius)
    actual=outside_radius(full,queries,radius,coarse_tree=coarse,coarse_error=np.sqrt(3)*cell*(1+1e-6))
    np.testing.assert_array_equal(actual,expected)
