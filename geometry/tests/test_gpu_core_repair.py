import copy
from types import SimpleNamespace
import numpy as np
import pytest
from atas_geometry.render import cinematic_camera as cc
from atas_geometry.register import matching as fv
from atas_geometry.register.anchored_fusion import validate_cross_shot
def plan(part='face'):
    return {'schema_version':1,'status':'ok','subjects':[{'label':'Alice','kind':'dynamic',
        'part':part,'visible':True,'confidence':.95,'bbox':[.2,.1,.8,.9],'anchor_uv':[.35,.35]}],
        'framing':{'screen_position':[.5,.5],'vertical_fraction':.65,'horizontal_fraction':.75,
                   'side_degrees':0,'elevation_degrees':0},
        'motion':{'name':'locked','degrees':0,'travel_ratio':0}}

def grounding_fixture():
    h,w=100,100;y,x=np.indices((h,w));depth=np.full((h,w),3.)
    xyz=np.column_stack((x.ravel(),y.ravel(),depth.ravel()))
    actor=(x>=25)&(x<45)&(y>=20)&(y<80)
    other=(x>=55)&(x<75)&(y>=20)&(y<80)
    obs=SimpleNamespace(xyz=xyz,foreground=(actor|other).ravel().astype(float))
    recon=SimpleNamespace(depth_at=lambda i:depth,unproject=lambda i:obs,exclude_sky=False)
    return recon,actor

def test_partial_region_cannot_claim_complete_unclipped_actor():
    recon,mask=grounding_fixture()
    for part in ('face','head','upper_body'):
        with pytest.raises(ValueError,match='complete visible instance'):
            cc.ground_subjects(recon,0,plan(part),subject_masks={'Alice':mask})

def test_entire_grounded_region_is_restricted_to_verified_instance():
    recon,mask=grounding_fixture()
    points,_,_,audit=cc.ground_subjects(recon,0,plan('full_body'),subject_masks={'Alice':mask})
    assert np.max(points[:,0])<45 and len(points)==int(mask.sum())
    assert audit[0]['identity_verified']
    assert audit[0]['part_semantics_verified'] is False

def test_face_box_changes_extent_and_does_not_use_neighbour_actor():
    recon,mask=grounding_fixture();p=plan()
    p['subjects'][0].update(bbox=[.25,.2,.45,.4],anchor_uv=[.35,.3])
    points,_,_,audit=cc.ground_subjects(recon,0,p,subject_masks={'Alice':mask})
    assert np.max(points[:,1])<40 and np.max(points[:,0])<45
    assert audit[0]['instance_fraction_in_region']<.5

def test_cropped_source_is_not_assumed_to_contain_a_complete_body():
    recon,mask=grounding_fixture();mask[:20,25:45]=True
    p=plan();cc.ground_subjects(recon,0,p,subject_masks={'Alice':mask})

def test_identity_mask_resolution_mismatch_fails():
    recon,mask=grounding_fixture()
    with pytest.raises(ValueError,match='shape mismatch'):
        cc.ground_subjects(recon,0,plan(),subject_masks={'Alice':mask[:50]})

def test_unknown_anatomical_part_is_not_silently_accepted():
    with pytest.raises(ValueError,match='unsupported part'):cc.validate_plan(plan('invented'))

def stores(scale=1,offset=0):
    import cv2
    rng=np.random.default_rng(4)
    rgb=rng.integers(0,256,(240,320,3),dtype=np.uint8)
    rgb=cv2.GaussianBlur(rgb,(3,3),0)
    k=np.array([[250,250,160,120]]*2,float)
    target=np.repeat(np.eye(4)[None],2,axis=0)
    source=target.copy();source[:,0,3]=(.06+offset)*scale
    def store(images,poses):
        return SimpleNamespace(count=2,frames=images,cam_c2w=poses,intrinsics=k,
            depth_at=lambda i:np.full((240,320),3*scale),
            dynamic_mask_at=lambda i,size:np.zeros(size,bool))
    return store([np.roll(rgb,-5,axis=1)]*2,source),store([rgb]*2,target)

def test_fixed_world_audit_never_applies_fitted_transform_to_hide_wrong_pose():
    new,ref=stores();good=validate_cross_shot(ref,new,.05)
    assert good['accepted'] and good['frame_pairs']>=2
    wrong,ref=stores(offset=1);bad=validate_cross_shot(ref,wrong,.05)
    assert not bad['accepted'] and bad['correspondences']==good['correspondences']
    assert bad['median_world_error']>.9

def test_fixed_world_decision_is_invariant_to_coordinate_units():
    new,ref=stores();a=validate_cross_shot(ref,new,.05)
    new,ref=stores(scale=10);b=validate_cross_shot(ref,new,.5)
    assert a['accepted'] and b['accepted']
    assert a['correspondences']==b['correspondences']
    np.testing.assert_allclose(a['median_world_error']*10,b['median_world_error'],atol=1e-8)

def test_wrong_image_correspondence_is_rejected_before_world_audit(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2,'findFundamentalMat',lambda *a,**kw:(None,None))
    new,ref=stores();r=fv.collect_sift_anchors(new,ref,profile='fixed_world_v1')
    assert len(r.source)==0
    assert all('insufficient_epipolar_matches' in p['reasons'] for p in r.diagnostics['pair_details'])

@pytest.mark.parametrize('tolerance',[None,0,-1,float('nan'),float('inf')])
def test_scaled_matching_requires_an_explicit_valid_scene_budget(tolerance):
    with pytest.raises(ValueError,match='world tolerance'):
        fv.collect_sift_anchors(None,None,profile='world_scaled_static_v1',world_tolerance=tolerance)
