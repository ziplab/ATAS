import numpy as np
from atas_geometry.register.anchored_fusion import overlap_prior_frames, fit_intrinsics
def test_overlap_priors_leave_disjoint_validation_views():
    prior=overlap_prior_frames({i:{} for i in range(20,24)},20,44)
    assert prior==[20,22]
    assert set(range(20,24))-set(prior)=={21,23}
    assert overlap_prior_frames({23:{}},23,47)==[]
    assert overlap_prior_frames({23:{}},24,48)==[]


def test_intrinsics_matches_all_pixel_lstsq():
    rng=np.random.default_rng(71)
    y,x=np.mgrid[:80,:120]
    depth=rng.uniform(1,4,x.shape)
    local=np.stack(((x+.5-58)/100*depth,(y+.5-39)/105*depth,depth),-1).astype(np.float32)
    actual,residual=fit_intrinsics(local)
    expected=[]
    for axis,pixels in [(0,x+.5),(1,y+.5)]:
        ray=(local[...,axis]/local[...,2]).ravel()
        expected.append(np.linalg.lstsq(np.column_stack([ray,np.ones(ray.size)]),pixels.ravel(),rcond=None)[0])
    np.testing.assert_allclose(actual,[expected[0][0],expected[1][0],expected[0][1],expected[1][1]],rtol=1e-8)
    assert residual < 1e-4


def test_internal_failure_is_not_reported_as_measured_cross_shot_failure():
    from atas_geometry.register.registration_recovery import diagnose
    r=diagnose({'reconstruction':{'geometry_accepted':False,'chunks':[]},
                'cross_shot_validation':{'accepted':False,'status':'not_run'}})
    assert r['rejection_category']=='internal_reconstruction'
    assert r['cross_shot_status']=='not_run'


def test_bad_internal_geometry_skips_world_load_and_sift(tmp_path,monkeypatch):
    import json,sys
    from types import SimpleNamespace
    from atas_geometry.register import anchored_fusion as m, matching as fusion_v2
    from atas_geometry.memory.fusion import FusedSceneMemory
    ref,new,derived,out=[tmp_path/n for n in ['reference','new','derived','out']]
    for p in [ref,new,derived]:p.mkdir()
    (new/'video.mp4').write_bytes(b'immutable source')
    l1=tmp_path/'world.npz';l1.write_bytes(b'immutable world')
    (derived/'anchored_reconstruction.json').write_text(json.dumps(dict(
        source_l0=str(new),reference_l0=str(ref),geometry_accepted=False)))
    monkeypatch.setattr(m,'strict_store',lambda path,*args:SimpleNamespace(count=150,duration=5))
    def forbidden(*a,**k):raise AssertionError('must not load world or match a broken reconstruction')
    monkeypatch.setattr(FusedSceneMemory,'load',forbidden)
    monkeypatch.setattr(fusion_v2,'collect_sift_anchors',forbidden)
    monkeypatch.setattr(sys,'argv',['probe','--reference-l0',str(ref),'--new-l0',str(new),
        '--l1',str(l1),'--output',str(out),'--reuse-conditioned-l0',str(derived),'--vista-repo','vista','--model-id','pi3x'])
    m.main()
    report=json.loads((out/'fusion_report.json').read_text())
    assert report['cross_shot_validation']['status']=='not_run'
    assert report['original_l1_unchanged'] and report['new_points']==0
    assert (new/'video.mp4').read_bytes()==b'immutable source'
