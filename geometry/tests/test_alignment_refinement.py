import numpy as np
import pytest
from atas_geometry.register.matching import AnchorSet
from atas_geometry.register.alignment_refinement import propose_correction
from atas_geometry.memory.fusion_projection import Projection
from atas_geometry.register.anchored_fusion import lift_depth, historical_visibility
def anchors(offset=.06, frames=6):
    x = np.random.default_rng(7).normal(size=(frames*40,3))
    return AnchorSet(x, x+[offset,0,0], np.repeat(np.arange(frames),40),
                     tuple(dict(pair_id=i,src_index=i) for i in range(frames)))


def test_small_offset_corrected_with_disjoint_source_frames():
    tr,r = propose_correction(anchors(), .05)
    assert tr is not None and r['accepted']
    assert not set(r['training_frames']) & set(r['validation_frames'])
    assert r['validation_after_median'] < 1e-6


def test_large_wrong_alignment_not_forced():
    tr,r = propose_correction(anchors(2), .05)
    assert tr is None and not r['accepted']


def test_two_frames_cannot_validate_independent_correction():
    tr,r = propose_correction(anchors(frames=2), .05)
    assert tr is None


def test_validation_frame_disagreement_rejected():
    a = anchors()
    a.target[a.pair_ids%2 == 1] += [1,0,0]
    tr,r = propose_correction(a, .05)
    assert tr is None


def test_sparse_pairs_are_pooled_without_fitting_or_residual_selection(monkeypatch):
    import cv2
    from types import SimpleNamespace
    from atas_geometry.register import matching as f
    kp = [SimpleNamespace(pt=(float(i+2),4.)) for i in range(14)]
    detector = SimpleNamespace(detectAndCompute=lambda *a: (kp,np.zeros((14,128),np.float32)))
    def knn(*args, **kwargs):
        return [[SimpleNamespace(queryIdx=i,trainIdx=i,distance=1.),
                 SimpleNamespace(queryIdx=i,trainIdx=(i+1)%14,distance=10.)] for i in range(6)]
    monkeypatch.setattr(cv2,'SIFT_create',lambda **kw:detector)
    monkeypatch.setattr(cv2,'BFMatcher',lambda *a:SimpleNamespace(knnMatch=knn))
    monkeypatch.setattr(f,'_align_indices',lambda count:[0,1,2])
    monkeypatch.setattr(f,'_grid_balance_matches',lambda matches,*a:matches)
    xyz = np.arange(18).reshape(6,3).astype(float)
    monkeypatch.setattr(f,'_lift_uv_robust',lambda *a:(xyz,np.ones(6,bool),{}))
    monkeypatch.setattr(f,'_fit_ransac',lambda *a,**kw:pytest.fail('must not fit individual sparse pairs'))
    store = SimpleNamespace(count=3, frames=[np.zeros((20,20,3),np.uint8)]*3,
                            dynamic_mask_at=lambda i,shape:np.zeros(shape,bool))
    pooled = f.collect_sift_anchors(store,store,profile='pooled_static_v1')
    assert len(pooled.source) == 54 and len(pooled.pair_reports) == 9
    strict = f.collect_sift_anchors(store,store,profile='world_scaled_static_v1',world_tolerance=.1)
    assert len(strict.source) == 0


def test_correction_preserves_dense_time_and_transforms_depth_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from atas_geometry.register.alignment_refinement import materialize_correction
    from atas_geometry.register.anchored_fusion import apply_sim3
    root = tmp_path/'original'; root.mkdir()
    (root/'video.mp4').write_bytes(b'immutable source')
    (root/'dynamic_mask').mkdir()
    (root/'dynamic_mask'/'00000.png').write_bytes(b'mask')
    poses = np.repeat(np.eye(4)[None],3,axis=0)
    poses[:,:3,3] = np.arange(9).reshape(3,3)
    k = np.repeat([[5,5,2,2]],3,axis=0)
    store = SimpleNamespace(recon_dir=root, count=3, cam_c2w=poses,
                            intrinsics=k, depth_at=lambda i: np.ones((4,4))*(i+1))
    written = {}
    monkeypatch.setattr('atas_geometry.register.anchored_fusion.write_exr',
                        lambda path,d: written.__setitem__(path.name,d.copy()))
    tr = (1.05,np.eye(3),np.array([.01,.02,.03]))
    report = {'source_timestamps_seconds':[0,.04,.08], 'frame_count':3,
              'chunks':[{'rotation':np.eye(3),'scale':np.float32(1.)}]}   # as reconstruct_conditioned writes it
    out = tmp_path/'refined'
    result = materialize_correction(store,out,tr,report,{'accepted':True})
    assert result['source_timestamps_seconds'] == report['source_timestamps_seconds']
    assert len(written) == 3
    for i in range(3): np.testing.assert_allclose(written[f'{i:05d}.exr'],1.05*(i+1))
    camera = np.load(out/'cameras.npz')
    np.testing.assert_allclose(camera['cam_c2w'][:,:3,3],apply_sim3(poses[:,:3,3],tr))
    assert (out/'dynamic_mask').resolve() == root/'dynamic_mask'
    assert (root/'video.mp4').read_bytes() == b'immutable source'


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_gpu_projection_matches_cpu_geometry(device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(1)
    depth = rng.uniform(1,3,(25,40)).astype(np.float32)
    pose = np.eye(4); pose[:3,3] = [.1,.2,.3]
    k = np.array([50,51,20,12.5])
    valid = depth > 1.5
    view = (depth,pose,k)
    p = Projection([view,view],device)
    xyz = p.lift(depth,pose,k,valid)
    np.testing.assert_allclose(xyz,lift_depth(depth,pose,k)[valid],atol=2e-7)
    points = np.concatenate([xyz,xyz+[0,0,-1],xyz+[0,0,1]])
    expected = historical_visibility(points,[view,view],.1)
    actual = p.visibility(points,.1,chunk_size=107)
    for x,y in zip(expected,actual): np.testing.assert_array_equal(x,y)
    assert len(p.visibility(np.zeros((0,3)),.1)[0]) == 0
