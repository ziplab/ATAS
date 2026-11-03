import numpy as np
import pytest
from atas_geometry.register.anchored_fusion import (
    frame_windows, fit_sim3, robust_sim3, apply_sim3, lift_depth,
    fit_intrinsics, historical_visibility, append_static,
)


@pytest.mark.parametrize("count,size,overlap", [(1,24,4), (24,24,4), (25,24,4), (210,24,4), (120,12,3)])
def test_all_frames_covered(count, size, overlap):
    windows = list(frame_windows(count, size, overlap))
    assert set(i for start, end in windows for i in range(start, end)) == set(range(count))
    assert all(end-start <= size for start,end in windows)
    assert all(end-start == min(size,count) for start,end in windows)   # no short remainder
    assert all(b[0] > a[0] and b[0] <= a[1]-overlap for a,b in zip(windows,windows[1:]))


@pytest.mark.parametrize("args", [(0,24,4), (10,4,4), (10,4,-1)])
def test_bad_windows(args):
    with pytest.raises(ValueError):
        list(frame_windows(*args))


def test_sim3_recovers_world_scale_rotation():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(1000,3))
    r = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    truth = (1.7, r, np.array([1,3,-2]))
    y = apply_sim3(x, truth)
    np.testing.assert_allclose(apply_sim3(x, fit_sim3(x,y)), y, atol=1e-10)


def test_robust_fit_rejects_outliers():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(1000,3))
    y = x*1.4 + [3,2,1]
    y[:100] += rng.normal(size=(100,3))*4
    tr = robust_sim3(x,y)
    assert np.median(np.linalg.norm(apply_sim3(x,tr)-y,axis=1)) < 1e-6


def test_degenerate_geometry_rejected():
    x = np.zeros((20,3)); x[:,0] = np.arange(20)
    with pytest.raises(ValueError):
        fit_sim3(x,x)


def test_intrinsic_roundtrip_and_half_pixel_convention():
    depth = np.ones((28,42))*3
    k = np.array([50,51,21,14])
    points = lift_depth(depth, np.eye(4), k)
    fitted, error = fit_intrinsics(points)
    np.testing.assert_allclose(fitted,k,atol=1e-8)
    assert error < 1e-8


def test_depth_and_pose_scale_applied_once():
    d = np.ones((3,4))*2
    k = np.array([3,3,2,1.5])
    pose = np.eye(4); pose[:3,3] = [1,2,3]
    tr = (2.,np.eye(3),np.array([4,5,6]))
    aligned = pose.copy(); aligned[:3,3] = apply_sim3(pose[:3,3],tr)
    np.testing.assert_allclose(lift_depth(d*2,aligned,k),apply_sim3(lift_depth(d,pose,k),tr))


def test_historical_occlusion_is_not_free_space_conflict():
    view = (np.ones((10,10))*3,np.eye(4),np.array([5,5,5,5]))
    points = np.array([[0,0,3], [0,0,1], [0,0,5], [100,0,3]])
    surface, conflict = historical_visibility(points,[view,view],.1)
    assert surface.tolist() == [True,False,False,False]
    assert conflict.tolist() == [0,2,0,0]


def test_dynamic_or_unknown_history_depth_does_not_veto():
    view = (np.zeros((10,10)),np.eye(4),np.array([5,5,5,5]))
    surface, conflict = historical_visibility(np.array([[0,0,2]]),[view],.1)
    assert not surface.any() and not conflict.any()


def test_append_preserves_old_attributes_and_provenance():
    from atas_geometry.memory.fuse import _memory_from_voxels
    old = _memory_from_voxels(np.array([[0,0,1],[1,0,1]],np.float32), np.ones((2,3),np.uint8)*10,
        voxel_size=.01,origin=np.zeros(3),shot_id="old")
    new = _memory_from_voxels(np.array([[2,0,1]],np.float32), np.ones((1,3),np.uint8)*20,
        voxel_size=.01,origin=np.zeros(3),shot_id="new")
    result = append_static(old,new)
    np.testing.assert_array_equal(result.xyz[:2],old.xyz)
    np.testing.assert_array_equal(result.rgb[:2],old.rgb)
    assert len(result.provenance_offsets) == len(result.xyz)+1
    assert result.provenance_offsets[-1] == len(result.provenance_input_indices)
    assert result.voxel_size == old.voxel_size


@pytest.mark.parametrize("offset,accepted", [(.001,True), (.2,False)])
def test_cross_shot_check_does_not_refit_away_wrong_world_pose(monkeypatch, offset, accepted):
    from atas_geometry.register.anchored_fusion import validate_cross_shot
    from atas_geometry.register.matching import AnchorSet
    x = np.random.default_rng(1).normal(size=(50,3))
    anchors = AnchorSet(x, x + offset, np.repeat([0,1],25), ())
    monkeypatch.setattr("atas_geometry.register.matching.collect_sift_anchors", lambda *a,**kw: anchors)
    assert validate_cross_shot(None,None,.05)["accepted"] is accepted


def test_cross_shot_requires_multiple_frame_pairs(monkeypatch):
    from atas_geometry.register.anchored_fusion import validate_cross_shot
    from atas_geometry.register.matching import AnchorSet
    x = np.random.default_rng(1).normal(size=(50,3))
    anchors = AnchorSet(x,x,np.zeros(50,int),())
    monkeypatch.setattr("atas_geometry.register.matching.collect_sift_anchors", lambda *a,**kw: anchors)
    assert not validate_cross_shot(None,None,.05)["accepted"]
