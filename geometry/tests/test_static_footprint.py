import os
from types import SimpleNamespace

import numpy as np
import pytest

from atas_geometry.render.point_render_policy import semantic_point_render_options, resolve_point_render_mode
def test_semantic_default_and_legacy_isolation():
    assert semantic_point_render_options({}, .1) == {
        "splat_mode":"pixel", "point_footprint":None, "max_point_radius":0}
    assert semantic_point_render_options({"point_render_mode":"bounded_square"}, .1) == {
        "splat_mode":"bounded_square", "point_footprint":.1, "max_point_radius":2}
    assert resolve_point_render_mode(None) == "pixel"
    assert semantic_point_render_options({"point_render_mode":"pixel"}, .1)["max_point_radius"] == 0


@pytest.mark.parametrize("radius", [-1, 3, 1.5, True, "2"])
def test_semantic_rejects_unbounded_radius(radius):
    with pytest.raises(ValueError):
        semantic_point_render_options({"max_point_radius":radius}, .1)


@pytest.mark.parametrize("voxel", [0, -1, float("nan"), float("inf")])
def test_invalid_footprint(voxel):
    with pytest.raises(ValueError):
        semantic_point_render_options({"point_render_mode":"bounded_square"}, voxel)


def render(points, colors, *, footprint=.6, radius=2, dynamic=None, mode="bounded_square"):
    import torch
    from atas_geometry.memory.observation import CameraCalibration
    from atas_geometry.render.render import render_l1_gpu
    camera = CameraCalibration("fixed", np.eye(4), np.array([[10.,0,16],[0,10.,16],[0,0,1]]), (32,32))
    memory = SimpleNamespace(xyz=np.asarray(points,dtype=np.float32).reshape(-1,3),
        rgb=np.asarray(colors,dtype=np.uint8).reshape(-1,3))
    before = memory.xyz.copy()
    images, stats = render_l1_gpu(memory,[camera],width=32,height=32,
        device=torch.device("cuda:"+os.environ["ATAS_TEST_GPU"]),splat_mode=mode,
        point_footprint=footprint,max_point_radius=radius,dynamic_frames=dynamic)
    np.testing.assert_array_equal(memory.xyz,before)
    np.testing.assert_array_equal(camera.world_to_camera,np.eye(4))
    return images[0],stats


gpu = pytest.mark.skipif("ATAS_TEST_GPU" not in os.environ,reason="explicit idle GPU required")


@gpu
def test_static_projection_scales_and_hard_cap_preserves_large_holes():
    near,s=render([[0,0,2]],[[200,0,0]])
    far,_=render([[0,0,8]],[[200,0,0]])
    assert np.count_nonzero(near[:,:,0]==200)==9
    assert np.count_nonzero(far[:,:,0]==200)==1
    huge,_=render([[0,0,.1]],[[200,0,0]],footprint=10)
    assert np.count_nonzero(huge[:,:,0]==200)==25
    np.testing.assert_array_equal(huge[0,0],[127,127,127])
    assert s["mean_coverage"] > s["mean_point_center_coverage"]


@gpu
@pytest.mark.parametrize("dynamic_z,center", [(2,[255,0,0]),(4,[0,0,255])])
def test_dynamic_is_one_pixel_and_shares_static_depth_buffer(dynamic_z,center):
    dynamic=iter([(np.array([[0,0,dynamic_z]],dtype=np.float32),np.array([[255,0,0]],dtype=np.uint8))])
    im,stats=render([[0,0,3]],[[0,0,255]],dynamic=dynamic)
    np.testing.assert_array_equal(im[16,16],center)
    np.testing.assert_array_equal(im[16,17],[0,0,255])
    assert np.count_nonzero(im[:,:,0]==255) == (1 if dynamic_z==2 else 0)
    assert stats["dynamic_splat_mode"] == "pixel"


@gpu
def test_zero_expansion_is_exact_pixel_reference():
    points=[[0,0,2],[.6,.2,3],[-.4,-.3,4]]
    colors=[[200,10,20],[10,200,20],[10,20,200]]
    old,_=render(points,colors,mode="pixel")
    zero,_=render(points,colors,radius=0)
    np.testing.assert_array_equal(old,zero)
