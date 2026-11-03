import os
from types import SimpleNamespace
import numpy as np
import pytest
from atas_geometry.render.dynamic_footprint import source_pixel_support, dynamic_render_options, dynamic_frame
def test_source_scale_and_boundary_caps():
    depth=np.full((7,7),2.,np.float32);mask=np.zeros((7,7),bool);mask[1:6,1:6]=True
    support,caps=source_pixel_support(depth,mask,100,100)
    np.testing.assert_allclose(support,.02)
    assert caps.reshape(5,5)[2,2]==2 and caps.reshape(5,5)[0,2]==1
    doubled,_=source_pixel_support(depth*2,mask,100,100)
    np.testing.assert_allclose(doubled,support*2)
    depth[3,4]=4
    _,caps=source_pixel_support(depth,mask,100,100)
    assert caps.reshape(5,5)[2,2]==0


def test_dense_dynamic_order_preserved():
    depth=np.ones((3,4),np.float32);depth[0,0]=np.nan
    xyz=np.arange(33).reshape(11,3);rgb=np.zeros((11,3),np.uint8)
    foreground=np.arange(11)%2
    recon=SimpleNamespace(depth_at=lambda i:depth,exclude_sky=False,intrinsics=np.array([[10,20,2,2]]))
    obs=SimpleNamespace(xyz=xyz,rgb=rgb,foreground=foreground)
    data=dynamic_frame(recon,0,obs)
    np.testing.assert_array_equal(data[0],xyz[foreground==1])
    assert len(data[0])==len(data[2])==len(data[3])==5


@pytest.mark.parametrize("value",[-1,3,True,1.5,"2"])
def test_caps_are_bounded(value):
    with pytest.raises(ValueError):dynamic_render_options({"max_dynamic_radius":value})


gpu=pytest.mark.skipif("ATAS_TEST_GPU" not in os.environ,reason="explicit idle GPU required")


def render(dynamic,static_z=None,cap=2,mode="source_pixel"):
    import torch
    from atas_geometry.memory.observation import CameraCalibration
    from atas_geometry.render.render import render_l1_gpu
    memory=SimpleNamespace(xyz=np.array([[0,0,static_z]],np.float32) if static_z else np.empty((0,3),np.float32),
        rgb=np.array([[0,0,255]],np.uint8) if static_z else np.empty((0,3),np.uint8))
    camera=CameraCalibration("fixed",np.eye(4),np.array([[10,0,16],[0,10,16],[0,0,1.]]),(32,32))
    return render_l1_gpu(memory,[camera]*len(dynamic),width=32,height=32,
        device=torch.device("cuda:"+os.environ["ATAS_TEST_GPU"]),splat_mode="bounded_square",
        point_footprint=1.5,max_point_radius=2,dynamic_frames=dynamic,
        dynamic_splat_mode=mode,max_dynamic_radius=cap)


def frame(z=2,support=.6,cap=2,x=0):
    return np.array([[x,0,z]],np.float32),np.array([[255,0,0]],np.uint8),np.array([support],np.float32),np.array([cap])


@gpu
def test_near_dynamic_fills_grid_far_stays_small_and_cap_limits():
    images,stats=render([frame(),frame(z=8),frame(support=100)])
    assert [int((image[:,:,0]==255).sum()) for image in images]==[9,1,25]
    assert stats["dynamic_visible_pixels_per_frame"]==[9,1,25]
    assert stats["dynamic_splat_mode"]=="source_pixel"
    np.testing.assert_array_equal(images[2][0,0],[127,127,127])


@gpu
def test_dynamic_boundary_does_not_expand_past_one_pixel():
    images,_=render([frame(support=100,cap=1),frame(support=100,cap=0)])
    assert [(im[:,:,0]==255).sum() for im in images]==[9,1]


@gpu
def test_shared_depth_keeps_foreground_static_and_no_event_trails():
    images,stats=render([frame(z=4,support=1),frame(z=2),frame(z=2,x=2)],static_z=3)
    assert (images[0][:,:,0]==255).sum()==0
    np.testing.assert_array_equal(images[1][16,16],[255,0,0])
    np.testing.assert_array_equal(images[2][16,16],[0,0,255])
    assert stats["dynamic_visible_pixels_per_frame"][0]==0


@gpu
def test_zero_dynamic_radius_is_exact_legacy_pixels():
    baseline,_=render([frame()],mode="pixel")
    zero,_=render([frame()],cap=0)
    np.testing.assert_array_equal(baseline,zero)
