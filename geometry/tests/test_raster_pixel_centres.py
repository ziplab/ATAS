"""Half-pixel RGB-D centres must land in their own raster bins."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from atas_geometry.memory.observation import CameraCalibration
from atas_geometry.render.render import render_l1_gpu
@pytest.mark.parametrize('streamed',[False,True])
@pytest.mark.parametrize('mode',['pixel','bounded_square'])
def test_dense_rgbd_centres_round_trip_without_checkerboard(monkeypatch,streamed,mode):
    if not torch.cuda.is_available():
        pytest.skip('GPU renderer requires CUDA')
    h,w=12,16
    y,x=np.indices((h,w));z=np.full((h,w),2.,dtype=np.float32)
    xyz=np.stack(((x+.5-w/2)/8*z,(y+.5-h/2)/8*z,z),axis=-1).astype(np.float32)
    rgb=np.stack((x*11,y*17,(x+y)*7),axis=-1).astype(np.uint8)
    memory=SimpleNamespace(xyz=xyz.reshape(-1,3),rgb=rgb.reshape(-1,3))
    camera=CameraCalibration('test',np.eye(4),np.array([[8,0,w/2],[0,8,h/2],[0,0,1.]]),(w,h))
    monkeypatch.setenv('GEOMETRY_RENDER_RESIDENT_POINTS','1' if streamed else '10000')
    monkeypatch.setenv('GEOMETRY_RENDER_POINT_BATCH','37')
    frames,stats=render_l1_gpu(memory,[camera],width=w,height=h,device=torch.device('cuda:0'),splat_mode=mode,point_footprint=.001,max_point_radius=2)
    np.testing.assert_array_equal(frames[0],rgb)
    assert stats['mean_coverage']==1.
