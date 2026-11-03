from types import SimpleNamespace
import numpy as np
import pytest
from atas_geometry.register.anchored_fusion import lift_depth
from atas_geometry.register.matching import _lift_uv, _lift_uv_robust
def store(depth):
    pose=np.eye(4);pose[:3,3]=[1.,2.,3.]
    k=np.array([80.,100.,4.5,5.5])
    return SimpleNamespace(depth_at=lambda i:depth,cam_c2w=pose[None],intrinsics=k[None])


@pytest.mark.parametrize('lifter',[_lift_uv,_lift_uv_robust])
def test_matches_and_dense_points_share_pixel_centres(lifter):
    depth=np.full((12,14),3.,np.float32);s=store(depth)
    uv=np.array([[3.,4.],[7.,8.]])
    points,valid,*_=lifter(s,0,uv)
    dense=lift_depth(depth,s.cam_c2w[0],s.intrinsics[0])
    assert valid.all()
    np.testing.assert_allclose(points,dense[uv[:,1].astype(int),uv[:,0].astype(int)])


def test_fractional_matches_preserve_bilinear_depth_and_centre_convention():
    y,x=np.mgrid[:12,:14];depth=3.+x*.002+y*.003;s=store(depth)
    uv=np.array([[3.25,4.75]])
    points,valid,_=_lift_uv_robust(s,0,uv)
    z=3.+3.25*.002+4.75*.003
    expected=np.array([[(3.25+.5-4.5)*z/80+1,(4.75+.5-5.5)*z/100+2,z+3]])
    assert valid.all();np.testing.assert_allclose(points,expected)
