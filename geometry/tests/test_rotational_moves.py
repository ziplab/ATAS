import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from atas_geometry.render.cinematic_camera import rotational_cameras, plan_query, validate_plan
from test_cinematic_camera import scene, plan

@pytest.mark.parametrize('move,axis,sign',[
    ('pan_right',0,1),('pan_left',0,-1),('tilt_up',1,-1),('tilt_down',1,1)])
def test_pan_tilt_true_rotation_not_orbit(move,axis,sign):
    camera,_,_=scene()
    result=rotational_cameras(camera,np.linspace(0,5,121),move,12,continuous=True)
    np.testing.assert_array_equal(result[0].world_to_camera,camera.world_to_camera)
    for c in result:
        np.testing.assert_allclose(c.center_world,camera.center_world,atol=1e-10)
        np.testing.assert_array_equal(c.intrinsic,camera.intrinsic)
    assert sign*result[-1].camera_to_world[axis,2]>.1
    r=Rotation.from_matrix(result[-1].camera_to_world[:3,:3])
    assert np.degrees(r.magnitude())==pytest.approx(12)

def test_cut_pan_ends_at_fitted_view():
    camera,_,_=scene()
    result=rotational_cameras(camera,[0,1,2],'pan_right',10,continuous=False)
    assert not np.allclose(result[0].world_to_camera,camera.world_to_camera)
    np.testing.assert_allclose(result[-1].world_to_camera,camera.world_to_camera)

def test_handheld_is_bounded_deterministic_and_nonzero():
    camera,_,_=scene();times=np.linspace(0,5,121)
    a=rotational_cameras(camera,times,'handheld',2,continuous=True)
    b=rotational_cameras(camera,times,'handheld',2,continuous=True)
    magnitudes=[]
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x.world_to_camera,y.world_to_camera)
        np.testing.assert_allclose(x.center_world,camera.center_world,atol=1e-10)
        magnitudes.append(np.degrees(Rotation.from_matrix(x.camera_to_world[:3,:3]).magnitude()))
    assert .5<max(magnitudes)<2

@pytest.mark.parametrize('move',['pan_left','pan_right','tilt_up','tilt_down','handheld',
                               'track_left','track_right','track_forward','track_back'])
def test_new_moves_pass_real_candidate_framing_and_safety(move):
    c,pts,mem=scene();p=plan('locked',side=0)
    angular=move.startswith(('pan_','tilt_')) or move=='handheld'
    p['motion']={'name':move,'degrees':1 if angular else 0,'travel_ratio':0 if angular else .08}
    p['framing']['vertical_fraction']=.4
    query,report=plan_query(mem,c,pts,np.array([0.,0,3.]),pts,p,
        tuple(np.linspace(0,5,21)),'continuous')
    np.testing.assert_array_equal(query.cameras[0].world_to_camera,c.world_to_camera)
    assert report['move']==move
    assert report['selected']['geometry_safety']['passed']
    assert report['selected']['framing_validity']['passed']
    if angular:
        for camera in query.cameras:np.testing.assert_allclose(camera.center_world,c.center_world,atol=1e-10)
    else:assert np.linalg.norm(query.cameras[-1].center_world-c.center_world)>0
