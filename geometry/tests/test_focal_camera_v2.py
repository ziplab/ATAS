import copy
from types import SimpleNamespace

import numpy as np
import pytest

from atas_geometry.memory.observation import CameraCalibration
from atas_geometry.render.cinematic_camera import focal_geometry, framing_check, plan_query, project
def fixture():
    cam = CameraCalibration('source', np.eye(4), np.array([[90.,0,80],[0,90,45],[0,0,1]]), (160,90))
    actor = np.array([[x,y,3.] for x in np.linspace(-.3,.3,20) for y in np.linspace(-.6,.6,30)])
    tree = actor * [8, 3, 1] + [-4, 0, 3]
    wall = np.array([[x,y,9.] for x in np.linspace(-8,8,90) for y in np.linspace(-5,5,60)])
    subjects = [dict(label='actor', kind='dynamic', part='upper_body', visible=True, confidence=.99,
                     bbox=[.25,.2,.75,.8], anchor_uv=[.5,.5], framing_role='focal'),
                dict(label='tree', kind='static', part='object', visible=True, confidence=.99,
                     bbox=[.02,.1,.2,.8], anchor_uv=[.1,.5], framing_role='context')]
    plan = dict(schema_version=1, status='ok', subjects=subjects,
                framing=dict(shot_size='medium_close_up', screen_position=[.625,.475],
                             vertical_fraction=.65,horizontal_fraction=.78,side_degrees=0,elevation_degrees=0),
                motion=dict(name='push_in',degrees=0,travel_ratio=.25))
    return cam, actor, tree, SimpleNamespace(xyz=wall), plan


def test_background_does_not_determine_focal_size_or_aim():
    _,actor,tree,_,plan=fixture()
    p,focus=focal_geometry(np.concatenate([actor,tree]),plan,[actor,tree])
    np.testing.assert_array_equal(p,actor)
    np.testing.assert_allclose(focus,np.median(actor,axis=0))


def test_push_in_ends_at_requested_focal_size_without_clipping():
    cam,actor,tree,memory,plan=fixture()
    q,r=plan_query(memory,cam,np.concatenate([actor,tree]),np.array([-2,0,4.5]),actor,
                   plan,tuple(np.linspace(0,2,13)),'cut_recompose',subject_groups=[actor,tree])
    a,_=project(actor,q.cameras[0]);b,_=project(actor,q.cameras[-1])
    h0=np.diff(np.percentile(a[:,1],[2,98]))[0];h1=np.diff(np.percentile(b[:,1],[2,98]))[0]
    assert h1>h0*1.1 and abs(h1-.65)<.025
    assert r['selected']['framing_validity']['passed']
    assert max(s['clipped_fraction'] for s in r['selected']['framing_validity']['samples'])<=.05


def test_bad_middle_frame_cannot_pass_on_good_endpoint():
    cam,actor,_,_,plan=fixture()
    from dataclasses import replace
    bad=cam.world_to_camera.copy();bad[0,3]=30
    report=framing_check(actor,[cam,replace(cam,world_to_camera=bad),cam],plan['framing'])
    assert not report['passed']
    assert 'leaves frame' in report['failures'][0]


def test_incompatible_continuous_closeup_is_not_claimed_successful():
    cam,actor,_,memory,plan=fixture()
    plan['subjects']=plan['subjects'][:1]
    plan['motion']=dict(name='orbit',degrees=25,travel_ratio=0.)
    plan['framing']['vertical_fraction']=.85
    with pytest.raises(ValueError,match='valid focal framing'):
        plan_query(memory,cam,actor,np.array([0,0,3.]),actor,plan,
                   tuple(np.linspace(0,2,9)),'continuous',subject_groups=[actor])
