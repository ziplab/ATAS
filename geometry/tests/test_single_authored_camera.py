import numpy as np
import pytest
from types import SimpleNamespace
from atas_geometry.memory.observation import CameraCalibration
from atas_geometry.render import cinematic_camera as c
def setup():
    camera = CameraCalibration('source',np.eye(4),np.array([[90.,0,80],[0,90,45],[0,0,1]]),(160,90))
    points = np.array([[x,y,3.] for x in np.linspace(-.3,.3,20) for y in np.linspace(-.6,.6,30)])
    memory = SimpleNamespace(xyz=np.array([[x,y,7.] for x in np.linspace(-8,8,80) for y in np.linspace(-5,5,45)]))
    plan = {'schema_version':1,'status':'ok','subjects':[{'label':'cat','kind':'dynamic','part':'full_body',
        'visible':True,'confidence':.95,'bbox':[.25,.2,.75,.8],'anchor_uv':[.5,.5]}],
        'framing':{'shot_size':'medium','screen_position':[.5,.5],'vertical_fraction':.65,
        'horizontal_fraction':.78,'side_degrees':35,'elevation_degrees':0},
        'motion':{'name':'orbit','degrees':35,'travel_ratio':.25}}
    return memory,camera,points,plan


def test_one_path_preserves_extent_even_with_extreme_coverage_risk(monkeypatch):
    memory,camera,points,plan = setup()
    calls = []
    original = c.generate_camera_query
    def generate(*args,**kw):
        calls.append(kw)
        return original(*args,**kw)
    monkeypatch.setattr(c,'generate_camera_query',generate)
    monkeypatch.setattr(c,'coverage_risk',lambda *a,**k:{'risk_score':1e12})
    q,r = c.plan_query(memory,camera,points,np.array([0.,0,3.]),points,plan,tuple(np.linspace(0,2,9)),'cut_recompose')
    assert len(calls)==1 and calls[0]['orbit_degrees']==35
    assert r['candidate_count']==1 and r['selected']['motion_scale']==1
    assert r['selected']['side_degrees']==35 and r['ranking']=='none_single_authored_path'
    a,b=q.cameras[0].center_world-[0,0,3],q.cameras[-1].center_world-[0,0,3]
    assert np.degrees(np.arccos(a@b/np.linalg.norm(a)/np.linalg.norm(b))) == pytest.approx(35)


def test_unsafe_path_reports_problem_without_hidden_retry(monkeypatch):
    memory,camera,points,plan = setup()
    calls=[]
    def safety(*a,**k):
        calls.append(1)
        return {'passed':False,'failures':['wall collision']}
    monkeypatch.setattr(c,'check_camera_query',safety)
    with pytest.raises(ValueError,match='wall collision'):
        c.plan_query(memory,camera,points,np.array([0.,0,3.]),points,plan,tuple(np.linspace(0,2,9)),'cut_recompose')
    assert len(calls)==1
