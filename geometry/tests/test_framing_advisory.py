import numpy as np
import pytest

from atas_geometry.render import cinematic_camera as cameras
from atas_geometry.render.camera_runtime import effective_framing_policy
from test_focal_camera_v2 import fixture


def mismatched_continuous():
    cam,actor,_,memory,plan=fixture()
    plan["subjects"]=plan["subjects"][:1]
    plan["motion"]=dict(name="orbit",degrees=25,travel_ratio=0.)
    plan["framing"]["vertical_fraction"]=.85
    return cam,actor,memory,plan


def test_default_strict_still_rejects_but_advisory_keeps_actual_failed_verdict():
    cam,actor,memory,plan=mismatched_continuous()
    args=(memory,cam,actor,np.array([0.,0.,3.]),actor,plan,tuple(np.linspace(0,2,9)),"continuous")
    with pytest.raises(ValueError,match="valid focal framing"):
        cameras.plan_query(*args,subject_groups=[actor])
    query,report=cameras.plan_query(*args,subject_groups=[actor],framing_policy="advisory")
    selected=report["selected"]
    assert report["framing_policy"]=="advisory"
    assert not selected["framing_validity"]["passed"]
    assert report["framing_warnings"]==selected["framing_validity"]["failures"]
    assert selected["geometry_safety"]["passed"]
    assert report["move"]==plan["motion"]["name"]=="orbit"
    assert report["requested_motion"]["degrees"]==25
    np.testing.assert_array_equal(query.cameras[0].world_to_camera,cam.world_to_camera)


def test_advisory_never_releases_unsafe_camera(monkeypatch):
    cam,actor,memory,plan=mismatched_continuous()
    monkeypatch.setattr(cameras,"check_camera_query",lambda *a,**k:{"passed":False,"failures":["surface collision"]})
    with pytest.raises(ValueError,match="surface collision"):
        cameras.plan_query(memory,cam,actor,np.array([0.,0.,3.]),actor,plan,(0.,1.,2.),"continuous",
                           subject_groups=[actor],framing_policy="advisory")


def test_advisory_never_releases_nonfinite_geometry():
    cam,actor,memory,plan=mismatched_continuous()
    actor[0,0]=np.nan
    with pytest.raises(ValueError,match="finite nonempty grounded"):
        cameras.plan_query(memory,cam,actor,np.array([0.,0.,3.]),actor,plan,(0.,1.,2.),"continuous",
                           subject_groups=[actor],framing_policy="advisory")


def test_plan_query_recapture_forces_strict_even_if_advisory_requested(monkeypatch):
    cam,actor,memory,plan=mismatched_continuous()
    monkeypatch.setattr(cameras,"framing_check",lambda *a,**k:{"passed":False,"failures":["event actor clipped"]})
    with pytest.raises(ValueError,match="event actor clipped"):
        cameras.plan_query(memory,cam,actor,np.array([0.,0.,3.]),actor,plan,(0.,1.,2.),"recapture",
                           subject_groups=[actor],framing_policy="advisory")


@pytest.mark.parametrize("override",[
    {"render_mode":"recapture"},
    {"camera_relation":"recapture"},
    {"geometry_layers":"l1+l2"},
    {"geometry_layers":None},
])
def test_runtime_limits_advisory_to_new_time_l1(override):
    payload={"framing_policy":"advisory","geometry_layers":"l1","camera_relation":"continuous",**override}
    assert effective_framing_policy(payload)=="strict"


def test_runtime_policy_default_and_explicit_normal_l1():
    assert effective_framing_policy({"geometry_layers":"l1"})=="strict"
    assert effective_framing_policy({"geometry_layers":"l1","framing_policy":"advisory"})=="advisory"
    with pytest.raises(ValueError,match="framing_policy"):
        effective_framing_policy({"geometry_layers":"l1","framing_policy":"ignore_everything"})
