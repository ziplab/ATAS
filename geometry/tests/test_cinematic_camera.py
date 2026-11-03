import copy
import json
import os
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from atas_geometry.memory.observation import CameraCalibration
from atas_geometry.render.cinematic_camera import validate_plan, ground_subjects, aimed_camera, fit_reference, project, plan_query, coverage_risk
from atas_geometry.render.camera_context import camera_context
def plan(move="locked", side=35):
    return {"schema_version":1,"status":"ok", "subjects":[{"label":"Alice", "kind":"dynamic", "part":"face", "visible":True,
        "confidence":.95, "bbox":[.25,.2,.75,.8], "anchor_uv":[.5,.5]}],
        "framing":{"shot_size":"close_up", "screen_position":[.5,.5], "vertical_fraction":.65, "horizontal_fraction":.78,
            "side_degrees":side, "elevation_degrees":0}, "motion":{"name":move,"degrees":35 if move in {"arc_left","arc_right","orbit"} else 0,
            "travel_ratio":.25 if move != "locked" else 0}}


def scene():
    camera = CameraCalibration("source", np.eye(4), np.array([[90.,0,80],[0,90,45],[0,0,1]]), (160,90))
    points = np.array([[x,y,3.] for x in np.linspace(-.3,.3,20) for y in np.linspace(-.6,.6,30)])
    wall = np.array([[x,y,7.] for x in np.linspace(-8,8,160) for y in np.linspace(-5,5,90)])
    return camera, points, SimpleNamespace(xyz=wall)


def test_plan_rejects_unresolved_and_nonfinite():
    for mutate in (lambda p:p.update(status="unresolved"), lambda p:p["subjects"][0].update(confidence=.3),
        lambda p:p["subjects"][0].update(anchor_uv=[.99,.99]), lambda p:p["motion"].update(degrees=float("nan"))):
        p = plan()
        mutate(p)
        with pytest.raises(ValueError): validate_plan(p)


def test_screen_placement_preserves_intrinsics():
    camera, points, memory = scene()
    focus = np.array([0.,0,3.])
    target = aimed_camera(camera, np.array([1.,0,0]), focus, [.35,.42], np.array([0,-1.,0]))
    uv, _ = project(focus[None], target)
    np.testing.assert_allclose(uv[0], [.35,.42], atol=1e-6)
    np.testing.assert_array_equal(camera.intrinsic, target.intrinsic)


def test_face_region_changes_distance_not_whole_body_zoom():
    camera, body, _ = scene()
    focus = np.array([0.,0,3.])
    framing = plan()["framing"]
    face = body.copy()
    face[:, :2] *= .25
    full = fit_reference(camera, body, focus, framing, 0)
    close = fit_reference(camera, face, focus, framing, 0)
    assert np.linalg.norm(close.center_world-focus) < .4*np.linalg.norm(full.center_world-focus)
    uv, _ = project(face, close)
    height = np.diff(np.percentile(uv[:,1],[2,98]))[0]
    assert abs(height - .65) < .02


@pytest.mark.parametrize("move", ["locked","push_in","orbit","truck_right","crane_up"])
def test_continuous_first_pose_exact(move):
    camera, points, memory = scene()
    intent = plan(move)
    # Isolate endpoint inheritance: orbit/truck/crane cannot turn a medium
    # source into a close-up without adding an unrequested radial move.
    if move in {"orbit", "truck_right", "crane_up"}:
        intent["framing"]["vertical_fraction"] = .4
    query, report = plan_query(memory,camera,points,np.array([0.,0,3.]),points,intent,tuple(np.linspace(0,2,9)),"continuous")
    np.testing.assert_array_equal(query.cameras[0].world_to_camera, camera.world_to_camera)
    assert report["frame0_inherited"] is True
    if move == "locked":
        for c in query.cameras: np.testing.assert_array_equal(c.world_to_camera,camera.world_to_camera)


def test_cut_reselects_camera_and_arc_not_capped_at_15():
    camera, points, memory = scene()
    focus = np.array([0.,0,3.])
    query, report = plan_query(memory,camera,points,focus,points,plan("orbit"),tuple(np.linspace(0,2,9)),"cut_recompose")
    assert not np.allclose(query.cameras[0].center_world,camera.center_world)
    a,b = query.cameras[0].center_world-focus,query.cameras[-1].center_world-focus
    angle = np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))
    assert angle > 15
    assert report["selected"]["motion_scale"] >= .7


def test_coverage_large_connected_hole_is_worse_than_scattered():
    camera, _, _ = scene()
    def unproject_grid(mask):
        y,x=np.nonzero(mask)
        u,v=(x+.5)/96*160,(y+.5)/54*90
        return np.column_stack(((u-80)/90*4,(v-45)/90*4,np.full(len(x),4)))
    scattered=np.indices((54,96)).sum(axis=0)%2==0
    large=np.ones((54,96),dtype=bool)
    large[:,:48]=False
    empty=np.empty((0,3))
    a=coverage_risk(unproject_grid(scattered),[camera],empty)
    b=coverage_risk(unproject_grid(large),[camera],empty)
    assert b["worst_largest_unknown_region"] > a["worst_largest_unknown_region"]
    assert a["policy"].startswith("soft_ranking")
    excluded=unproject_grid(~large)
    c=coverage_risk(unproject_grid(large),[camera],excluded)
    assert c["unknown_fraction"] < .01


def test_grounding_uses_named_region_without_asserting_global_identity():
    h,w=40,60
    y,x=np.indices((h,w))
    depth=np.full((h,w),3.)
    xyz=np.column_stack(((x.ravel()+.5-w/2)/40*3,(y.ravel()+.5-h/2)/40*3,depth.ravel()))
    observation=SimpleNamespace(xyz=xyz,foreground=np.ones(h*w))
    recon=SimpleNamespace(depth_at=lambda _:depth,unproject=lambda _:observation,exclude_sky=True,sky_files=[])
    points,focus,dynamic,audit=ground_subjects(recon,0,plan())
    assert len(points)>32 and len(dynamic)==h*w
    assert audit[0]["label"]=="Alice" and audit[0]["identity_verified"] is False
    observation.foreground[:]=0
    with pytest.raises(ValueError,match="seed depth"):
        ground_subjects(recon,0,plan())


def test_context_event_uses_actual_center_slice(tmp_path):
    source=tmp_path/"run/story/shots/source/l0"
    source.mkdir(parents=True)
    with imageio.get_writer(str(source/"video.mp4"),fps=6,macro_block_size=1) as writer:
        for i in range(6): writer.append_data(np.full((16,16,3),i*20,dtype=np.uint8))
    np.savez(source/"cameras.npz",cam_c2w=np.repeat(np.eye(4)[None],6,axis=0))
    scene_dir=tmp_path/"run/story/scenes/room"
    scene_dir.mkdir(parents=True)
    (scene_dir/"l1.npz").touch()
    meta={"run_id":"run","story_id":"story","scene_id":"room","shot_id":"source","event_id":"event",
        "duration":2.,"registered":True,"raw_to_scene":{"scale":1},"original_frame_count":12,"source_fps":6.}
    (source/"observation.json").write_text(json.dumps(meta))
    p={"run_id":"run","story_id":"story","scene_id":"room","shot_id":"new","source_shot_id":"source","source_event_id":"event",
        "event_id":"event","render_mode":"recapture","source_time_start":.5,"source_time_end":1.5,"duration":1.,"fps":6}
    context=camera_context(tmp_path,p)
    assert context["frame_index"]==0
    with pytest.raises(ValueError,match="center slice"):
        camera_context(tmp_path,{**p,"source_time_start":0.,"source_time_end":1.})


@pytest.mark.skipif("ATAS_TEST_GPU" not in os.environ,reason="explicit idle GPU required")
@pytest.mark.parametrize("event",[False,True])
def test_semantic_runtime_writes_real_mp4_and_camera(tmp_path,monkeypatch,event):
    from PIL import Image
    from atas_geometry.render import camera_runtime as runtime
    from atas_geometry.memory.camera_query import CameraQuery
    from atas_geometry.memory import vista4d_layers
    source=tmp_path/"run/story/shots/source/l0"
    source.mkdir(parents=True)
    w,h=64,36
    mask=np.zeros((h,w),dtype=bool)
    mask[8:28,22:42]=True
    depth=np.where(mask,3.,5.).astype(np.float32)
    color=np.full((h,w,3),180,dtype=np.uint8)
    color[mask]=[200,30,30]
    with imageio.get_writer(str(source/"video.mp4"),fps=2,macro_block_size=1) as writer:
        for _ in range(4):writer.append_data(color)
    np.savez(source/"cameras.npz",cam_c2w=np.repeat(np.eye(4)[None],4,axis=0),intrinsics=np.tile([40.,40.,32.,18.],(4,1)))
    for dirname in ("depths","dynamic_mask"):(source/dirname).mkdir()
    for i in range(4):
        (source/"depths"/f"{i:04d}.exr").touch()
        Image.fromarray(mask.astype(np.uint8)*255).save(source/"dynamic_mask"/f"{i:04d}.png")
    scene_dir=tmp_path/"run/story/scenes/room"
    scene_dir.mkdir(parents=True)
    (scene_dir/"home_shot_id.txt").write_text("source\n")
    (scene_dir/"l1.npz").touch()
    y,x=np.nonzero(~mask)
    xyz=np.column_stack(((x+.5-32)/40*5,(y+.5-18)/40*5,np.full(len(x),5)))
    memory=SimpleNamespace(xyz=xyz,rgb=np.full(xyz.shape,180,dtype=np.uint8),voxel_size=.125)
    monkeypatch.setattr(runtime.FusedSceneMemory,"load",lambda _:memory)
    monkeypatch.setattr(runtime,"_install_depth_reader",lambda:None)
    monkeypatch.setattr(vista4d_layers,"_read_exr_depth",lambda _:depth)
    p={"run_id":"run","story_id":"story","scene_id":"room","shot_id":"new","event_id":"event",
        "duration":2.,"fps":4,"camera_relation":"cut_recompose","geometry_layers":"l1"}
    if event:
        meta={"run_id":"run","story_id":"story","scene_id":"room","shot_id":"source","event_id":"event",
            "duration":2.,"registered":True,"original_frame_count":4,"source_fps":2.,
            "raw_to_scene":{"scale":1.,"rotation":np.eye(3).tolist(),"translation":[0.,0.,0.]}}
        (source/"observation.json").write_text(json.dumps(meta))
        p.update(render_mode="recapture",geometry_layers="l1+l2",camera_relation="recapture",source_shot_id="source",source_event_id="event",source_time_start=0.,source_time_end=2.)
    intent=plan("locked",side=25)
    intent["context"]=camera_context(tmp_path,p)
    p["camera_plan"]=intent
    out=tmp_path/"render/memory_input.mp4"
    report=runtime.render_semantic_camera(tmp_path,p,out,w,h,int(os.environ["ATAS_TEST_GPU"]))
    assert out.stat().st_size>0
    with imageio.get_reader(str(out)) as reader:assert reader.count_frames()==8
    query=CameraQuery.load(out.with_name("camera_query.json"))
    assert not np.allclose(query.cameras[0].center_world,[0,0,0])
    assert report["anchor_audit"][0]["identity_verified"] is False
    assert report["splat_mode"] == "pixel"
    assert report["max_point_radius"] == 0
    assert report["mean_coverage"] >= report["mean_point_center_coverage"] - 1e-6
    if event:
        assert report["dynamic_splat_mode"] == "source_pixel"
        assert report["max_dynamic_radius"] == 2
        assert len(report["source_frame_indices"])==8
        assert report["source_frame_indices"][-1]==3
        assert min(report["dynamic_points_per_frame"])>0
    else:assert report["dynamic_points_per_frame"]==[]
