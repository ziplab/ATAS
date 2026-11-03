import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from atas.shot.camera_planner import reference_frame, plan_camera, _parse
class Reader:
    indices=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def count_frames(self):return 10
    def get_data(self,index):
        self.indices.append(index)
        return np.full((20,40,3),index,dtype=np.uint8)


def context():
    return {"context_id":"stable","source_shot_id":"source","frame_index":5,"recon_frame_count":6,
        "original_frame_count":10,"source_sampling":"center_slice_then_center_crop","width":16,"height":16}


def test_reference_uses_center_slice_not_original_tail(monkeypatch):
    monkeypatch.setattr("atas.shot.camera_planner.imageio.get_reader",lambda _:Reader())
    frame,index=reference_frame(Path("source.mp4"),context())
    assert index==7 and frame.shape==(20,20,3)
    assert frame[0,0,0]==7
    with pytest.raises(ValueError,match="frame count"):
        reference_frame(Path("source.mp4"),{**context(),"original_frame_count":12})


def test_one_cached_visual_call_and_focal_subset(tmp_path,monkeypatch):
    source=tmp_path/"story_story/source"
    source.mkdir(parents=True)
    (source/"result.mp4").write_bytes(b"fixture")
    out=tmp_path/"target"
    out.mkdir()
    monkeypatch.setattr("atas.shot.camera_planner.imageio.get_reader",lambda _:Reader())
    shot=SimpleNamespace(story_id="story",shot_id="target",entity_ids=("Alice","Bob"),prompt="Close-up Alice while Bob listens",
        shot_perspective="close up",shot_size="close_up",camera_angle="eye-level",camera_move_name="locked",camera_side_degrees=35,
        camera_movement_degrees=0,character_reference_images=())
    response={"schema_version":1,"status":"ok","subjects":[{"label":"Alice","kind":"dynamic","part":"face",
        "visible":True,"confidence":.95,"bbox":[.2,.2,.6,.7],"anchor_uv":[.4,.4]}],
        "framing":{"side_degrees":35,"elevation_degrees":0,"screen_position":[.5,.5],"vertical_fraction":.65,"horizontal_fraction":.78},
        "motion":{"name":"locked","degrees":0,"travel_ratio":0}}
    geometry=SimpleNamespace(camera_context=lambda _:context())
    with patch("atas.providers.vlm.chat",return_value=json.dumps(response)) as model:
        one=plan_camera(geometry,{"camera_relation":"cut_recompose"},shot,tmp_path,out)
        two=plan_camera(geometry,{"camera_relation":"cut_recompose"},shot,tmp_path,out)
    assert model.call_count==1
    assert one==two and one["target_entity_ids"]==["Alice"]
    assert one["camera_contract"]=="semantic-camera-v1"
    assert one["camera_plan"]["provenance"]["source_original_frame_index"]==7


def test_unknown_subject_and_malformed_plan_fail():
    with pytest.raises(ValueError):_parse('{"schema_version":1,"status":"unresolved","reason":"occluded"}')
    with pytest.raises(ValueError):_parse('{"schema_version":1,"status":"ok"}')


def test_onstage_context_character_and_compiled_brief_are_valid_evidence():
    from types import SimpleNamespace
    from atas.shot.camera_planner import validate_for_shot
    raw = {"planning_provenance": {"kind": "atas-recursive-compiler-v1"}, "onstage_labels": ["chen", "lin"],
           "camera_brief": "slow push toward the counter", "source_requirements": [{"content": {"goal": "x"}}]}
    shot = SimpleNamespace(raw=raw, entity_ids=("chen",), prompt="Chen pulls a book.", provider_prompt="",
                           shot_perspective="", camera_move_name="push_in")
    plan = {"subjects": [{"label": "chen", "kind": "dynamic", "framing_role": "focal"},
                         {"label": "lin", "kind": "dynamic", "framing_role": "context"}],
            "motion": {"name": "push_in"},
            "source_motion": {"explicit": True, "evidence": "slow push toward the counter", "allowed_names": ["push_in"]}}
    assert validate_for_shot(plan, shot, {"require_source_motion_evidence": True}) is plan
    plan["subjects"][1]["label"] = "stranger"
    import pytest
    with pytest.raises(ValueError, match="not a known character"):
        validate_for_shot(plan, shot, {})
