"""Two-level compilation: skeleton once, segments expanded against W."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atas.compiler import recursive as rc
from atas.request import plan_from_payload
from atas.world.ledger import WorldLedger
def skeleton():
    return {
        "source": {"input_path": "", "n_shots": 3, "per_shot_duration_s": [12, 6, 6], "seed": 0},
        "style": {"medium": "photographic live action", "visual_finish": "soft", "genre": "drama",
                  "contract": "photographic live action, soft natural light"},
        "portrait_plan": [{"id": "alice", "name": "Alice", "prompt": "a woman"}],
        "location_plan": {"hall": {"id": "hall", "reference_name": "hall"},
                          "yard": {"id": "yard", "reference_name": "yard"}},
        "boundarys": {"boundary_anchors": [{"id": f"a0{i}_x_start", "prompt": "frame"} for i in range(3)]},
        "shots": [
            {"id": "shot01_leave", "duration_s": 12, "story_goal": "g", "start_state": "s", "end_state": "e",
             "visual_intent": "v", "location_id": "hall", "characters": ["alice"], "focal": ["alice"]},
            {"id": "shot02_fountain", "duration_s": 6, "story_goal": "g", "start_state": "s", "end_state": "e",
             "visual_intent": "v", "location_id": "yard", "characters": ["alice"], "focal": ["alice"]},
            {"id": "shot03_replay", "duration_s": 5, "story_goal": "g", "start_state": "s", "end_state": "e",
             "visual_intent": "v", "location_id": "hall", "characters": ["alice"], "focal": ["alice"],
             "recapture_of": {"shot_id": "shot01_leave", "start_s": 6, "end_s": 11}},
        ],
        "transitions": [
            {"id": "tr01", "from_shot": "shot01_leave", "to_shot": "shot02_fountain", "mode": "bridge",
             "duration_s": 4, "first_frame": "from_shot_last_frame", "last_frame": "to_shot_start_anchor",
             "prompt": "camera passes through the door"},
            {"id": "tr02", "from_shot": "shot02_fountain", "to_shot": "shot03_replay", "mode": "cut"},
        ],
    }


def segment(sid, index, duration):
    return {"id": sid, "segment_index_in_shot": index, "duration_s": duration,
            "first_frame_path": "shot_start_anchor" if index == 0 else "previous_segment_last_frame",
            "prompt": "p", "end_state": "e", "camera_brief": "b", "camera_angle": "eye_level",
            "shot_size": "medium", "camera_move_name": "push_in", "focal": ["alice"], "characters": ["alice"]}


def test_skeleton_validation_rules():
    rc.validate_skeleton(skeleton())
    bad = copy.deepcopy(skeleton()); bad["shots"][2]["recapture_of"]["end_s"] = 13
    with pytest.raises(ValueError, match="recapture"):
        rc.validate_skeleton(bad)
    bad = copy.deepcopy(skeleton()); bad["shots"][1]["location_id"] = "nowhere"
    with pytest.raises(ValueError, match="location_id"):
        rc.validate_skeleton(bad)
    with pytest.raises(ValueError, match="exactly"):
        rc.validate_skeleton(skeleton(), ["bob"])


def test_segment_validation_rules():
    shot, anchor = skeleton()["shots"][0], {"id": "a00_x_start"}
    rows = rc.validate_segments({"s0": segment("s0", 0, 6), "s1": segment("s1", 1, 6)}, shot, anchor, {"alice"},
                                single=False)
    assert [r["id"] for r in rows] == ["s0", "s1"]
    with pytest.raises(ValueError, match="sum"):
        rc.validate_segments({"s0": segment("s0", 0, 6)}, shot, anchor, {"alice"}, single=False)
    with pytest.raises(ValueError, match="4, 15"):
        rc.validate_segments({"s0": segment("s0", 0, 16)}, {**shot, "duration_s": 16}, anchor, {"alice"},
                             single=False)


def test_lazy_expansion_sees_world_state_and_maps_requests(tmp_path, monkeypatch):
    screenplay = tmp_path / "story.txt"; screenplay.write_text("Alice leaves the hall.")
    portrait = tmp_path / "alice.png"; portrait.write_bytes(b"png")
    chars = tmp_path / "chars.json"; chars.write_text(json.dumps({"alice": "alice.png"}))
    compiler = rc.RecursiveCompiler(story_id="s", story_dir=tmp_path / "story_s", screenplay=screenplay,
                                    characters=chars)
    monkeypatch.setattr(compiler, "skeleton", lambda: skeleton())
    seen_worlds = []
    plans = {"shot01_leave": [segment("seg_a0", 0, 6), segment("seg_a1", 1, 6)],
             "shot02_fountain": [segment("seg_b0", 0, 6)], "shot03_replay": [segment("seg_c0", 0, 5)]}

    def fake_segments(index, world):
        seen_worlds.append(world)
        return plans[skeleton()["shots"][index]["id"]]
    monkeypatch.setattr(compiler, "_segments", fake_segments)
    runner = SimpleNamespace(plan=compiler.initial_plan(), references={},
                             world=WorldLedger(tmp_path / "w.json"))
    out = []
    for request in compiler.requests(runner):
        out.append(request)
        if request.shot_id == "seg_a1":   # executed and admitted before shot02 is expanded
            runner.world.admit(shot_id="seg_a1", scene_id="hall", event_id="shot01_leave", world_time=6,
                               duration=6, world_version="v1", video=Path("v.mp4"), anchor=None,
                               recapture=False, event_offset=6.0)
    ids = [r.shot_id for r in out]
    assert ids == ["seg_a0", "seg_a1", "bridge_tr01", "seg_b0", "seg_c0"]
    assert seen_worlds[0]["scenes"] == {} and "hall" in seen_worlds[1]["scenes"]
    by_id = {r.shot_id: r for r in out}
    assert by_id["seg_a0"].camera_relation == "new_scene" and by_id["seg_a1"].camera_relation == "continuous"
    assert by_id["seg_a1"].raw["event_offset"] == 6.0 and by_id["seg_a1"].start == 6.0
    assert by_id["bridge_tr01"].raw["film_only"] and by_id["bridge_tr01"].start == 12.0
    # a bridge shows the people of the shots it joins, with their identity references
    assert list(by_id["bridge_tr01"].entity_ids) == ["alice"]
    assert by_id["bridge_tr01"].raw["character_reference_map"] == {"alice": str(portrait)}
    assert by_id["seg_b0"].start == 16.0 and by_id["seg_b0"].camera_relation == "new_scene"
    replay = by_id["seg_c0"]
    assert replay.is_recapture and replay.raw["source_event_id"] == "shot01_leave"
    assert (replay.raw["source_time_start"], replay.raw["source_time_end"]) == (6.0, 11.0)
    assert replay.event_id == "shot01_leave" and replay.raw["event_offset"] == 6.0
    assert [s.shot_id for s in runner.plan.shots] == ids
    from atas.runner.pipeline import bind_request
    binding = bind_request(replay, runner.plan, runner.world, memory="full")
    assert binding.mode == "recapture" and binding.recapture["source_shot_id"] == "seg_a1"
    assert binding.recapture["source_time_start"] == 0.0



def test_recapture_interval_is_moved_inside_one_source_clip(tmp_path):
    screenplay = tmp_path / "story.txt"; screenplay.write_text("Alice leaves the hall.")
    compiler = rc.RecursiveCompiler(story_id="s", story_dir=tmp_path / "story_s",
                                    screenplay=screenplay, characters=None)
    compiler.folder.mkdir(parents=True, exist_ok=True)
    (compiler.folder / "segments_src.json").write_text(json.dumps(
        {"segments": [{"duration_s": 6}, {"duration_s": 6}]}))
    assert compiler._within_one_source_segment("src", 3.0, 9.0) == (0.0, 6.0)      # spans the cut at 6 s
    assert compiler._within_one_source_segment("src", 7.0, 11.0) == (7.0, 11.0)    # already inside clip 2
    assert compiler._within_one_source_segment("src", 4.0, 12.0) == (6.0, 12.0)   # longer than a clip
