"""Requests are bound against admitted memory only (paper Eq. 7 and Eq. 11)."""
from pathlib import Path

from atas.runner.pipeline import bind_request
from atas.request import plan_from_payload
from atas.world.ledger import WorldLedger
def plan(*rows):
    shots = []
    for i, row in enumerate(rows):
        shots.append({"id": row[0], "location_id": row[1], "camera_relation": row[2], "event_id": row[0],
                      "duration": 5, "start": 5 * i, **(row[3] if len(row) > 3 else {})})
    return plan_from_payload("story", {"shots": shots}, Path("plan.json"))


def admit(world, shot, scene, event=None, offset=0.0, duration=5.0):
    world.admit(shot_id=shot, scene_id=scene, event_id=event or shot, world_time=0.0, duration=duration,
                world_version="v-" + shot, video=Path("x.mp4"), anchor=None, recapture=False, event_offset=offset)


def test_first_observation_is_cold_start(tmp_path):
    p = plan(("a", "room", "new_scene"))
    b = bind_request(p.shots[0], p, WorldLedger(tmp_path / "w.json"), memory="full")
    assert (b.mode, b.relation, b.still_mode) == ("cold_start", "new_scene", "gpt_cold")


def test_unadmitted_scene_does_not_block_later_shots(tmp_path):
    p = plan(("a", "room", "new_scene"), ("b", "room", "continuous"))
    b = bind_request(p.shots[1], p, WorldLedger(tmp_path / "w.json"), memory="full")
    assert b.mode == "cold_start" and "scene_memory_unavailable" in b.notes and b.still_mode == "gpt_cold"


def test_continuous_without_memory_keeps_frame_continuity_from_delivered_film(tmp_path):
    p = plan(("a", "room", "new_scene"), ("b", "room", "continuous"))
    world = WorldLedger(tmp_path / "w.json")
    world.record_shot("a", {"scene_id": "room", "video": str(tmp_path / "a.mp4"), "admitted": False})
    b = bind_request(p.shots[1], p, world, memory="full")
    assert (b.mode, b.still_mode, b.start_from) == ("cold_start", "continue", "a")
    assert "frame_continuity_only" in b.notes


def test_continuous_requires_admitted_immediate_predecessor(tmp_path):
    p = plan(("a", "room", "new_scene"), ("b", "room", "cut_recompose"), ("c", "room", "continuous"))
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "a", "room")
    b = bind_request(p.shots[2], p, world, memory="full")
    assert b.mode == "revisit" and "continuity_source_not_admitted" in b.notes
    admit(world, "b", "room")
    b = bind_request(p.shots[2], p, world, memory="full")
    assert (b.mode, b.start_from, b.still_mode) == ("continue", "b", "continue")


def test_revisit_uses_scene_memory(tmp_path):
    p = plan(("a", "room", "new_scene"), ("b", "yard", "new_scene"), ("c", "room", "revisit"))
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "a", "room")
    b = bind_request(p.shots[2], p, world, memory="full")
    assert (b.mode, b.relation, b.still_mode) == ("revisit", "cut_recompose", "gpt_anchor")


def test_event_retrieval_is_indexed_by_event_time(tmp_path):
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "seg0", "room", event="ev", offset=0.0, duration=6.0)
    admit(world, "seg1", "room", event="ev", offset=6.0, duration=8.0)
    record = world.retrieve_event("room", "ev", 7.0, 12.0)
    assert record["shot_id"] == "seg1" and record["local_start"] == 1.0 and record["local_end"] == 6.0
    assert world.retrieve_event("yard", "ev", 7.0, 12.0) is None


def test_recapture_binds_to_the_covering_clip(tmp_path):
    p = plan(("seg0", "room", "new_scene"), ("seg1", "room", "continuous"),
             ("r", "room", "recapture", {"render_mode": "recapture", "source_event_id": "ev",
                                         "source_time_start": 7.0, "source_time_end": 12.0}))
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "seg0", "room", event="ev", offset=0.0, duration=6.0)
    admit(world, "seg1", "room", event="ev", offset=6.0, duration=8.0)
    b = bind_request(p.shots[2], p, world, memory="full")
    assert b.mode == "recapture" and b.recapture["source_shot_id"] == "seg1"
    assert (b.recapture["source_time_start"], b.recapture["source_time_end"]) == (1.0, 6.0)


def test_missing_event_record_gives_empty_event_context(tmp_path):
    p = plan(("a", "room", "new_scene"),
             ("r", "room", "recapture", {"render_mode": "recapture", "source_event_id": "ev",
                                         "source_time_start": 0.0, "source_time_end": 5.0}))
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "a", "room")          # scene memory exists, event "ev" was never admitted
    b = bind_request(p.shots[1], p, world, memory="full")
    assert b.mode == "revisit" and "event_memory_unavailable" in b.notes


def test_memory_none_ablation_never_uses_memory(tmp_path):
    p = plan(("a", "room", "new_scene"), ("b", "room", "continuous"))
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "a", "room")
    assert bind_request(p.shots[1], p, world, memory="none").mode == "cold_start"


def test_bridge_is_film_only(tmp_path):
    p = plan(("bridge_x", "bridge_x", "new_scene", {"film_only": True}))
    b = bind_request(p.shots[0], p, WorldLedger(tmp_path / "w.json"), memory="full")
    assert "film_only_transition" in b.notes


def test_ledger_persists(tmp_path):
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "a", "room")
    again = WorldLedger(tmp_path / "w.json")
    assert again.last_admitted("room") == "a" and again.has_memory("room")


def test_executed_relation_replaces_unavailable_recapture_downstream(tmp_path):
    from atas.runner.pipeline import apply_binding
    p = plan(("a", "room", "new_scene"),
             ("r", "room", "recapture", {"render_mode": "recapture", "source_event_id": "ev",
                                         "source_time_start": 0.0, "source_time_end": 5.0}))
    world = WorldLedger(tmp_path / "w.json")
    admit(world, "a", "room")
    shot = apply_binding(p.shots[1], bind_request(p.shots[1], p, world, memory="full"))
    assert not shot.is_recapture and shot.camera_relation == "cut_recompose"
    assert "render_mode" not in shot.raw and shot.raw["recapture_requested_but_unavailable"]


def test_opening_stills_are_regenerated_until_landscape(tmp_path, monkeypatch):
    from PIL import Image
    from atas.providers import imagegen
    sizes = iter([(1024, 1536), (1672, 941)])
    calls = []
    def fake(prompt, output, **kw):
        calls.append(prompt)
        Image.new("RGB", next(sizes)).save(output)
        return output
    monkeypatch.setattr(imagegen, "_generate", fake)
    out = imagegen.generate("a hall", tmp_path / "still.png", aspect=16 / 9)
    assert Image.open(out).size == (1672, 941) and len(calls) == 2 and "16:9" in calls[1]


def test_review_rows_for_people_outside_the_registry_are_ignored():
    from types import SimpleNamespace
    import pytest
    from atas.world.scene_state import validate_observations
    shot = SimpleNamespace(raw={"_scene_context": {"entities": {"alice": {}}}})
    row = lambda label: {"label": label, "visibility": "visible", "confidence": 0.9, "evidence": "f0"}
    assert [r["label"] for r in validate_observations([row("alice"), row("Passerby")], shot)] == ["alice"]
    with pytest.raises(ValueError):
        validate_observations([row("Passerby")], shot)


def test_actor_seen_only_in_unadmitted_film_is_ungrounded_for_the_planner():
    from types import SimpleNamespace
    from atas.shot.camera_planner import memory_continuity
    ctx = {"entities": {"a": {"source_shot_id": "s0"}, "b": {"source_shot_id": "s1"}}}
    shot = SimpleNamespace(raw={"_scene_context": ctx, "_world_admitted_ids": ["s1"]})
    view = memory_continuity(shot)["entities"]
    assert view["a"]["source_shot_id"] is None and view["a"]["film_only_observation"] == "s0"
    assert view["b"]["source_shot_id"] == "s1" and ctx["entities"]["a"]["source_shot_id"] == "s0"


def test_evidence_may_join_several_verbatim_quotes():
    from types import SimpleNamespace
    from atas.shot.camera_planner import quotes_source
    shot = SimpleNamespace(raw={"camera_brief": "镜头贴着书架侧向移动，随后镜头沿书架缓慢向左平移"}, prompt="",
                           provider_prompt="", shot_perspective="")
    assert quotes_source("镜头贴着书架侧向移动；镜头沿书架缓慢向左平移", shot)
    assert not quotes_source("镜头贴着书架侧向移动；镜头快速推近", shot)


def test_stored_response_is_revalidated_after_a_validator_fix(tmp_path):
    import json
    from atas.runner.ai_recovery import corrected_json
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "attempt_00.json").write_text(json.dumps({"index": 0, "status": "invalid", "response": "ok"}))
    def never(history):
        raise AssertionError("no new call needed")
    assert corrected_json(tmp_path, "k", never, lambda text: text.upper()) == "OK"


def test_image_backend_switches_when_the_probe_retry_reveals_a_missing_model(tmp_path, monkeypatch):
    import sys, types
    from atas.providers import imagegen
    calls = iter([RuntimeError("Error code: 524"), RuntimeError("model_not_found")])
    class Gen:
        retries = 4
        def __init__(self, *_): pass
        def generate(self, *a, **k): raise next(calls)
    monkeypatch.setitem(sys.modules, "atas.providers.gpt_image", types.SimpleNamespace(GPTImage2Generator=Gen))
    monkeypatch.setattr(imagegen, "_BACKEND", "auto")
    used = []
    monkeypatch.setattr(imagegen, "_responses_tool", lambda p, out, refs: (used.append(p), out.write_bytes(b"x")))
    imagegen._generate("a hall", tmp_path / "x.png")
    assert used and imagegen._BACKEND == "responses_tool"


def test_registration_reason_carries_measured_failure():
    from atas.runner.pipeline import registration_summary
    diagnosis = {"chunks_total": 12, "failed_chunks": [
        {"chunk": 3, "checks": ["heldout_history_median"], "heldout_x_limit": 1.3},
        {"chunk": 7, "checks": ["chunk_seam"], "seam_x_limit": 1.04}]}
    text = registration_summary(diagnosis)
    assert "2/12 chunks" in text and "heldout max 1.3x" in text and "seam max 1.04x" in text
    assert registration_summary(None) == ""


def test_actor_from_another_admitted_clip_is_staged_not_searched():
    from types import SimpleNamespace
    from atas.shot.camera_planner import memory_continuity
    ctx = {"entities": {"a": {"source_shot_id": "s1", "location": "by the door"}, "b": {"source_shot_id": "s2"}}}
    raw = {"_scene_context": ctx, "_world_admitted_ids": ["s1", "s2"]}
    view = memory_continuity(SimpleNamespace(raw=raw), "s2")["entities"]
    assert view["a"]["presence"] == "planned" and view["a"]["last_observed_location"] == "by the door"
    assert view["b"]["source_shot_id"] == "s2"
    recapture = memory_continuity(SimpleNamespace(raw={**raw, "render_mode": "recapture"}), "s2")["entities"]
    assert recapture["a"]["source_shot_id"] == "s2"           # grounded in the event frames


def test_recapture_grounds_participants_seen_only_in_unadmitted_film():
    from types import SimpleNamespace
    from atas.shot.camera_planner import memory_continuity
    raw = {"_scene_context": {"entities": {"su_qing": {"source_shot_id": "not_admitted"}}},
           "_world_admitted_ids": ["event_clip"], "render_mode": "recapture"}
    row = memory_continuity(SimpleNamespace(raw=raw), "event_clip")["entities"]["su_qing"]
    assert row["source_shot_id"] == "event_clip" and row.get("presence") != "planned"


def test_oversized_condition_video_is_capped_before_upload(tmp_path, monkeypatch):
    import numpy as np, imageio.v2 as imageio
    from atas import media
    video = tmp_path / "memory_input.mp4"
    rng = np.random.default_rng(0)
    with imageio.get_writer(str(video), fps=24, codec="libx264", macro_block_size=1, quality=10) as w:
        for _ in range(24):
            w.append_data(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
    assert media.fit_upload_limit(video) == video
    monkeypatch.setattr(media, "UPLOAD_LIMIT_BYTES", video.stat().st_size - 1)
    capped = media.fit_upload_limit(video)
    assert capped != video and capped.stat().st_size < video.stat().st_size


def test_cross_shot_rejection_reason_is_measured():
    from atas.runner.pipeline import registration_summary
    text = registration_summary({"rejection_category": "cross_shot_alignment", "failed_chunks": [],
                                 "cross_shot": {"correspondences": 376, "median_x_limit": 1.08,
                                                "within_2t_fraction": .98}})
    assert "376 old/new matches" in text and "1.08x" in text and "98%" in text


def test_provider_internal_error_is_retried_but_bad_input_is_not():
    from atas.runner.pipeline import transient
    assert transient(RuntimeError("task ended with 'failed': {'code': 'PeRuntimeError', "
                                  "'message': '<500> InternalError.Algo: Prompt enhancement service internal error'}"))
    assert not transient(RuntimeError("{'code': 'InvalidParameter', 'message': 'File size exceeds maximum'}"))


def test_landscape_still_of_another_shape_is_cropped_not_regenerated(tmp_path, monkeypatch):
    from PIL import Image
    from atas.providers import imagegen
    calls = []
    def fake(prompt, output, **kw):
        calls.append(prompt); Image.new("RGB", (1536, 1024)).save(output); return output
    monkeypatch.setattr(imagegen, "_generate", fake)
    out = imagegen.generate("a hall", tmp_path / "still.png", aspect=16 / 9)
    w, h = Image.open(out).size
    assert len(calls) == 1 and abs(w / h - 16 / 9) < 0.01 and w == 1536


def test_memory_render_is_sent_to_the_generator_by_default():
    from atas.providers.settings import Wan27Config
    assert Wan27Config().include_geometry_reference is True


def test_bridge_opens_from_the_previous_delivered_clip(tmp_path):
    p = plan(("a", "room", "new_scene"), ("bridge_x", "bridge_x", "new_scene", {"film_only": True}))
    world = WorldLedger(tmp_path / "w.json")
    assert bind_request(p.shots[1], p, world, memory="full").still_mode == "gpt_cold"   # nothing delivered yet
    world.record_shot("a", {"scene_id": "room", "video": str(tmp_path / "a.mp4"), "admitted": False})
    b = bind_request(p.shots[1], p, world, memory="full")
    assert (b.still_mode, b.start_from) == ("continue", "a") and "film_only_transition" in b.notes
