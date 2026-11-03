"""End-to-end execution loop with fake providers and real media files.

Story: s1 (room, new) -> s2 (room, continuous) -> s3 (yard, review always holds)
       -> s4 (room, revisit, registration rejected) -> s5 (room, recapture of s1)
Checks the f/r/a decisions, memory binding, event retrieval and the final film.
"""
import asyncio
from dataclasses import replace
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from atas.review.film_review import file_hash
from atas.runner.pipeline import Policy, StoryRunner
from atas.request import plan_from_payload
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def write_video(path: Path, seconds: float, color: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=24, codec="libx264", macro_block_size=1) as writer:
        for i in range(int(seconds * 24)):
            writer.append_data(np.full((36, 64, 3), (color + i) % 255, dtype=np.uint8))
    return path


def review_for(video, decision="pass"):
    return {"schema_version": 1, "stage": "film", "decision": decision, "cache_key": "k" + file_hash(video)[:8],
            "video_sha256": file_hash(video), "lineage": {}, "summary": "fake",
            "checks": [{"criterion": c, "status": "pass" if decision != "hold" else "fail", "severity": "critical",
                        "confidence": 0.95, "evidence": "fake"} for c in
                       ("subject", "framing", "motion", "action", "event", "world_consistency")],
            "scene_observations": [{"label": "alice", "visibility": "visible", "confidence": 0.9,
                                    "evidence": "fake", "location": "center"}],
            "observed_dynamic_classes": [{"noun": "person", "evidence": "fake"}]}


class FakeGeometry:
    enabled = True
    rejected = {"s4"}

    def __init__(self, *a, **k):
        self.renders, self.recons, self.jobs = [], [], {}

    def render(self, intent):
        self.renders.append(intent)
        job = SimpleNamespace(job_id=f"render-{intent['shot_id']}-{len(self.renders)}")
        self.jobs[job.job_id] = {"status": "succeeded", "report": {"start_from_role": "previous_shot"}}
        return job

    crash = set()

    def recon(self, **kw):
        self.recons.append(kw)
        shot = kw["shot_id"]
        if shot in self.crash:
            import atas.world.geometry_client as geo
            raise geo.GeometryJobFailed("RuntimeError: Vista4D recon exited 1")
        registered = shot not in self.rejected
        job = SimpleNamespace(job_id=f"recon-{shot}")
        self.jobs[job.job_id] = {"status": "succeeded", "job_id": job.job_id, "cameras_npz": "c", "l0_dir": "l0",
                                 "l1_npz": "l1", "l1_point_count": 10,
                                 "fuse": {"registered": registered, "publication_status": "pending_review",
                                          "candidate_job_id": job.job_id, "reason": "low overlap"}}
        return job

    def wait(self, job_id, **kw):
        return self.jobs[job_id]

    def fetch_memory_input(self, job, dest):
        return write_video(dest, 4, 100)

    def _request(self, method, path, payload=None, **kw):
        raise RuntimeError("entity projection unavailable in this fake")


@pytest.fixture
def story(tmp_path, monkeypatch):
    import atas.shot.camera_planner as cp
    import atas.review.editorial as ed
    import atas.world.geometry_client as geo
    import atas.shot.stills as im
    import atas.review.film_review as ir
    import atas.world.keywords as kw
    import atas.shot.prompt_author as pa
    import atas.runner.publication_recovery as pr
    import atas.providers.wan.generation as wan

    alice = tmp_path / "alice.png"
    Image.new("RGB", (8, 8)).save(alice)
    base = {"duration": 4, "entity_ids": ["alice"], "onstage_labels": ["alice"],
            "character_reference_images": [str(alice)], "character_reference_map": {"alice": str(alice)},
            "prompt": "Alice acts.", "camera_move_name": "push_in", "shot_size": "medium"}
    rows = [
        {"id": "s1", "location_id": "room", "camera_relation": "new_scene", "event_id": "ev1", "start": 0},
        {"id": "s2", "location_id": "room", "camera_relation": "continuous", "event_id": "ev2", "start": 4},
        {"id": "s3", "location_id": "yard", "camera_relation": "new_scene", "event_id": "ev3", "start": 8},
        {"id": "s4", "location_id": "room", "camera_relation": "revisit", "event_id": "ev4", "start": 12},
        {"id": "s5", "location_id": "room", "camera_relation": "recapture", "event_id": "ev1", "start": 16,
         "render_mode": "recapture", "source_event_id": "ev1", "source_time_start": 0.0, "source_time_end": 4.0},
    ]
    plan = plan_from_payload("demo", {"title": "demo", "character_reference_map": {"alice": str(alice)},
                                      "shots": [{**base, **r} for r in rows]}, tmp_path / "plan.json")

    fake_geometry = FakeGeometry()
    monkeypatch.setattr(geo, "GeometryClient", lambda *a, **k: fake_geometry)
    monkeypatch.setattr(cp, "plan_camera", lambda g, intent, *a, **k: {**intent, "camera_plan": {
        "subjects": [{"label": "alice", "framing_role": "focal", "kind": "dynamic"}], "framing": {}, "motion": {}}})
    monkeypatch.setattr(wan, "build_request", lambda shot, **k: (SimpleNamespace(
        condition=SimpleNamespace(references=()), output_dir=k["output_dir"], shot=shot), [], []))
    monkeypatch.setattr(pa, "author_prompts", lambda shot, *a, **k: shot)

    async def generate(request):
        return write_video(Path(request.output_dir) / "wan" / "result.mp4", request.shot.duration,
                           hash(request.shot.shot_id) % 200)
    monkeypatch.setattr(wan, "generate_shot", generate)
    for name in ("generate_cold_start_frame", "generate_anchor_still", "generate_geometry_aligned_still"):
        def still(shot, *a, output=None, **k):
            path = Path(output or a[0])
            Image.new("RGB", (64, 36), (10, 20, 30)).save(path)
            return path
        monkeypatch.setattr(im, name, still)
    monkeypatch.setattr(im, "write_composition_guide", lambda video, report, out: (Image.new("RGB", (8, 8)).save(out), out)[1])

    def review(shot, video, folder, *, stage, **k):
        return review_for(video, "hold" if shot.shot_id == "s3" else "pass")
    monkeypatch.setattr(ir, "review_shot", review)

    def best(shot, shot_dir):
        video = Path(shot_dir) / "result.mp4"
        return video, review_for(video, "hold"), {"same_scene": True, "actual_event": "partial"}, None
    monkeypatch.setattr(ed, "select_best_available", best)
    monkeypatch.setattr(kw, "sam_nouns_for_recon", lambda *a, **k: ("person",))
    monkeypatch.setattr(pr, "publish_with_rebase", lambda g, d, current, payload: (
        current, {"world_version": "v-" + payload["shot_id"], "publication_status": "committed"}))
    return plan, fake_geometry, tmp_path


def test_full_loop_delivers_every_shot_and_admits_only_f_r_a(story):
    plan, geometry, tmp_path = story
    runner = StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs", policy=Policy(film_repairs=1))
    manifest = asyncio.run(runner.run())

    rows = {r["shot_id"]: r for r in manifest["shots"]}
    assert manifest["shots_delivered"] == 5 and manifest["complete"] and Path(manifest["final_video"]).is_file()
    # f / r / a decisions
    assert rows["s1"]["admitted"] and rows["s2"]["admitted"] and rows["s5"]["admitted"]
    assert rows["s3"]["film_status"] == "best_available" and not rows["s3"]["f"] and not rows["s3"]["admitted"]
    assert rows["s4"]["f"] and not rows["s4"]["r"] and not rows["s4"]["admitted"]
    assert "registration_rejected" in rows["s4"]["reason"]
    # binding against admitted memory
    assert [rows[s]["mode"] for s in ("s1", "s2", "s3", "s4", "s5")] == [
        "cold_start", "continue", "cold_start", "revisit", "recapture"]
    recapture = [r for r in geometry.renders if r["shot_id"] == "s5"][0]
    assert recapture["source_shot_id"] == "s1" and recapture["geometry_layers"] == "l1+l2"
    # f=0 never reaches reconstruction; world time reaches the GPU
    assert {r["shot_id"] for r in geometry.recons} == {"s1", "s2", "s4", "s5"}
    assert [r["world_time_start"] for r in geometry.recons if r["shot_id"] == "s2"] == [4.0]
    ledger = json.loads((tmp_path / "runs/story_demo/world_ledger.json").read_text())
    assert ledger["scenes"]["room"]["admitted"] == ["s1", "s2", "s5"]
    assert "yard" not in ledger["scenes"] and list(ledger["events"]) == ["ev1", "ev2"]

    # Resume: everything is final, nothing is regenerated or reconstructed again.
    renders, recons = len(geometry.renders), len(geometry.recons)
    report = json.loads((tmp_path / "runs/story_demo/report.json").read_text())
    assert report["state_update_rate"] == 3 / 5 and report["modes"]["recapture"] == 1
    again = asyncio.run(StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs").run())
    assert again["shots_delivered"] == 5 and len(geometry.renders) == renders and len(geometry.recons) == recons


def test_reconstruction_crash_keeps_reviewed_film_and_never_refilms(story):
    plan, geometry, tmp_path = story
    geometry.crash = {"s2"}
    runner = StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs", policy=Policy(film_repairs=1))
    manifest = asyncio.run(runner.run())
    rows = {r["shot_id"]: r for r in manifest["shots"]}
    assert rows["s2"]["film_status"] == "reviewed" and rows["s2"]["f"] and not rows["s2"]["r"]
    assert rows["s2"]["reason"].startswith("reconstruction_failed")
    assert not (tmp_path / "runs/story_demo/s2__degraded").exists()
    assert manifest["shots_delivered"] == 5


def test_unadmitted_event_source_is_retaken_so_the_recapture_has_its_event(story):
    plan, geometry, tmp_path = story
    shots = tuple(replace(s, raw={**s.raw, "recapture_source": True}) if s.shot_id == "s1" else s
                  for s in plan.shots)
    plan = replace(plan, shots=shots)
    first_take = geometry.recon
    def recon(**kw):
        job = first_take(**kw)
        if kw["shot_id"] == "s1" and "__take2" not in str(kw["video"]):
            geometry.jobs[job.job_id]["fuse"] = {"registered": False, "reason": "internal_reconstruction_inconsistent"}
        return job
    geometry.recon = recon
    manifest = asyncio.run(StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs",
                                       policy=Policy(film_repairs=1)).run())
    rows = {r["shot_id"]: r for r in manifest["shots"]}
    assert rows["s1"]["admitted"] and "__take2" in rows["s1"]["video"]
    assert rows["s1"]["earlier_takes"][0]["reason"].startswith("registration_rejected")
    assert rows["s5"]["mode"] == "recapture"
    # an ordinary unadmitted shot is never retaken
    assert not (tmp_path / "runs/story_demo/s4__take2").exists()


def test_resume_waits_for_inflight_reconstruction_instead_of_re_reviewing(story, monkeypatch):
    import atas.review.film_review as ir
    plan, geometry, tmp_path = story
    asyncio.run(StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs", policy=Policy(film_repairs=1)).run())
    shot_dir = tmp_path / "runs/story_demo/s4"
    ledger = json.loads((tmp_path / "runs/story_demo/world_ledger.json").read_text())
    ledger["shots"].pop("s4")                              # killed while s4 was reconstructing
    (tmp_path / "runs/story_demo/world_ledger.json").write_text(json.dumps(ledger))
    (shot_dir / "geometry_recon_state.json").write_text(json.dumps({"status": "waiting", "job_id": "recon-s4"}))
    monkeypatch.setattr(ir, "review_shot", lambda *a, **k: pytest.fail("in-flight film must not be re-reviewed"))
    again = asyncio.run(StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs", policy=Policy(film_repairs=1)).run())
    row = {r["shot_id"]: r for r in again["shots"]}["s4"]
    assert row["f"] and row["film_status"] == "reviewed"


def test_reviewed_film_is_kept_when_only_admission_keeps_failing(story):
    import atas.world.geometry_client as geo
    plan, geometry, tmp_path = story
    def recon(**kw):
        if kw["shot_id"] == "s2":
            raise geo.GeometryWaitUncertain("geometry job polling unavailable")
        return FakeGeometry.recon(geometry, **kw)
    geometry.recon = recon
    manifest = asyncio.run(StoryRunner(plan, run_id="run", run_dir=tmp_path / "runs",
                                       policy=Policy(film_repairs=1, backoff_seconds=(0, 0, 0))).run())
    row = {r["shot_id"]: r for r in manifest["shots"]}["s2"]
    assert row["film_status"] == "reviewed" and row["f"] and not row["r"]
    assert row["reason"].startswith("admission_failed") and "__degraded" not in str(row["video"])
    assert not (tmp_path / "runs/story_demo/s2__degraded").exists()
