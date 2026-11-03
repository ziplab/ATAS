"""ATAS execution loop (paper Eq. 2 and Eq. 11).

    (u, W_t) --Compile--> q_t --Query--> C_t --G--> V_t --Reconstruct/Register/Validate--> (V_t, W_{t+1})

Each bounded request is first *bound* against the current world state: it may
only use scene or event memory that was actually admitted. The clip is always
delivered as film when it exists; the world state advances only when the film
review (f), registration (r) and state-update check (a) all pass.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from atas.world import scene_state
from atas.runner.ai_recovery import safe_error
from atas.media import observation_dir
from atas.request import PlanShot, StoryPlan
from atas.runner.recon_ledger import write_json_atomic
from atas.world.ledger import WorldLedger
GEOMETRY_RELATIONS = {"continue", "revisit", "recapture"}


@dataclass
class Policy:
    film_repairs: int = int(os.getenv("ATAS_FILM_REPAIRS", "2"))
    camera_replans: int = int(os.getenv("ATAS_CAMERA_REPLANS", "2"))
    operational_attempts: int = int(os.getenv("ATAS_OPERATIONAL_ATTEMPTS", "3"))
    backoff_seconds: tuple = (15, 60, 180)
    memory: str = os.getenv("ATAS_MEMORY", "full")      # full | none
    # A clip a later shot will recapture is regenerated this many times when it is
    # filmed but not admitted: without its event record the recapture has no L2.
    event_source_retries: int = int(os.getenv("ATAS_EVENT_SOURCE_RETRIES", "1"))


@dataclass
class Binding:
    """How a request is executed against the current world state."""
    mode: str                    # cold_start | continue | revisit | recapture
    relation: str                # camera relation sent to the GPU / planner
    still_mode: str              # gpt_cold | continue | gpt_anchor | event_reference
    start_from: str | None = None
    recapture: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def transient(exc: BaseException) -> bool:
    """Operational failure worth retrying the same stage (not a content verdict)."""
    from atas.world.geometry_client import GeometryWaitUncertain, GeometryJobFailed
    from atas.runner.ai_recovery import content_policy_refusal
    from atas.providers.llm import transient as llm_transient
    if content_policy_refusal(exc):
        return False
    if isinstance(exc, GeometryJobFailed):
        return exc.retryable and exc.error_code in {
            "cuda_out_of_memory", "gpu_capacity_unavailable", "cuda_linalg_initialization_failed",
            "cuda_native_runtime_failed"}
    if isinstance(exc, GeometryWaitUncertain) or llm_transient(exc):
        return True
    text = str(exc).lower()
    return any(t in text for t in ("http 429", "http 500", "http 502", "http 503", "http 504", "http 520",
                                   "http 522", "http 524", "http 530", "gateway time", "timed out",
                                   "connection reset", "temporarily unavailable", "throttl", "queuequota",
                                   "did not finish within", "internalerror", "<500>", "<503>"))


def bind_request(shot: PlanShot, plan: StoryPlan, world: WorldLedger, *, memory: str) -> Binding:
    """Resolve the planned camera relation against admitted scene/event memory."""
    relation = (shot.camera_relation or "").strip().lower()
    has_memory = memory != "none" and world.has_memory(shot.scene_id)
    if shot.raw.get("film_only"):
        # A bridge continues from the delivered previous clip (its last frame opens
        # the bridge), so the transition smooths the cut instead of adding one.
        previous = _previous_in_story(shot, plan)
        if previous and (world.shots.get(previous) or {}).get("video"):
            return Binding("cold_start", "new_scene", "continue", previous,
                           notes=["film_only_transition", "frame_continuity_only"])
        return Binding("cold_start", "new_scene", "gpt_cold", notes=["film_only_transition"])
    if relation == "recapture" or shot.raw.get("render_mode") == "recapture":
        start = float(shot.raw.get("source_time_start", 0.0))
        end = float(shot.raw.get("source_time_end", start + shot.duration))
        event = str(shot.raw.get("source_event_id") or shot.event_id)
        record = world.retrieve_event(shot.scene_id, event, start, end) if memory != "none" else None
        if record is not None and record["local_end"] - record["local_start"] >= shot.duration - 1e-6:
            # The GPU replays the source clip in its own time; the recapture
            # keeps the event's speed, so the interval equals the duration.
            fields = {"render_mode": "recapture", "geometry_layers": "l1+l2",
                      "source_shot_id": record["shot_id"], "source_event_id": event,
                      "source_time_start": record["local_start"],
                      "source_time_end": record["local_start"] + shot.duration}
            return Binding("recapture", "recapture", "event_reference", record["shot_id"], fields)
        # Eq. 7: no compatible event record -> empty event context. The shot is
        # still filmed, as a new view of the scene when its memory exists.
        note = "event_memory_unavailable"
        return (Binding("revisit", "cut_recompose", "gpt_anchor", notes=[note]) if has_memory
                else Binding("cold_start", "new_scene", "gpt_cold", notes=[note]))
    if not has_memory:
        notes = [] if relation in {"", "new_scene"} else ["scene_memory_unavailable"]
        previous = _previous_in_scene(shot, plan)
        if relation == "continuous" and previous and (world.shots.get(previous) or {}).get("video"):
            # No geometry to condition on, but the delivered previous clip still
            # supplies frame continuity (its last frame opens this one).
            return Binding("cold_start", "new_scene", "continue", previous, notes=notes + ["frame_continuity_only"])
        return Binding("cold_start", "new_scene", "gpt_cold", notes=notes)
    if relation == "continuous":
        previous = _previous_in_scene(shot, plan)
        if previous is not None and previous == world.last_admitted(shot.scene_id):
            return Binding("continue", "continuous", "continue", previous)
        return Binding("revisit", "cut_recompose", "gpt_anchor", notes=["continuity_source_not_admitted"])
    return Binding("revisit", "cut_recompose", "gpt_anchor")


def apply_binding(shot: PlanShot, binding: Binding) -> PlanShot:
    """Make every downstream stage (camera planner, prompt author, review) see the
    relation actually executed, not the one originally planned."""
    raw = dict(shot.raw)
    if binding.mode != "recapture" and (shot.is_recapture or raw.get("render_mode") == "recapture"):
        for key in ("render_mode", "source_shot_id", "source_event_id", "source_time_start", "source_time_end"):
            raw.pop(key, None)
        raw["recapture_requested_but_unavailable"] = True
    raw["executed_relation"] = binding.relation
    return replace(shot, camera_relation=binding.relation, raw=raw)


def _previous_in_story(shot: PlanShot, plan: StoryPlan) -> str | None:
    ids = [item.shot_id for item in plan.shots]
    index = ids.index(shot.shot_id) if shot.shot_id in ids else -1
    return ids[index - 1] if index > 0 else None


def _previous_in_scene(shot: PlanShot, plan: StoryPlan) -> str | None:
    previous = None
    for item in plan.shots:
        if item.shot_id == shot.shot_id:
            return previous
        if item.scene_id == shot.scene_id and item.camera_relation != "recapture":
            previous = item.shot_id
    return previous


def registration_summary(diagnosis) -> str:
    """Compact, measurable reason: how many chunks failed which check, by how much."""
    if not isinstance(diagnosis, dict):
        return ""
    cross = diagnosis.get("cross_shot") or {}
    if not diagnosis.get("failed_chunks"):
        if diagnosis.get("rejection_category") == "cross_shot_alignment" and cross.get("correspondences") is not None:
            return (f" ({cross['correspondences']} old/new matches, median {cross.get('median_x_limit')}x limit, "
                    f"{round(100 * (cross.get('within_2t_fraction') or 0))}% within 2t)")
        return ""
    failed = diagnosis["failed_chunks"]
    worst = {}
    for row in failed:
        for key in ("heldout_x_limit", "seam_x_limit"):
            if row.get(key) is not None:
                worst[key] = max(worst.get(key, 0), row[key])
    checks = sorted({c for row in failed for c in row.get("checks", [])})
    detail = ", ".join(f"{k.replace('_x_limit', '')} max {v}x" for k, v in sorted(worst.items()))
    return f" ({len(failed)}/{diagnosis.get('chunks_total', '?')} chunks: {'+'.join(checks)}; {detail})"


def shot_stats(shot_dir: Path) -> dict[str, Any]:
    """Measured cost of one shot: stage wall time, regenerations, GPU memory."""
    def read(name):
        path = shot_dir / name
        try:
            return json.loads(path.read_text()) if path.is_file() else {}
        except ValueError:
            return {}
    stages = read("stages.json").get("stages", [])
    seconds: dict[str, float] = {}
    for current, following in zip(stages, stages[1:]):
        seconds[current["stage"]] = seconds.get(current["stage"], 0.0) + following["at"] - current["at"]
    ledger = read("repairs/film/ledger.json")
    recon = read("geometry_recon_result.json")
    render = read("geometry_render_result.json")
    timings = recon.get("timings") or {}
    gpu_peak = {f"recon:{p['stage']}": p.get("gpu_reserved_peak_mib") for p in timings.get("recon_phases", [])
                if p.get("gpu_reserved_peak_mib") is not None}
    if (recon.get("fuse") or {}).get("gpu_reserved_peak_mib") is not None:
        gpu_peak["registration"] = recon["fuse"]["gpu_reserved_peak_mib"]
    if (render.get("report") or {}).get("gpu_reserved_peak_mib") is not None:
        gpu_peak["render"] = render["report"]["gpu_reserved_peak_mib"]
    return {"stage_seconds": seconds,
            "wall_seconds": (stages[-1]["at"] - stages[0]["at"]) if len(stages) > 1 else None,
            "film_regenerations": len(ledger.get("attempts", [])),
            "first_review_held": bool(ledger.get("attempts")),
            "gpu_reserved_peak_mib": gpu_peak,
            "recon_timings": timings, "l1_point_count": recon.get("l1_point_count")}


class StoryRunner:
    def __init__(self, plan: StoryPlan, *, run_id: str, run_dir: Path, policy: Policy | None = None):
        from atas.world.geometry_client import GeometryClient
        self.plan, self.run_id, self.run_dir = plan, run_id, Path(run_dir)
        self.story_dir = self.run_dir / f"story_{plan.story_id}"
        self.story_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy or Policy()
        self.world = WorldLedger(self.story_dir / "world_ledger.json")
        self.geometry = GeometryClient(enabled=self.policy.memory != "none")
        self.references = scene_state.catalog(plan)
        self.scene_states: dict[str, dict] = {}
        for row in self.world.shots.values():
            if row.get("scene_state"):
                self.scene_states[row["scene_id"]] = row["scene_state"]

    # ------------------------------------------------------------------
    def shot_dir(self, shot: PlanShot) -> Path:
        path = self.story_dir / shot.shot_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, shot: PlanShot | None, message: str) -> None:
        prefix = f"[{self.plan.story_id}/{shot.shot_id}]" if shot else f"[{self.plan.story_id}]"
        print(time.strftime("%H:%M:%S"), prefix, message, flush=True)

    def stage(self, shot_dir: Path, name: str) -> None:
        path = shot_dir / "stages.json"
        data = json.loads(path.read_text()) if path.is_file() else {"stages": []}
        data["stages"].append({"stage": name, "at": time.time()})
        write_json_atomic(path, data)

    # ------------------------------------------------------------------
    async def run(self, requests=None) -> dict[str, Any]:
        """Execute requests in order. ``requests`` may be a lazy iterator from a
        recursive compiler that reads ``self.world`` between requests."""
        rows = []
        for shot in (requests if requests is not None else self.plan.shots):
            previous = self.world.shots.get(shot.shot_id)
            if previous and previous.get("final"):
                rows.append(previous)
                continue
            row = await self.run_shot(shot)
            rows.append(row)
        from atas.film.assemble import assemble
        from atas.film.report import write_report
        manifest = assemble(self.story_dir, rows, title=self.plan.title)
        write_report(self.story_dir, manifest)
        self.log(None, f"final film: {manifest['final_video']} "
                       f"({manifest['shots_delivered']}/{manifest['shots_total']} delivered, "
                       f"{manifest['admitted_to_world']} admitted to world)")
        return manifest

    async def run_shot(self, shot: PlanShot) -> dict[str, Any]:
        shot_dir = self.shot_dir(shot)
        binding = bind_request(shot, self.plan, self.world, memory=self.policy.memory)
        self.log(shot, f"mode={binding.mode} relation={binding.relation} {' '.join(binding.notes)}")
        try:
            row = await self._with_retries(shot, shot_dir, binding, admit=True)
        except Exception as error:  # noqa: BLE001 - the conditioned path is exhausted
            row = await self._fallback(shot, binding, error)
        row = await self._retry_event_source(shot, binding, row)
        row["final"] = True
        self.world.record_shot(shot.shot_id, row)
        return row

    async def _retry_event_source(self, shot, binding, row):
        """Regenerate a recapture source whose film passed but whose observation was
        not admitted; deliver whichever take entered memory (both passed review)."""
        if (self.policy.memory != "full" or not shot.raw.get("recapture_source") or row.get("admitted")
                or not row.get("f") or not str(row.get("reason") or "").startswith(
                    ("registration_rejected", "reconstruction_failed"))):
            return row
        for take in range(1, self.policy.event_source_retries + 1):
            folder = self.story_dir / f"{shot.shot_id}__take{take + 1}"
            self.log(shot, f"event source not admitted ({row['reason'][:120]}); regenerating take {take + 1}")
            try:
                # A new seed: the generator is otherwise deterministic for the same inputs.
                retry = await self._with_retries(replace(shot, seed=shot.seed + take), folder, binding, admit=True)
            except Exception as exc:  # noqa: BLE001 - keep the delivered first take
                self.log(shot, f"take {take + 1} failed: {safe_error(exc)[:200]}")
                continue
            if retry.get("admitted"):
                retry["earlier_takes"] = [{"video": row.get("video"), "reason": row.get("reason")}]
                return retry
            row.setdefault("earlier_takes", []).append({"video": retry.get("video"), "reason": retry.get("reason")})
        return row

    async def _with_retries(self, shot, shot_dir, binding, *, admit):
        attempts = self.policy.operational_attempts
        for attempt in range(attempts):
            try:
                return await self._execute(shot, shot_dir, binding, admit=admit)
            except Exception as exc:  # noqa: BLE001 - classified below
                write_json_atomic(shot_dir / "failure_diagnostic.json", {
                    "attempt": attempt + 1, "error_type": type(exc).__name__, "error": safe_error(exc),
                    "traceback": safe_error(traceback.format_exc())})
                self.log(shot, f"attempt {attempt + 1} failed: {type(exc).__name__}: {safe_error(exc)[:300]}")
                if not transient(exc) or attempt == attempts - 1:
                    raise
                await asyncio.sleep(self.policy.backoff_seconds[min(attempt, len(self.policy.backoff_seconds) - 1)])
        raise AssertionError("unreachable")

    async def _fallback(self, shot, binding, error) -> dict[str, Any]:
        """The conditioned path failed: still deliver a film, never touch memory."""
        reviewed = self._reviewed_film(self.shot_dir(shot))
        if reviewed is not None:
            # The film already passed review; only admission failed. Deliver it
            # (r = 0) instead of paying for a worse, unconditioned replacement.
            return self._row(shot, binding, video=reviewed, film_status="reviewed", f=True,
                             reason="admission_failed:" + safe_error(error)[:300])
        if "degraded_after_failure" in binding.notes:
            return self._row(shot, binding, video=None, film_status="missing", reason=safe_error(error))
        degraded = Binding("cold_start", "new_scene", "gpt_cold", notes=binding.notes + ["degraded_after_failure"])
        folder = self.story_dir / f"{shot.shot_id}__degraded"
        try:
            row = await self._with_retries(shot, folder, degraded, admit=False)
            row.update(film_status="degraded", reason=safe_error(error))
            return row
        except Exception as exc:  # noqa: BLE001
            return self._row(shot, binding, video=None, film_status="missing",
                             reason=safe_error(error) + " | degraded: " + safe_error(exc))

    @staticmethod
    def _reviewed_film(shot_dir: Path) -> Path | None:
        from atas.review.film_review import file_hash
        film, stored = shot_dir / "result.mp4", shot_dir / "film_intent_review.json"
        if not (film.is_file() and stored.is_file()):
            return None
        review = json.loads(stored.read_text())
        if review.get("decision") in ("pass", "warning") and review.get("video_sha256") == file_hash(film):
            return film
        return None

    def _row(self, shot, binding, *, video, film_status, reason=None, f=False, r=False, a=False,
             world_version=None, scene_state_after=None) -> dict[str, Any]:
        return {"shot_id": shot.shot_id, "scene_id": shot.scene_id, "event_id": shot.event_id,
                "requested_relation": shot.camera_relation, "mode": binding.mode,
                "relation": binding.relation, "binding_notes": binding.notes,
                "video": str(video) if video else None, "film_status": film_status,
                "f": f, "r": r, "a": a, "admitted": bool(f and r and a),
                "world_version": world_version, "reason": reason, "scene_state": scene_state_after}

    # ------------------------------------------------------------------
    async def _execute(self, shot: PlanShot, shot_dir: Path, binding: Binding, *, admit: bool = True) -> dict:
        from atas.shot.camera_planner import camera_conditioned_shot
        from atas.media import existing_artifact
        from atas.providers.wan.generation import build_request, generate_shot
        shot_dir.mkdir(parents=True, exist_ok=True)
        geometry = self.geometry
        recapture = binding.recapture
        shot = apply_binding(shot, binding)

        # ---- textual scene context (who is here, last observed positions)
        event_state = None
        if recapture:
            source_dir = observation_dir(self.story_dir, recapture["source_shot_id"])
            before = source_dir / "scene_state_before.json"
            after = source_dir / "scene_state_after.json"
            event_state = scene_state.event_snapshot(
                json.loads(before.read_text()) if before.is_file() else {},
                json.loads(after.read_text()) if after.is_file() else {},
                recapture["source_shot_id"], recapture["source_time_start"])
        previous_state = self.scene_states.get(shot.scene_id, {})
        shot = scene_state.prepare(shot, previous_state, self.references, recapture_state=event_state)
        history = ([recapture["source_shot_id"]] if recapture
                   else list(self.world.scene(shot.scene_id).get("admitted", []))[-3:])
        shot = replace(shot, raw={**shot.raw, "_world_history_ids": history,
                                  "_world_admitted_ids": list(self.world.scene(shot.scene_id).get("admitted", []))})
        write_json_atomic(shot_dir / "scene_state_before.json", event_state if recapture else previous_state)
        write_json_atomic(shot_dir / "binding.json", binding.__dict__)

        # ---- Query: scene / event memory rendered at the requested camera
        render_intent, geometry_video, degraded = None, None, False
        if binding.mode in GEOMETRY_RELATIONS and geometry.enabled:
            self.stage(shot_dir, "geometry_render")
            try:
                render_intent, geometry_video = self._query_memory(shot, shot_dir, binding)
            except Exception as exc:  # noqa: BLE001
                if transient(exc):
                    raise
                self.log(shot, f"memory query unusable, filming without geometry: {safe_error(exc)[:200]}")
                write_json_atomic(shot_dir / "geometry_degraded.json", {"reason": safe_error(exc)})
                degraded, render_intent, geometry_video = True, None, None
        write_json_atomic(shot_dir / "scene_context.json", scene_state.context(shot))

        # ---- opening still and generation prompts
        still, still_kind, pending, author_refs = self._prepare_still(shot, shot_dir, binding, render_intent,
                                                                     geometry_video, degraded)
        use_continuity = still_kind == "continue" and still is not None
        use_first_frame = still_kind in {"gpt_cold", "gpt_anchor", "event_reference"}
        expect_geometry = geometry_video is not None
        packing = dict(output_dir=shot_dir, geometry_video=geometry_video,
                       continuity_still=still if use_continuity else None,
                       first_frame=still if use_first_frame else None,
                       expect_geometry=expect_geometry, expect_continuity=use_continuity,
                       expect_first_frame=use_first_frame,
                       same_event_recapture=recapture is not None and not degraded)
        already = existing_artifact(shot_dir)
        from atas.shot.prompt_author import author_prompts
        self.stage(shot_dir, "prompt_author")
        draft, _, _ = build_request(shot, require_files=False, **packing)
        shot = author_prompts(shot, shot_dir, references=draft.condition.references, geometry_video=geometry_video,
                              scene_reference=author_refs.get("scene"), composition_guide=author_refs.get("guide"),
                              generate_first_frame=pending is not None, intent=render_intent,
                              existing_still=still if (use_continuity or still_kind == "event_reference") else None)
        if pending is not None:
            self.stage(shot_dir, "image_generation")
            pending(shot)
            from PIL import Image
            with Image.open(still) as image:
                image.verify()

        # ---- G: generate the clip
        request, _, _ = build_request(camera_conditioned_shot(shot, render_intent), require_files=True, **packing)
        if already is None:
            self.stage(shot_dir, "video_generation")
            already = await generate_shot(request)
        artifact = shot_dir / "result.mp4"
        if Path(already).resolve() != artifact.resolve():
            shutil.copy2(already, artifact)

        # ---- f: film review with bounded repair; delivery never waits on memory
        self.stage(shot_dir, "film_review")
        shot, film, review, f = await self._review(shot, shot_dir, artifact, render_intent, binding,
                                                   packing, author_refs)

        # ---- r, a: reconstruct, register, validate, then publish W_{t+1}
        r = a = False
        world_version, reason = None, None
        if not f:
            reason = "film_review_failed_best_available_delivered"
        elif not admit or degraded or shot.raw.get("film_only"):
            reason = "film_only_unconditioned"
        elif not geometry.enabled:
            reason = "memory_disabled"
        else:
            self.stage(shot_dir, "reconstruction")
            try:
                r, a, world_version, reason = self._admit(shot, shot_dir, film, review, render_intent, binding)
            except Exception as exc:  # noqa: BLE001
                # The film is already reviewed: a reconstruction/registration
                # failure withholds memory admission (r = 0), it never refilms.
                if transient(exc):
                    raise
                r, a, world_version = False, False, None
                reason = "reconstruction_failed:" + safe_error(exc)[:300]
            if r and a:
                anchor = shot_dir / "cold_start_first_frame.png" if binding.mode == "cold_start" else None
                self.world.admit(shot_id=shot.shot_id, scene_id=shot.scene_id, event_id=shot.event_id,
                                 world_time=float(shot.start), duration=float(shot.duration),
                                 world_version=world_version, video=film,
                                 anchor=anchor if anchor and anchor.is_file() else None,
                                 recapture=recapture is not None,
                                 event_offset=float(shot.raw.get("event_offset", 0.0)))
        # Textual continuity (who is present, last observed positions) follows
        # the delivered film of the same scene; it is prompt context, not W.
        state_after = None
        if recapture is None and (f or (review.get("selection", {}) or {}).get("same_scene") is True):
            state_after = scene_state.advance(shot, previous_state, review, {},
                                              world_version or previous_state.get("world_version"))
            self.scene_states[shot.scene_id] = state_after
            write_json_atomic(shot_dir / "scene_state_after.json", state_after)
        status = "reviewed" if f else "best_available"
        self.stage(shot_dir, "done")
        self.log(shot, f"film={status} f={f} r={r} a={a} {reason or ''}")
        row = self._row(shot, binding, video=film, film_status=status, reason=reason, f=f, r=r, a=a,
                        world_version=world_version, scene_state_after=state_after)
        row["stats"] = shot_stats(shot_dir)
        return row

    # ------------------------------------------------------------------
    def _query_memory(self, shot, shot_dir, binding):
        from atas.shot.camera_planner import plan_camera
        from atas.world.geometry_client import render_chain_status, chain_silently_broke, render_payload
        geometry = self.geometry
        intent = render_payload(
            run_id=self.run_id, story_id=self.plan.story_id, shot_id=shot.shot_id, scene_id=shot.scene_id,
            event_id=shot.event_id, duration=shot.duration, fps=24, camera_move_name=shot.camera_move_name,
            camera_side_degrees=shot.camera_side_degrees, camera_movement_degrees=shot.camera_movement_degrees,
            start_from_shot_id=binding.start_from,
            start_from_role="previous_shot" if binding.mode == "continue" else "previous_shot_in_scene",
            scene_revisit=binding.mode == "revisit",
            geometry_layers="l1+l2" if binding.recapture else "l1",
            shot_size=shot.shot_size, camera_angle=shot.camera_angle, shot_perspective=shot.shot_perspective,
            target_entity_ids=shot.entity_ids, camera_relation=binding.relation,
            recapture=binding.recapture, semantic_camera=True)
        intent = {**intent, "framing_policy": "strict" if binding.recapture else "advisory",
                  "execution_profile": "film_first", "camera_runtime_revision": "film-first-framing-v1"}
        intent = plan_camera(geometry, intent, shot, self.run_dir, shot_dir)
        write_json_atomic(shot_dir / "geometry_render_intent.json", intent)
        intent, job, result = self._render(intent, shot, shot_dir)
        chain = render_chain_status(result)
        write_json_atomic(shot_dir / "geometry_render_result.json", {**result, "chain": chain})
        if binding.mode != "recapture" and chain_silently_broke(chain):
            raise ValueError("camera chain broke: the previous endpoint was unavailable")
        video = geometry.fetch_memory_input({**result, "job_id": job.job_id}, shot_dir / "memory_input.mp4")
        if video is None or not video.is_file():
            raise ConnectionError("geometry render succeeded but its video could not be downloaded")
        from atas.media import fit_upload_limit
        return intent, fit_upload_limit(video)

    def _render(self, intent, shot, shot_dir):
        """Render with at most ``camera_replans`` VLM re-plans on a recoverable camera failure."""
        import hashlib
        from atas.runner.ai_recovery import camera_failure_recoverable
        from atas.shot.camera_planner import plan_camera
        from atas.world.geometry_client import GeometryJob, GeometryJobFailed
        geometry = self.geometry
        state_path = shot_dir / "geometry_render_state.json"
        replans = 0
        while True:
            key = hashlib.sha256(json.dumps(intent, sort_keys=True).encode()).hexdigest()
            saved = json.loads(state_path.read_text()) if state_path.is_file() else {}
            if saved.get("request_key") == key and saved.get("job_id") and saved.get("status") != "failed":
                job = GeometryJob(saved["job_id"], "render", intent)
            else:
                job = geometry.render(intent)
                write_json_atomic(state_path, {"request_key": key, "job_id": job.job_id, "status": "waiting"})
            try:
                result = geometry.wait(job.job_id)
            except GeometryJobFailed as exc:
                write_json_atomic(state_path, {"request_key": key, "job_id": job.job_id, "status": "failed"})
                if not camera_failure_recoverable(exc) or replans >= self.policy.camera_replans:
                    raise
                replans += 1
                feedback = {"failed_job_id": job.job_id, "error": safe_error(exc),
                            "previous_plan": intent.get("camera_plan"),
                            "hint": ("If a subject could not be lifted (no reliable seed depth), it is too small "
                                     "or thin to anchor the camera: ground the larger visible surface it rests on "
                                     "(table, rim, floor) as a static subject and keep the small object as a "
                                     "planned prop.")}
                intent = plan_camera(geometry, {**intent, "camera_execution_feedback": feedback},
                                     shot, self.run_dir, shot_dir)
                write_json_atomic(shot_dir / "geometry_render_intent.json", intent)
                continue
            write_json_atomic(state_path, {"request_key": key, "job_id": job.job_id, "status": "succeeded"})
            return intent, job, result

    def _scene_anchor(self, scene_id: str) -> Path | None:
        anchor = self.world.anchor(scene_id)
        if anchor is not None:
            return anchor
        admitted = self.world.scene(scene_id).get("admitted") or []
        if not admitted:
            return None
        from atas.media import write_event_frame
        source = observation_dir(self.story_dir, admitted[0]) / "result.mp4"
        target = self.story_dir / "anchors" / f"{scene_id}.png"
        if source.is_file() and not target.is_file():
            write_event_frame(source, target, 0.0)
        return target if target.is_file() else None

    def _prepare_still(self, shot, shot_dir, binding, render_intent, geometry_video, degraded):
        """Return (still_path, kind, pending_generator, author_references)."""
        from atas.shot.stills import (generate_anchor_still, generate_cold_start_frame,
                                       generate_geometry_aligned_still, write_composition_guide)
        from atas.media import existing_artifact, write_event_frame, write_last_frame
        kind = binding.still_mode
        if kind == "event_reference":
            target = shot_dir / "event_reference.png"
            source = existing_artifact(observation_dir(self.story_dir, binding.recapture["source_shot_id"]))
            if source is None:
                raise ValueError("event source film is missing")
            write_event_frame(source, target, binding.recapture["source_time_start"])
            return target, kind, None, {}
        if kind == "continue":
            target = shot_dir / "continuity_last_frame.png"
            delivered = (self.world.shots.get(binding.start_from) or {}).get("video")
            source = Path(delivered) if delivered else observation_dir(self.story_dir, binding.start_from) / "result.mp4"
            if source.is_file():
                write_last_frame(source, target)
                return target, kind, None, {}
            kind = "gpt_anchor"
        if kind == "gpt_anchor":
            anchor = self._scene_anchor(shot.scene_id)
            if anchor is not None:
                if render_intent and geometry_video and render_intent.get("composition_contract") == \
                        "projected-subject-regions-v1" and not degraded:
                    target = shot_dir / "geometry_anchor_still.png"
                    report = json.loads((shot_dir / "geometry_render_result.json").read_text())["report"]
                    report = {**report, "scene_actor_regions": [
                        r for r in scene_state.context(shot).get("visibility", {}).get("actors", [])
                        if r["label"] not in shot.entity_ids]}
                    guide = write_composition_guide(geometry_video, report, shot_dir / "composition_guide.png")
                    pending = (lambda current: generate_geometry_aligned_still(
                        current, scene_reference=anchor, composition_guide=guide, output=target))
                    return target, kind, pending, {"scene": anchor, "guide": guide}
                target = shot_dir / "anchor_still.png"
                return (target, kind, lambda current: generate_anchor_still(current, anchor=anchor, output=target),
                        {"scene": anchor})
            kind = "gpt_cold"
        target = shot_dir / "cold_start_first_frame.png"
        return target, "gpt_cold", (lambda current: generate_cold_start_frame(current, target)), {}

    async def _review(self, shot, shot_dir, artifact, render_intent, binding, packing, author_refs):
        """Film review f with bounded repair. Returns (shot, film, review, f)."""
        from atas.review.editorial import select_best_available
        from atas.review.repair import RepairExhausted, promote_reviewed, repair_until_reviewed
        from atas.review.film_review import review_shot
        from atas.media import existing_artifact
        recapture = binding.recapture
        lineage = {"camera_relation": binding.relation,
                   "geometry_layers": "l1+l2" if recapture else ("l1" if packing["geometry_video"] else None),
                   **(recapture or {})}
        source_event = None
        if recapture:
            source = existing_artifact(observation_dir(self.story_dir, recapture["source_shot_id"]))
            source_event = (source, recapture["source_time_start"], recapture["source_time_end"])
        # A film whose reconstruction was already submitted was certified (f = 1) by the
        # review stored with it; a resumed run waits for that reconstruction instead
        # of re-reviewing (and possibly regenerating) a film already in flight.
        stored, state = shot_dir / "film_intent_review.json", shot_dir / "geometry_recon_state.json"
        if stored.is_file() and state.is_file() and artifact.is_file():
            from atas.review.film_review import file_hash
            review = json.loads(stored.read_text())
            if (json.loads(state.read_text()).get("status") in ("submitted", "waiting", "succeeded")
                    and review.get("video_sha256") == file_hash(artifact)
                    and review.get("decision") in ("pass", "warning")):
                return shot, artifact, review, True
        revise = self._video_revision(shot_dir, render_intent, packing, author_refs)
        try:
            shot, reviewed, review = await repair_until_reviewed(
                shot, artifact, shot_dir, stage="film", review=review_shot, revise=revise,
                max_repairs=self.policy.film_repairs,
                review_kwargs={"intent": render_intent, "lineage": lineage, "source_event": source_event},
                context_key={"intent": render_intent, "lineage": lineage})
            f = True
        except RepairExhausted:
            reviewed, review, selection, authored = select_best_available(shot, shot_dir)
            if authored:
                shot = replace(shot, raw={**shot.raw, "_generation_prompts": authored})
            review = {**review, "selection": selection}
            f = False
        film = promote_reviewed(reviewed, shot_dir / "result.mp4", stage="film", shot_dir=shot_dir)
        write_json_atomic(shot_dir / "film_intent_review.json", {k: v for k, v in review.items() if k != "selection"})
        return shot, film, review, f

    def _video_revision(self, shot_dir, render_intent, packing, author_refs):
        from atas.shot.camera_planner import camera_conditioned_shot
        from atas.media import existing_artifact
        from atas.shot.prompt_author import author_prompts
        from atas.providers.wan.generation import build_request, generate_shot
        async def revise(current, attempt, feedback, failed):
            bare = replace(current, raw={k: v for k, v in current.raw.items() if k != "_generation_prompts"})
            attempt_packing = {**packing, "output_dir": attempt}
            draft, _, _ = build_request(camera_conditioned_shot(bare, render_intent), require_files=True,
                                        **attempt_packing)
            still = packing["continuity_still"] or packing["first_frame"]
            current = author_prompts(bare, attempt, references=draft.condition.references,
                                     geometry_video=packing["geometry_video"],
                                     scene_reference=author_refs.get("scene"),
                                     composition_guide=author_refs.get("guide"), generate_first_frame=False,
                                     intent=render_intent, existing_still=still, revision_feedback=feedback,
                                     failed_video=failed, history_root=shot_dir.parent)
            revised, _, _ = build_request(camera_conditioned_shot(current, render_intent), require_files=True,
                                          **attempt_packing)
            candidate = existing_artifact(attempt)
            if candidate is None:
                candidate = await generate_shot(revised)
            return current, candidate
        return revise

    def _admit(self, shot, shot_dir, film, review, render_intent, binding):
        """Reconstruct + register (r) and publish under the state-update check (a)."""
        from atas.world.identity import registry_for_shot
        from atas.review.film_review import file_hash
        from atas.world.keywords import sam_nouns_for_recon
        from atas.runner.publication_recovery import publish_with_rebase
        from atas.runner.recon_ledger import completed_result, ensure_recon_writeback, record_intent
        geometry = self.geometry
        # a (client side): the reviewed bytes are exactly the bytes to reconstruct.
        if review.get("video_sha256") != file_hash(film):
            return False, False, None, "review_does_not_certify_film"
        observed = scene_state.visible_cast(shot, review)
        lineage = review.get("lineage", {})
        current = completed_result(shot_dir)
        if current is None:
            keywords = sam_nouns_for_recon(observed, film, shot_dir)
            record_intent(shot_dir, {"type": "recon", "shot_id": shot.shot_id, "scene_id": shot.scene_id,
                                     "video": str(film), "dynamic_keywords": list(keywords)})
            current, _, _ = ensure_recon_writeback(
                shot_dir,
                submit=lambda: geometry.recon(
                    run_id=self.run_id, story_id=self.plan.story_id, shot_id=shot.shot_id,
                    scene_id=shot.scene_id, event_id=shot.event_id, duration=shot.duration, video=film,
                    dynamic_keywords=keywords, entity_registry=registry_for_shot(observed),
                    publication_mode="reviewed_candidate_v1", review_lineage=lineage,
                    world_time_start=float(shot.start)).job_id,
                wait=geometry.wait)
        fuse = current.get("fuse", {})
        if fuse.get("registered") is not True:
            return False, False, None, ("registration_rejected:" + str(fuse.get("reason", ""))[:200]
                                        + registration_summary(fuse.get("registration_diagnosis")))
        if fuse.get("publication_status") == "committed":
            return True, True, fuse.get("world_version"), None
        candidate = fuse.get("candidate_job_id")
        if not candidate:
            return True, False, None, "no_publishable_candidate"
        payload = {"run_id": self.run_id, "story_id": self.plan.story_id, "shot_id": shot.shot_id,
                   "scene_id": shot.scene_id, "candidate_job_id": candidate, "film_selection": None,
                   "identity_policy": "frame_local_v1",
                   "film_review": {k: review[k] for k in ("schema_version", "stage", "decision", "cache_key",
                                                         "video_sha256", "lineage", "checks", "summary")}}
        try:
            current, publication = publish_with_rebase(geometry, shot_dir, current, payload)
        except RuntimeError as exc:
            if transient(exc):
                raise
            return True, False, None, "state_update_rejected:" + safe_error(exc)[:200]
        write_json_atomic(shot_dir / "world_publication.json", publication)
        return True, True, publication.get("world_version"), None
