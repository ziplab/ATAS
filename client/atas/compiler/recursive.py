"""Recursive world-state compiler (paper Sec. 3.2, Appendix B).

Level 1 (skeleton): the whole screenplay -> cast, physical scenes, narrative
shots (each one event observation with start/end state and visual intent),
and cut/bridge transitions. Planned once.

Level 2 (segments): each narrative shot is expanded only when execution reaches
it, in a sub-conversation that inherits [story, skeleton] plus a summary of the
*current* world state. Each segment is one bounded request q = (d, s, xi, c, tau, e)
that fits the generator budget (4-15 s). The execution loop then binds each
request against admitted memory before rendering.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from atas.providers.llm import Message, chat, strip_fence
from atas.runner.recon_ledger import write_json_atomic
from atas.request import COMPILER_VERSION, MAX_DURATION, MIN_DURATION, StoryPlan, plan_from_payload, request_from_dict
PROMPTS = Path(__file__).resolve().parent / "prompts"
VERSION = COMPILER_VERSION
SHOT_SIZES = {"extreme_close_up", "close_up", "medium_close_up", "medium", "medium_wide", "wide", "extreme_wide"}
MOVES = {"locked", "push_in", "pull_back", "orbit", "arc_left", "arc_right", "push_in_with_arc", "truck_left",
         "truck_right", "crane_up", "crane_down", "pan_left", "pan_right", "tilt_up", "tilt_down", "handheld",
         "track_left", "track_right", "track_forward", "track_back"}
SLUG = re.compile(r"^[A-Za-z0-9_-]+$")


def skeleton_system() -> str:
    return (PROMPTS / "skeleton.md").read_text(encoding="utf-8") + (PROMPTS / "skeleton_world.md").read_text(encoding="utf-8")


def segments_system() -> str:
    return (PROMPTS / "segments.md").read_text(encoding="utf-8") + (PROMPTS / "segments_world.md").read_text(encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


# ---- validation --------------------------------------------------------------
def validate_skeleton(data: Any, fixed_labels: list[str] | None = None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("skeleton must be a JSON object")
    style = data.get("style")
    if not isinstance(style, dict) or not all(isinstance(style.get(k), str) and style[k].strip()
                                              for k in ("medium", "contract")):
        raise ValueError("style needs medium and contract")
    portraits = data.get("portrait_plan")
    if not isinstance(portraits, list) or not portraits:
        raise ValueError("portrait_plan must be a nonempty list")
    ids = [p.get("id") for p in portraits if isinstance(p, dict)]
    if len(ids) != len(portraits) or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("portrait ids must be unique nonempty strings")
    if fixed_labels is not None and set(ids) != set(fixed_labels):
        raise ValueError(f"portrait_plan ids must be exactly {sorted(fixed_labels)}")
    locations = data.get("location_plan")
    if not isinstance(locations, dict) or not locations or not all(SLUG.match(k) for k in locations):
        raise ValueError("location_plan must map slug ids to locations")
    shots = data.get("shots")
    anchors = (data.get("boundarys") or {}).get("boundary_anchors")
    if not isinstance(shots, list) or not shots or not isinstance(anchors, list) or len(anchors) != len(shots):
        raise ValueError("shots and boundarys.boundary_anchors must be nonempty and equal length")
    seen: dict[str, dict] = {}
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict) or not re.match(r"^shot\d{2}_\w+$", str(shot.get("id", ""))):
            raise ValueError(f"shot {index}: id must look like shot01_slug")
        sid = shot["id"]
        if sid in seen:
            raise ValueError(f"duplicate shot id {sid}")
        duration = shot.get("duration_s")
        if type(duration) is not int or not MIN_DURATION <= duration <= 240:
            raise ValueError(f"{sid}: duration_s must be an integer in [4, 240]")
        for key in ("story_goal", "start_state", "end_state", "visual_intent"):
            _text(shot, key)
        if shot.get("location_id") not in locations:
            raise ValueError(f"{sid}: location_id must be a location_plan key")
        characters, focal = shot.get("characters", []), shot.get("focal", [])
        if (not isinstance(characters, list) or not isinstance(focal, list) or not set(focal) <= set(characters)
                or not set(characters) <= set(ids)):
            raise ValueError(f"{sid}: focal must be within characters, characters within portrait ids")
        recap = shot.get("recapture_of")
        if recap is not None:
            source = seen.get(recap.get("shot_id")) if isinstance(recap, dict) else None
            if source is None or source["location_id"] != shot["location_id"]:
                raise ValueError(f"{sid}: recapture_of must name an earlier shot in the same location")
            start, end = recap.get("start_s"), recap.get("end_s")
            if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not 0 <= start < end <= source["duration_s"] or end - start != duration
                    or not MIN_DURATION <= duration <= MAX_DURATION):
                raise ValueError(f"{sid}: recapture interval must lie in the source and equal duration_s (4-15 s)")
        anchor = anchors[index]
        if not isinstance(anchor, dict) or not str(anchor.get("id", "")).startswith("a"):
            raise ValueError(f"{sid}: boundary anchor {index} needs an id like a01_slug_start")
        _text(anchor, "prompt")
        seen[sid] = shot
    transitions = data.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != len(shots) - 1:
        raise ValueError("transitions must have len(shots) - 1 entries")
    for index, tr in enumerate(transitions):
        if not isinstance(tr, dict) or tr.get("mode") not in {"cut", "bridge"}:
            raise ValueError(f"transition {index}: mode must be cut or bridge")
        if tr.get("from_shot") != shots[index]["id"] or tr.get("to_shot") != shots[index + 1]["id"]:
            raise ValueError(f"transition {index} must join adjacent shots in order")
        if tr["mode"] == "bridge":
            _text(tr, "prompt")
            if type(tr.get("duration_s")) is not int or not MIN_DURATION <= tr["duration_s"] <= 8:
                raise ValueError(f"transition {index}: bridge duration_s must be 4-8")
    return data


def validate_segments(data: Any, shot: dict, anchor: dict, portrait_ids: set[str], *, single: bool) -> list[dict]:
    if not isinstance(data, dict) or not data:
        raise ValueError("segments must be a nonempty JSON object keyed by segment id")
    rows = sorted(data.values(), key=lambda r: r.get("segment_index_in_shot", -1) if isinstance(r, dict) else -1)
    if single and len(rows) != 1:
        raise ValueError("this shot is a recapture: return exactly one segment")
    total = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("segment_index_in_shot") != index:
            raise ValueError(f"segment_index_in_shot must run 0..{len(rows) - 1}")
        if not SLUG.match(str(row.get("id", ""))):
            raise ValueError(f"segment {index}: id must be a slug")
        duration = row.get("duration_s")
        if type(duration) is not int or not MIN_DURATION <= duration <= MAX_DURATION:
            raise ValueError(f"segment {index}: duration_s must be an integer in [4, 15]")
        total += duration
        for key in ("prompt", "end_state", "camera_brief", "camera_angle"):
            _text(row, key)
        if row.get("shot_size") not in SHOT_SIZES:
            raise ValueError(f"segment {index}: shot_size must be one of {sorted(SHOT_SIZES)}")
        if row.get("camera_move_name") not in MOVES:
            raise ValueError(f"segment {index}: camera_move_name must be one of {sorted(MOVES)}")
        characters, focal = row.get("characters", []), row.get("focal", [])
        if (not isinstance(characters, list) or not isinstance(focal, list) or not set(focal) <= set(characters)
                or not set(characters) <= portrait_ids):
            raise ValueError(f"segment {index}: focal must be within characters, characters within portrait ids")
        expected = "shot_start_anchor" if index == 0 else "previous_segment_last_frame"
        if row.get("first_frame_path") != expected:
            raise ValueError(f"segment {index}: first_frame_path must be {expected}")
    if total != shot["duration_s"]:
        raise ValueError(f"sum of segment durations {total} != shot duration_s {shot['duration_s']}")
    return rows


# ---- compiler ----------------------------------------------------------------
class RecursiveCompiler:
    def __init__(self, *, story_id: str, story_dir: Path, screenplay: Path, characters: Path | None):
        self.story_id, self.story_dir = story_id, Path(story_dir)
        self.folder = self.story_dir / "planning"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.screenplay = Path(screenplay).read_text(encoding="utf-8-sig")
        self.provided = self._load_characters(characters)
        self._skeleton: dict | None = None
        self._assets: dict[str, str] | None = None
        self.render_assets = True   # False for --plan-only: plan without paid image generation

    @staticmethod
    def _load_characters(path: Path | None) -> dict[str, str]:
        if path is None:
            return {}
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise ValueError("characters manifest must map labels to image paths")
        resolved = {k: str((Path(path).parent / v).resolve()) for k, v in data.items()}
        missing = [v for v in resolved.values() if not Path(v).is_file()]
        if missing:
            raise ValueError(f"character images missing: {missing}")
        return resolved

    # ---- level 1 -------------------------------------------------------------
    def _parent_messages(self) -> list[Message]:
        story = self.screenplay
        if self.provided:
            story += "\n\n[固定角色列表 — portrait_plan 必须且只能使用这些 id]\n" + "\n".join(self.provided)
        return [Message("system", skeleton_system()), Message("user", story)]

    def skeleton(self) -> dict:
        if self._skeleton is not None:
            return self._skeleton
        path = self.folder / "skeleton.json"
        key = _digest({"version": VERSION, "system": skeleton_system(), "story": self.screenplay,
                       "labels": sorted(self.provided)})
        if path.is_file():
            cached = json.loads(path.read_text())
            if cached.get("key") == key:
                self._skeleton = cached["skeleton"]
                return self._skeleton
        labels = sorted(self.provided) if self.provided else None
        skeleton = self._ask(self._parent_messages(), lambda d: validate_skeleton(d, labels),
                             log=self.folder / "skeleton_attempts.json")
        write_json_atomic(path, {"key": key, "skeleton": skeleton})
        self._skeleton = skeleton
        return skeleton

    def _ask(self, messages: list[Message], validate, *, log: Path, attempts: int = 4):
        history, rows, last = list(messages), [], None
        for attempt in range(attempts):
            reply = chat(history)
            try:
                result = validate(json.loads(strip_fence(reply)))
                rows.append({"attempt": attempt, "status": "valid"})
                write_json_atomic(log, {"attempts": rows})
                return result
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                last = exc
                rows.append({"attempt": attempt, "status": "invalid", "error": str(exc)[:2000], "reply": reply[:20000]})
                write_json_atomic(log, {"attempts": rows})
                history = list(messages) + [Message("assistant", reply), Message(
                    "user", f"上一次输出未通过校验: {str(exc)[:3000]}\n请修正后输出完整 JSON。")]
        raise ValueError(f"planner output invalid after {attempts} attempts: {last}")

    def assets(self) -> dict[str, str]:
        """Character appearance references: provided images, else rendered from portrait prompts."""
        if self._assets is not None:
            return self._assets
        skeleton = self.skeleton()
        assets = dict(self.provided)
        missing = [p for p in skeleton["portrait_plan"] if p["id"] not in assets]
        if missing and not self.render_assets:
            return {p["id"]: str(self.story_dir / "cast" / f"{p['id']}.png") for p in skeleton["portrait_plan"]}
        if missing:
            from atas.compiler.cast import render_portraits
            assets.update(render_portraits(missing, skeleton["style"]["contract"], self.story_dir / "cast"))
        self._assets = assets
        return assets

    def _payload(self, shots: list[dict]) -> dict:
        skeleton = self.skeleton()
        style = skeleton["style"]
        return {"schema_version": "atas-story-v1", "story_id": self.story_id,
                "title": str(skeleton.get("title") or self.story_id), "planning_provenance": {"kind": VERSION},
                "style_contract": style["contract"], "visual_style": style,
                "character_reference_map": self.assets(), "scene_graph": skeleton["location_plan"],
                "shots": shots}

    def initial_plan(self) -> StoryPlan:
        return plan_from_payload(self.story_id, self._payload([]), self.folder / "plan.json")

    # ---- level 2 -------------------------------------------------------------
    def _segments(self, index: int, world_summary: dict | None) -> list[dict]:
        skeleton = self.skeleton()
        shot = skeleton["shots"][index]
        anchor = skeleton["boundarys"]["boundary_anchors"][index]
        path = self.folder / f"segments_{shot['id']}.json"
        shot_key = hashlib.sha256(json.dumps([shot, anchor], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if path.is_file():
            cached = json.loads(path.read_text())
            # The first expansion is frozen: resuming must not re-plan executed requests.
            # A re-planned skeleton that changed this shot gets a new expansion.
            if cached.get("shot_key", shot_key) == shot_key:
                return cached["segments"]
            path.rename(path.with_name(f"segments_{shot['id']}.{cached['shot_key'][:12]}.json"))
        portraits = {p["id"] for p in skeleton["portrait_plan"]}
        recap = shot.get("recapture_of")
        user = self._segment_user_prompt(shot, anchor, skeleton, world_summary)
        if recap:
            user += (f"\n[重拍约束]\n本 shot 是对 {recap['shot_id']} 第 {recap['start_s']}–{recap['end_s']} 秒"
                     f"同一事件的换机位重拍: 只输出 1 个 segment, duration_s = {shot['duration_s']},"
                     " 动作与时间顺序必须与原事件一致, 不要续写新动作。")
        messages = [Message("system", segments_system()), Message("user", self._parent_messages()[1].content),
                    Message("assistant", "```json\n" + json.dumps({k: v for k, v in skeleton.items()},
                                                                   ensure_ascii=False, indent=1) + "\n```"),
                    Message("user", user)]
        rows = self._ask(messages, lambda d: validate_segments(d, shot, anchor, portraits, single=bool(recap)),
                         log=self.folder / f"segments_{shot['id']}_attempts.json")
        write_json_atomic(path, {"world_summary": world_summary, "shot_key": shot_key, "segments": rows})
        return rows

    @staticmethod
    def _segment_user_prompt(shot: dict, anchor: dict, skeleton: dict, world: dict | None) -> str:
        portraits = ", ".join(p["id"] for p in skeleton["portrait_plan"])
        places = ", ".join(skeleton["location_plan"])
        body = f"""shot id = {shot["id"]}, duration_s = {shot["duration_s"]}s, location = {shot["location_id"]}

[硬约束]
- ∑ segment.duration_s = {shot["duration_s"]}, 单段 ∈ [4, 15]
- segment 0 first_frame = anchor (id={anchor["id"]})
- segment k≥1 first_frame = previous segment 尾帧

[时间链端点]
- start_state (T=0):    {shot["start_state"]}
- end_state   (T={shot["duration_s"]}s): {shot["end_state"]}

[anchor 第一帧画面]
{anchor["prompt"]}

[镜头运动 + 风格 + 氛围 — 所有 segment 共享]
{shot["visual_intent"]}

[功能/意义 — 背景参考]
{shot["story_goal"]}

[在场角色] {", ".join(shot.get("characters", []))}   [主要拍摄] {", ".join(shot.get("focal", []))}

[可选 reference_inputs ID]
- portrait: {portraits}
- place: {places}
"""
        if world is not None:
            body += "\n[世界状态 — 已写入记忆的观测]\n" + json.dumps(world, ensure_ascii=False)
        return body

    # ---- requests --------------------------------------------------------------
    def _raw_requests(self, index: int, rows: list[dict], world_time: float, first_seen: set[str]) -> list[dict]:
        skeleton = self.skeleton()
        shot, anchor = skeleton["shots"][index], skeleton["boundarys"]["boundary_anchors"][index]
        assets = self.assets()
        previous = skeleton["shots"][index - 1] if index else None
        transition = skeleton["transitions"][index - 1] if index else None
        location = shot["location_id"]
        if location not in first_seen:
            first_relation = "new_scene"
        elif previous and previous["location_id"] == location and transition and transition["mode"] == "cut":
            first_relation = "cut_recompose"
        else:
            first_relation = "revisit"
        sources = {s["recapture_of"]["shot_id"] for s in skeleton["shots"] if s.get("recapture_of")}
        out, offset = [], 0.0
        for row in rows:
            k = row["segment_index_in_shot"]
            relation = first_relation if k == 0 else "continuous"
            raw = {
                "id": row["id"], "location_id": location, "event_id": shot["id"], "event_offset": offset,
                "duration": float(row["duration_s"]), "start": world_time + offset,
                "prompt": row["prompt"], "provider_prompt": row["prompt"],
                "style_contract": skeleton["style"]["contract"], "visual_style": skeleton["style"],
                "shot_size": row["shot_size"], "camera_angle": row["camera_angle"],
                "camera_move_name": row["camera_move_name"], "camera_brief": row["camera_brief"],
                "shot_perspective": row["camera_brief"], "camera_relation": relation,
                "entity_ids": list(row["focal"]), "focal_labels": list(row["focal"]),
                "onstage_labels": list(row["characters"]),
                "character_reference_images": [assets[x] for x in row["focal"]],
                "character_reference_map": {x: assets[x] for x in row["characters"]},
                "memory_reference_shot_ids": [], "state_dependencies": [out[-1]["id"]] if out else [],
                "source_requirements": [{"id": shot["id"], "content": {
                    "story_goal": shot["story_goal"], "start_state": shot["start_state"],
                    "end_state": shot["end_state"], "visual_intent": shot["visual_intent"],
                    "segment": row["prompt"]}}],
                "planned_scene": skeleton["location_plan"][location],
                "staging": anchor["prompt"] if k == 0 else out[-1]["final_state"],
                "initial_state": shot["start_state"] if k == 0 else out[-1]["final_state"],
                "final_state": row["end_state"], "narrative_purpose": shot["story_goal"],
                "negative_prompt": row.get("negative_prompt", ""), "narrative_shot_id": shot["id"],
                "segment_index_in_shot": k, "planning_provenance": {"kind": VERSION},
                "recapture_source": shot["id"] in sources,
            }
            recap = shot.get("recapture_of")
            if recap:
                # A recapture observes the SAME event (paper q.e): each segment covers
                # its own slice of the recorded event time, replayed in real time.
                start = float(recap["start_s"]) + offset
                end = min(float(recap["end_s"]), start + float(row["duration_s"]))
                start, end = self._within_one_source_segment(recap["shot_id"], start, end)
                if end - start >= MIN_DURATION:
                    raw["duration"] = end - start
                    raw.update(camera_relation="recapture", render_mode="recapture",
                               event_id=recap["shot_id"], event_offset=start,
                               source_event_id=recap["shot_id"], source_time_start=start,
                               source_time_end=end)
            out.append(raw)
            offset += float(row["duration_s"])
        first_seen.add(location)
        return out

    def _within_one_source_segment(self, source_shot_id: str, start: float, end: float) -> tuple[float, float]:
        """Event memory holds one record per generated clip, so a recapture interval
        must lie inside one source segment: shift it into the segment it overlaps
        most, shortening it only when that segment is shorter than the interval."""
        path = self.folder / f"segments_{source_shot_id}.json"
        if not path.is_file():
            return start, end
        bounds, offset = [], 0.0
        for row in json.loads(path.read_text())["segments"]:
            bounds.append((offset, offset + float(row["duration_s"])))
            offset += float(row["duration_s"])
        if any(a <= start and end <= b for a, b in bounds):
            return start, end
        a, b = max(bounds, key=lambda seg: min(end, seg[1]) - max(start, seg[0]))
        length = min(end - start, b - a)
        start = min(max(start, a), b - length)
        return start, start + length

    def _bridge(self, index: int, world_time: float) -> dict:
        """A cross-space/time transition clip: delivered as film, never written to memory.

        It carries the focal cast of the two shots it joins, so the people it shows keep
        their identity references and the film review knows who may appear."""
        skeleton = self.skeleton()
        transition = skeleton["transitions"][index - 1]
        cast = list(dict.fromkeys([*skeleton["shots"][index - 1].get("focal", []),
                                   *skeleton["shots"][index].get("focal", [])]))
        assets = self.assets()
        return {"id": "bridge_" + re.sub(r"[^A-Za-z0-9_-]", "_", transition["id"]),
                "location_id": "bridge_" + re.sub(r"[^A-Za-z0-9_-]", "_", transition["id"]),
                "event_id": "bridge_" + transition["id"], "duration": float(transition["duration_s"]),
                "start": world_time, "prompt": transition["prompt"], "provider_prompt": transition["prompt"],
                "style_contract": skeleton["style"]["contract"], "visual_style": skeleton["style"],
                "shot_size": "wide", "camera_angle": "eye_level", "camera_move_name": "push_in",
                "camera_brief": transition["prompt"], "shot_perspective": transition["prompt"],
                "camera_relation": "new_scene", "entity_ids": cast, "focal_labels": cast, "onstage_labels": cast,
                "character_reference_images": [assets[x] for x in cast if x in assets],
                "character_reference_map": {x: assets[x] for x in cast if x in assets},
                "source_requirements": [{"id": transition["id"], "content": transition["prompt"]}],
                "film_only": True, "planning_provenance": {"kind": VERSION}}

    def requests(self, runner, limit=None) -> Iterator:
        """Yield bounded requests in order; each narrative shot is expanded against
        the world state reached after the previous requests were executed."""
        skeleton = self.skeleton()
        runner.references.update(self.assets())
        world_time, first_seen, emitted = 0.0, set(), 0
        for index, shot in enumerate(skeleton["shots"]):
            batch = []
            if index and skeleton["transitions"][index - 1]["mode"] == "bridge":
                batch.append(self._bridge(index, world_time))
                world_time += batch[-1]["duration"]
            rows = self._segments(index, runner.world.snapshot())
            batch += self._raw_requests(index, rows, world_time, first_seen)
            world_time += float(shot["duration_s"])
            for raw in batch:
                if limit is not None and emitted >= limit:
                    return
                request = request_from_dict(self.story_id, raw)
                runner.plan = replace(runner.plan, shots=runner.plan.shots + (request,))
                write_json_atomic(self.folder / "plan.json",
                                  self._payload([s.raw for s in runner.plan.shots]))
                emitted += 1
                yield request

    def compile_all(self) -> StoryPlan:
        """Expand every shot without a world state (inspection / --plan-only)."""
        skeleton = self.skeleton()
        raws, world_time, first_seen = [], 0.0, set()
        for index, shot in enumerate(skeleton["shots"]):
            if index and skeleton["transitions"][index - 1]["mode"] == "bridge":
                raws.append(self._bridge(index, world_time))
                world_time += raws[-1]["duration"]
            raws += self._raw_requests(index, self._segments(index, None), world_time, first_seen)
            world_time += float(shot["duration_s"])
        payload = self._payload(raws)
        write_json_atomic(self.folder / "plan.json", payload)
        return plan_from_payload(self.story_id, payload, self.folder / "plan.json")
