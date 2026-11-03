"""Bounded shot requests q = (d, s, xi, c, tau, e) and the story they belong to.

    d   description          ``prompt`` (+ ``provider_prompt``, source requirements in ``raw``)
    s   scene                ``scene_id``
    xi  spatial anchor       grounded later by the visual camera planner (``camera_plan.subjects``)
    c   target camera        ``camera_relation``, ``shot_size``, ``camera_angle``, ``camera_move_name``,
                             ``shot_perspective`` (the camera brief)
    tau time interval        ``[start, start + duration)`` in story time; recaptures also carry
                             the source interval ``source_time_start/end`` of event ``e``
    e   event                ``event_id`` (and ``source_event_id`` for a historical recapture)
"""
from __future__ import annotations

import re

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPILER_VERSION = "atas-recursive-compiler-v1"   # planning_provenance.kind of compiled shots
MIN_DURATION, MAX_DURATION = 4.0, 15.0   # Wan 3.0 output and point-cloud reference budget


@dataclass(frozen=True)
class ShotRequest:
    story_id: str
    shot_id: str
    scene_id: str
    duration: float
    start: float
    prompt: str
    provider_prompt: str
    video_edit_prompt: str
    style_contract: str
    camera_move_name: str
    camera_side_degrees: float
    camera_movement_degrees: float
    shot_size: str
    camera_angle: str
    shot_perspective: str
    event_id: str
    camera_path: tuple[dict[str, Any], ...]
    scene_revisit: bool
    camera_relation: str
    still_mode: str
    memory_reference_shot_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    character_reference_images: tuple[Path, ...]
    reference_image: Path | None
    seed: int
    raw: dict[str, Any]

    @property
    def time_interval(self) -> tuple[float, float]:
        return (self.start, self.start + self.duration)

    @property
    def is_recapture(self) -> bool:
        return self.camera_relation == "recapture" or self.raw.get("render_mode") == "recapture"


PlanShot = ShotRequest   # name used by the generation/review modules


@dataclass(frozen=True)
class StoryPlan:
    story_id: str
    path: Path
    title: str
    shots: tuple[ShotRequest, ...]
    payload: dict[str, Any]

    def shot(self, shot_id: str) -> ShotRequest:
        for item in self.shots:
            if item.shot_id == shot_id:
                return item
        raise KeyError(shot_id)


def _paths(value: object) -> tuple[Path, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(Path(str(item)) for item in value if str(item).strip())


def request_from_dict(story_id: str, raw: dict[str, Any], *, sam_keywords=None) -> ShotRequest:
    raw = dict(raw)
    if raw.get("sam_keywords") is None and isinstance(sam_keywords, list):
        raw["sam_keywords"] = sam_keywords
    reference = str(raw.get("reference_image") or "").strip()
    return ShotRequest(
        story_id=story_id,
        shot_id=str(raw.get("id") or ""),
        scene_id=str(raw.get("location_id") or ""),
        duration=float(raw.get("duration") or 0),
        start=float(raw.get("start") or 0),
        prompt=str(raw.get("prompt") or ""),
        provider_prompt=str(raw.get("provider_prompt") or raw.get("prompt") or ""),
        video_edit_prompt=str(raw.get("video_edit_prompt") or ""),
        style_contract=str(raw.get("style_contract") or ""),
        camera_move_name=str(raw.get("camera_move_name") or ""),
        camera_side_degrees=float(raw.get("camera_side_degrees") or 0.0),
        camera_movement_degrees=float(raw.get("camera_movement_degrees") or 0.0),
        shot_size=str(raw.get("shot_size") or ""),
        camera_angle=str(raw.get("camera_angle") or ""),
        shot_perspective=str(raw.get("shot_perspective") or ""),
        event_id=str(raw.get("event_id") or ""),
        camera_path=tuple(item for item in (raw.get("camera_path") or []) if isinstance(item, dict)),
        scene_revisit=bool(raw.get("scene_revisit")),
        camera_relation=str(raw.get("camera_relation") or "").strip(),
        still_mode=str(raw.get("still_mode") or ""),
        memory_reference_shot_ids=tuple(str(x) for x in (raw.get("memory_reference_shot_ids") or [])),
        entity_ids=tuple(str(x) for x in (raw.get("entity_ids") or [])),
        character_reference_images=_paths(raw.get("character_reference_images")),
        reference_image=Path(reference) if reference else None,
        seed=int(raw.get("seed") or 0),
        raw=raw,
    )


def plan_from_payload(story_id: str, payload: dict[str, Any], path: Path) -> StoryPlan:
    shots = tuple(request_from_dict(story_id, raw, sam_keywords=payload.get("sam_keywords"))
                  for raw in payload.get("shots") or [])
    return StoryPlan(story_id=story_id, path=Path(path), title=str(payload.get("title") or story_id),
                     shots=shots, payload=payload)


def load_plan(path: Path) -> StoryPlan:
    """Read a compiled plan (JSON; YAML is accepted when PyYAML is installed)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        payload = yaml.safe_load(text) or {}
    return plan_from_payload(str(payload.get("story_id") or path.stem), payload, path)


def identifier(value):
    """Run, story, scene and shot ids become path components and API fields."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('identifiers must use letters, digits, underscore or hyphen')
    return value


def planning_context(shot):
    """Pass immutable source requirements separately from AI interpretation."""
    raw = getattr(shot, 'raw', {})
    return {k: raw[k] for k in ('source_requirements', 'planned_scene', 'onstage_labels', 'staging',
            'initial_state', 'final_state', 'narrative_purpose', 'camera_brief', 'state_dependencies') if k in raw}
