"""Client view of the world state W_t = (S_t, E_t).

The GPU service owns the geometry. This ledger records only what the service
has committed for this story: which observations each scene admitted (scene
memory) and which time-indexed event records exist (event memory). Requests
are bound against it, so a shot never depends on an observation that was not
admitted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atas.runner.recon_ledger import write_json_atomic
class WorldLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.scenes: dict[str, dict[str, Any]] = data.get("scenes", {})
        self.events: dict[str, list[dict[str, Any]]] = data.get("events", {})
        self.shots: dict[str, dict[str, Any]] = data.get("shots", {})

    def save(self) -> None:
        write_json_atomic(self.path, {"schema_version": 1, "scenes": self.scenes,
                                      "events": self.events, "shots": self.shots})

    # ---- scene memory -------------------------------------------------
    def scene(self, scene_id: str) -> dict[str, Any]:
        return self.scenes.get(scene_id, {})

    def has_memory(self, scene_id: str) -> bool:
        return bool(self.scene(scene_id).get("admitted"))

    def last_admitted(self, scene_id: str) -> str | None:
        admitted = self.scene(scene_id).get("admitted") or []
        return admitted[-1] if admitted else None

    def anchor(self, scene_id: str) -> Path | None:
        value = self.scene(scene_id).get("anchor")
        return Path(value) if value and Path(value).is_file() else None

    # ---- event memory -------------------------------------------------
    def retrieve_event(self, scene_id: str, event_id: str, start: float, end: float) -> dict[str, Any] | None:
        """Retrieve(E_t, s, e, tau) with tau in event time.

        An event (one narrative action) may be observed by several consecutive
        bounded clips; each record knows its offset inside the event. Returns
        the admitted record covering [start, end) with clip-local times, or None.
        """
        for record in reversed(self.events.get(event_id, [])):
            if record["scene_id"] != scene_id:
                continue
            offset = float(record.get("event_offset", 0.0))
            local_start, local_end = start - offset, end - offset
            if 0 <= local_start < local_end <= record["duration"] + 1e-6:
                return {**record, "local_start": local_start, "local_end": min(local_end, record["duration"])}
        return None

    # ---- state update -------------------------------------------------
    def admit(self, *, shot_id: str, scene_id: str, event_id: str, world_time: float, duration: float,
              world_version: str, video: Path, anchor: Path | None, recapture: bool,
              event_offset: float = 0.0) -> None:
        scene = self.scenes.setdefault(scene_id, {"admitted": []})
        if shot_id not in scene["admitted"]:
            scene["admitted"].append(shot_id)
        scene["world_version"] = world_version
        if anchor is not None and not scene.get("anchor"):
            scene["anchor"] = str(anchor)
        if not recapture and event_id:
            rows = self.events.setdefault(event_id, [])
            if not any(r["shot_id"] == shot_id for r in rows):
                rows.append({"scene_id": scene_id, "shot_id": shot_id, "world_time": world_time,
                             "event_offset": event_offset, "duration": duration, "video": str(video)})
        self.save()

    def record_shot(self, shot_id: str, row: dict[str, Any]) -> None:
        self.shots[shot_id] = row
        self.save()

    def snapshot(self) -> dict[str, Any]:
        """Compact summary given to the compiler before it expands later requests."""
        return {
            "scenes": {sid: {"admitted_shots": s.get("admitted", []),
                             "last_admitted_shot": (s.get("admitted") or [None])[-1]}
                       for sid, s in self.scenes.items()},
            "events": {eid: [{"scene_id": r["scene_id"], "shot_id": r["shot_id"],
                              "event_time": [r.get("event_offset", 0.0), r.get("event_offset", 0.0) + r["duration"]]}
                             for r in rows] for eid, rows in self.events.items()},
            "delivered_shots": {sid: {"scene_id": r.get("scene_id"), "film": r.get("film_status"),
                                      "admitted": r.get("admitted", False)}
                                for sid, r in self.shots.items()},
        }
