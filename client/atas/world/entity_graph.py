from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EntityGraph:
    nodes: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]

    def node_ids(self) -> set[str]:
        return {
            str(item.get("node_id") or "").strip()
            for item in self.nodes
            if str(item.get("node_id") or "").strip()
        }

    def closure_errors(self) -> tuple[str, ...]:
        known = self.node_ids()
        errors: list[str] = []
        for item in self.relations:
            relation_id = str(item.get("relation_id") or "<missing>")
            for key in ("subject_id", "object_id", "subject", "object"):
                raw = item.get(key)
                if raw is None or str(raw).strip() == "":
                    continue
                node_id = str(raw).strip()
                if node_id not in known:
                    errors.append(
                        f"{relation_id}: {key} {node_id!r} is not declared in nodes"
                    )
        return tuple(errors)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "type": "avwm_entity_scene_graph",
            "nodes": list(self.nodes),
            "relations": list(self.relations),
            "closure_errors": list(self.closure_errors()),
        }
