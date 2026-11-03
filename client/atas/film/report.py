"""Run report: the paper's process measures, computed from a story manifest.

Reference values (Wan 3.0, full state) are the paper's Fig. 6 and Tab. H; they
describe the reported configuration and are shown for orientation, not as targets
a single story must hit.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from atas.runner.recon_ledger import write_json_atomic
PAPER = {"film_rejected_first_review": 0.368, "retries_per_shot": 0.54, "state_update_rate": 0.674,
         "minutes_per_story": 64.9, "memory_overhead_s_per_shot": 98.0}


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    shots = [r for r in manifest["shots"] if not r.get("binding_notes") or "film_only_transition" not in r["binding_notes"]]
    stats = [r.get("stats") or {} for r in shots]
    reviewed = [r for r in shots if r.get("film_status") in {"reviewed", "best_available"}]
    memory_stages = ("geometry_render", "reconstruction")
    by_stage: dict[str, int] = {}
    for s in stats:
        for stage, mib in (s.get("gpu_reserved_peak_mib") or {}).items():
            by_stage[stage] = max(by_stage.get(stage, 0), mib)
    wall = [s.get("wall_seconds") for s in stats]
    return {
        "shots": len(shots), "delivered": sum(bool(r.get("video")) for r in shots),
        "complete": manifest.get("complete"), "final_video": manifest.get("final_video"),
        "film_status": {k: sum(r.get("film_status") == k for r in shots)
                        for k in ("reviewed", "best_available", "degraded", "missing")},
        "f_rate": _mean([float(bool(r.get("f"))) for r in shots]),
        "r_rate_given_f": _mean([float(bool(r.get("r"))) for r in shots if r.get("f")]),
        "state_update_rate": _mean([float(bool(r.get("admitted"))) for r in shots]),
        "film_rejected_first_review": _mean([float(bool(s.get("first_review_held"))) for s in stats]) if reviewed else None,
        "retries_per_shot": _mean([s.get("film_regenerations", 0) for s in stats]),
        "minutes_total": sum(w for w in wall if w) / 60 if any(wall) else None,
        "seconds_per_shot": _mean(wall),
        "memory_overhead_s_per_shot": _mean([sum((s.get("stage_seconds") or {}).get(k, 0.0) for k in memory_stages)
                                             for s in stats]),
        "gpu_reserved_peak_mib": {"overall": max(by_stage.values()) if by_stage else None, "by_stage": by_stage},
        "modes": {m: sum(r.get("mode") == m for r in shots) for m in ("cold_start", "continue", "revisit", "recapture")},
        "not_admitted_reasons": [{"shot_id": r["shot_id"], "reason": r.get("reason")} for r in shots
                                 if not r.get("admitted")],
        "paper_reference": PAPER,
    }


def write_report(story_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = summarize(manifest)
    write_json_atomic(Path(story_dir) / "report.json", summary)
    fmt = lambda v, p="{:.2f}": "n/a" if v is None else p.format(v)
    lines = [f"# Run report: {manifest.get('title') or Path(story_dir).name}", "",
             f"- final film: `{summary['final_video']}` (complete: {summary['complete']})",
             f"- shots delivered: {summary['delivered']}/{summary['shots']}  film status: {summary['film_status']}",
             f"- binding modes: {summary['modes']}", "",
             "| measure | this run | paper (Wan 3.0, full state) |", "|---|---|---|",
             f"| film rejected at first review | {fmt(summary['film_rejected_first_review'])} | {PAPER['film_rejected_first_review']} |",
             f"| regenerations per shot | {fmt(summary['retries_per_shot'])} | {PAPER['retries_per_shot']} |",
             f"| state update rate (f=r=a=1) | {fmt(summary['state_update_rate'])} | {PAPER['state_update_rate']} |",
             f"| registration rate given f | {fmt(summary['r_rate_given_f'])} | — |",
             f"| wall time per story (min) | {fmt(summary['minutes_total'], '{:.1f}')} | {PAPER['minutes_per_story']} |",
             f"| memory overhead (s/shot) | {fmt(summary['memory_overhead_s_per_shot'], '{:.0f}')} | {PAPER['memory_overhead_s_per_shot']} |",
             "", f"GPU memory, peak reserved by our processes per stage (MiB): {summary['gpu_reserved_peak_mib']}", "",
             "Not admitted to world state:"] + [f"- {r['shot_id']}: {r['reason']}" for r in summary["not_admitted_reasons"]]
    (Path(story_dir) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
