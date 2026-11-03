"""Command line: screenplay in, long video out.

    python -m atas.cli run --screenplay story.txt --story-id demo [--characters chars.json]
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from pathlib import Path

from atas.config import ROOT, configure
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atas", description="ATAS: screenplay to long video")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compile a screenplay and render the whole film")
    run.add_argument("--screenplay", type=Path, required=True, help="text/Markdown screenplay")
    run.add_argument("--story-id", required=True)
    run.add_argument("--characters", type=Path, help="JSON: exact character label -> appearance image path")
    run.add_argument("--run-id", default="default")
    run.add_argument("--output", type=Path, default=ROOT / "runs")
    run.add_argument("--memory", choices=("full", "none"), default="full",
                     help="'none' disables scene/event memory (ablation)")
    run.add_argument("--max-shots", type=int, default=None)
    run.add_argument("--plan-only", action="store_true", help="compile requests without generating media")
    return parser


def main(argv=None) -> None:
    configure()
    args = build_parser().parse_args(argv)
    from atas.request import identifier
    identifier(args.story_id)
    identifier(args.run_id)
    run_dir = args.output / args.run_id
    story_dir = run_dir / f"story_{args.story_id}"
    story_dir.mkdir(parents=True, exist_ok=True)
    with (story_dir / "story.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        from atas.compiler import RecursiveCompiler
        compiler = RecursiveCompiler(story_id=args.story_id, story_dir=story_dir,
                                     screenplay=args.screenplay, characters=args.characters)
        if args.plan_only:
            compiler.render_assets = False
            plan = compiler.compile_all()
            print(json.dumps({"story_id": args.story_id, "shots": len(plan.shots),
                              "plan": str(plan.path)}, ensure_ascii=False))
            return
        from atas.runner.pipeline import Policy, StoryRunner
        runner = StoryRunner(compiler.initial_plan(), run_id=args.run_id, run_dir=run_dir,
                             policy=Policy(memory=args.memory))
        manifest = asyncio.run(runner.run(compiler.requests(runner, limit=args.max_shots)))
        print(json.dumps({"final_video": manifest["final_video"], "manifest": str(story_dir / "manifest.json"),
                          "delivered": manifest["shots_delivered"], "total": manifest["shots_total"],
                          "admitted": manifest["admitted_to_world"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
