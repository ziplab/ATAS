"""Commit-driven incremental mirror; publish each archive pointer last."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from atas_geometry.service.durable import atomic_json
from atas_geometry.world.world_state import checked_path
def _copy_file(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    temporary = target.with_name(target.name + ".archive-partial")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as copied:
        os.fsync(copied.fileno())
    os.replace(temporary, target)


def _copy_tree(source, target):
    # Materialize symlinked raw RGB/masks, never create an archive dependent
    # on /var/tmp. Only source paths selected by a committed manifest are used.
    for parent, dirs, files in os.walk(source, followlinks=True):
        rel = Path(parent).relative_to(source)
        for name in files:
            _copy_file(Path(parent) / name, Path(target) / rel / name)


def archive_story(hot_root, archive_root, run_id, story_id):
    # Allow only one archive at a time so stories do not compete for storage bandwidth.
    destination = Path(archive_root)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".archive-global.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _archive_story(hot_root, archive_root, run_id, story_id)


def _archive_story(hot_root, archive_root, run_id, story_id):
    source = Path(hot_root) / run_id / story_id
    destination = Path(archive_root) / run_id / story_id
    receipt = source / "archive_status.json"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / ".archive.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atomic_json(receipt, {"status": "copying", "updated_at": time.time(),
                                  "archive_root": str(destination)})
            revisions = {}
            for pointer in sorted((source / "scenes").glob("*/world_state.json")):
                state = json.loads(pointer.read_text())
                scene, target = pointer.parent, destination / "scenes" / pointer.parent.name
                _copy_file(checked_path(scene, state["l1"]), target / state["l1"])
                for shot, row in state["observations"].items():
                    observation = checked_path(scene, row["l0"])
                    _copy_tree(observation, target / row["l0"])
                    commit = scene / ".world_versions" / row["version"] / "commit.json"
                    if commit.exists():
                        history = json.loads(commit.read_text())
                        _copy_file(checked_path(scene, history["l1"]), target / history["l1"])
                        _copy_file(commit, target / ".world_versions" / row["version"] / "commit.json")
                marker = scene / "home_shot_id.txt"
                if marker.exists():
                    _copy_file(marker, target / marker.name)
                # A reader sees only old-complete or new-complete scene data.
                atomic_json(target / "world_state.json", state)
                revisions[scene.name] = state["version"]
            atomic_json(receipt, {"status": "complete", "updated_at": time.time(),
                                  "archive_root": str(destination), "scene_versions": revisions})
    except Exception as exc:
        atomic_json(receipt, {"status": "retry_pending", "updated_at": time.time(),
                              "error": f"{type(exc).__name__}: {exc}"})


def reconcile_archives(hot_root, archive_root):
    # Do not scan/copy every historical run on service startup. Only durable
    # interrupted archive tasks resume; new commits repair legacy copies.
    for receipt in Path(hot_root).glob("*/*/archive_status.json"):
        try:
            pending = json.loads(receipt.read_text()).get("status") in {"queued", "copying", "retry_pending"}
        except (OSError, ValueError):
            continue
        if pending:
            archive_story(hot_root, archive_root, receipt.parent.parent.name, receipt.parent.name)
