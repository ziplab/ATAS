"""One atomic commit pointer publishes L1 and registered RGB-D together."""
from __future__ import annotations
import json
from pathlib import Path
from atas_geometry.service.durable import atomic_json, file_hash
def load_world(scene):
    path=Path(scene)/"world_state.json"
    return json.loads(path.read_text()) if path.is_file() else None


def checked_path(scene, relative):
    p=(Path(scene)/relative).resolve()
    # Versioned world artefacts must live inside this scene directory.
    p.relative_to(Path(scene).resolve())
    if not p.exists(): raise FileNotFoundError(p)
    return p


def current_l1(scene):
    state=load_world(scene)
    return checked_path(scene,state["l1"]) if state else Path(scene)/"l1.npz"


def committed_observation(story, shot):
    matches=[]
    for manifest in (Path(story)/"scenes").glob("*/world_state.json"):
        state=json.loads(manifest.read_text())
        record=state.get("observations",{}).get(shot)
        if record: matches.append(checked_path(manifest.parent,record["l0"]))
    if len(matches)>1: raise ValueError("shot registered in multiple scenes")
    return matches[0] if matches else None


def commit_world(scene, *, version, l1, observation, shot_id, source_sha256, expected_version):
    scene=Path(scene); old=load_world(scene)
    if old and shot_id in old.get("observations",{}):
        record=old["observations"][shot_id]
        if record["source_sha256"] != source_sha256:
            raise ValueError("accepted shot id cannot be replaced by a different video")
        return old, False
    if (old or {}).get("version") != expected_version:
        raise RuntimeError("world changed before commit; reconstruct against latest revision")
    meta=json.loads((Path(observation)/"observation.json").read_text())
    if meta.get("registered") is not True: raise ValueError("cannot commit unregistered geometry")
    relative_l1=str(Path(l1).resolve().relative_to(scene.resolve()))
    relative_l0=str(Path(observation).resolve().relative_to(scene.resolve()))
    records=dict((old or {}).get("observations",{}))
    records[shot_id]={"l0":relative_l0,"source_sha256":source_sha256,"version":version}
    state={"schema_version":1,"version":version,"parent_version":expected_version,
           "l1":relative_l1,"l1_sha256":file_hash(l1),"observations":records}
    # Version manifest first, then the single authoritative pointer. A crash
    # before pointer replacement leaves the old world completely readable.
    atomic_json(scene/".world_versions"/version/"commit.json",state)
    atomic_json(scene/"world_state.json",state)
    return state, True
