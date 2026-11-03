"""Explicit, restart-safe publication of reviewed candidates; no model calls."""
import json
from atas_geometry.world.recapture import identifier
from atas_geometry.service.paths import scene_dir, story_root, remember_home_shot
from atas_geometry.world.world_state import checked_path, commit_world, load_world
from atas_geometry.service.durable import atomic_json
def candidate_source(root,payload):
    ids={k:identifier(payload.get(k),k) for k in ("run_id","story_id","scene_id","shot_id","candidate_job_id")}
    scene=scene_dir(root,ids["run_id"],ids["story_id"],ids["scene_id"])
    manifest=checked_path(scene,".world_versions/"+ids["candidate_job_id"]+"/candidate.json")
    data=json.loads(manifest.read_text())
    for key in ("run_id","story_id","scene_id","shot_id"):
        if data[key]!=ids[key]: raise ValueError("candidate scope mismatch")
    if data["job_id"]!=ids["candidate_job_id"] or data.get("geometry_accepted") is not True:
        raise ValueError("candidate geometry not accepted")
    return scene,data,checked_path(scene,data["l0"])


def publish_candidate(root,payload):
    scene,data,source=candidate_source(root,payload)
    review=payload.get("film_review",{})
    selection=payload.get('film_selection') or {}
    choices=selection.get('candidates',[]);index=selection.get('selected_index')
    selected=(selection.get('schema')=='best_available_v1' and selection.get('cache_key') and
        selection.get('budget_exhausted') is True and selection.get('audit_passed') is False and
        selection.get('same_scene') is True and selection.get('video_sha256')==data['source_sha256'] and
        selection.get('original_decision')==review.get('decision') and
        selection.get('review_cache_key')==review.get('cache_key') and selection.get('actual_event') and
        1<=len(choices)<=5 and selection.get('candidate_count')==len(choices) and
        selection.get('repair_budget')==len(choices)-1 and type(index) is int and 0<=index<len(choices) and
        choices[index].get('sha256')==data['source_sha256'] and all(
            choices[index].get('review',{}).get(k)==v for k,v in review.items()))
    if (review.get("schema_version")!=1 or review.get("stage")!="film"
        or (review.get("decision") not in {"pass","warning"} and not selected)
        or review.get("video_sha256")!=data["source_sha256"]
        or not review.get("cache_key")):
        raise ValueError("publication requires a content-bound film intent review")
    if review.get("lineage",{}) != data.get("lineage",{}):
        raise ValueError("review and reconstruction lineage differ")
    meta=json.loads((source/"observation.json").read_text())
    shot_dir=story_root(root,data["run_id"],data["story_id"])/"shots"/data["shot_id"]
    old=load_world(scene)
    existing=(old or {}).get("observations",{}).get(data["shot_id"])
    if existing and existing["version"]!=data["job_id"]:
        raise ValueError("shot already published by another candidate")
    # No persistent role-to-track certification is required. Frame/time-local
    # SAM masks and registered depths remain unchanged. Scene lock held by caller.
    if not existing and (old or {}).get("version")!=data["parent_version"]:
        raise RuntimeError("world changed before publication; candidate must be rebuilt")
    atomic_json(scene/".world_versions"/data["job_id"]/"publication_review.json",review)
    if not existing:
        meta.update(identity_policy='frame_local_v1',film_audit_decision=review['decision'],
            scripted_event_verified=review['decision'] in {'pass','warning'} and all(
                any(c.get('criterion')==name and c.get('status')=='pass' for c in review.get('checks',[]))
                for name in ('action','event')),
            actual_event=selection.get('actual_event'),film_selection_key=selection.get('cache_key'))
        atomic_json(source/'observation.json',meta)
    if selection:atomic_json(scene/'.world_versions'/data['job_id']/'film_selection.json',selection)
    world,published=commit_world(scene,version=data["job_id"],l1=checked_path(scene,data["l1"]),
        observation=source,shot_id=data["shot_id"],source_sha256=data["source_sha256"],expected_version=data["parent_version"])
    if data["home"]:
        remember_home_shot(root,data["run_id"],data["story_id"],data["scene_id"],data["shot_id"])
    from atas_geometry.world.memory_manifest import observation_manifest
    return {"publication_status":"committed","world_version":world["version"],
        "candidate_job_id":data["job_id"],"commit_published":published,"novel_points":data["novel_points"],
        "world_manifest_path":str(scene/"world_state.json"),
        "memory_manifest":observation_manifest(source,checked_path(scene,world["l1"]),payload=data,
            world_version=world["version"],publication_status="committed",candidate_job_id=data["job_id"])}
