"""Content-addressed raw observations for unaccepted film revisions."""
import json,re,hashlib,time
from pathlib import Path
from atas_geometry.service.paths import raw_l0_dir, story_root
from atas_geometry.world.world_state import committed_observation
from atas_geometry.service.durable import file_hash
def reconstruction_contract(settings, payload):
    fields = ("pi3_model_id", "sam3_checkpoint", "recon_width", "recon_height",
              "pi3_pixel_limit", "pi3_vo_chunk_size", "pi3_vo_overlap", "low_vram")
    contract = {k: str(getattr(settings, k)) for k in fields}
    contract.update(schema="dense_rgbd_and_seg_v2", frame_policy="all_original_frames",
                    dynamic_keywords=list(payload.get("dynamic_keywords") or []))
    for name in ("pi3_model_id", "sam3_checkpoint"):
        path = Path(getattr(settings, name))
        if path.exists():
            stat = path.stat()
            contract[name + "_stamp"] = [stat.st_size, stat.st_mtime_ns]
            if path.is_dir():
                # Replacing weights in place does not update their parent's
                # directory mtime. Include files, not merely the model folder.
                contract[name + "_files"] = [[str(p.relative_to(path)), p.stat().st_size,
                    p.stat().st_mtime_ns] for p in sorted(path.rglob("*")) if p.is_file()]
    return contract


def contract_digest(payload):
    return hashlib.sha256(json.dumps([payload["video_sha256"], payload["reconstruction_contract"]],
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dense_manifest(l0, payload, expected_frames, *, write=False):
    """A camera file is not a completed dense observation."""
    import numpy as np
    from atas_geometry.service.durable import atomic_json
    l0 = Path(l0)
    contract = payload.get("reconstruction_contract")
    if not contract:
        return False
    receipt = l0 / "dense_complete.json"
    if not write and not receipt.is_file():
        return False
    try:
        with np.load(l0 / "cameras.npz") as cameras:
            counts = {"cam_c2w": len(cameras["cam_c2w"]), "intrinsics": len(cameras["intrinsics"])}
        for name, suffix in (("depths", "*.exr"), ("dynamic_mask", "*.png"), ("sky_mask", "*.png")):
            files = sorted((l0 / name).glob(suffix))
            counts[name] = len(files)
            if any(p.stat().st_size == 0 for p in files):
                return False
        if set(counts.values()) != {expected_frames} or not (l0 / "video.mp4").is_file():
            return False
        tracks = json.loads((l0 / "instance_tracks.json").read_text())
        if tracks["frame_count"] != expected_frames:
            return False
        for track in tracks["tracks"].values():
            directory = (l0 / track["mask_dir"]).resolve()
            directory.relative_to(l0.resolve())
            if len(list(directory.glob("*.png"))) != expected_frames:
                return False
        manifest = {"schema": "dense_complete_v1", "video_sha256": payload["video_sha256"],
                    "contract_digest": contract_digest(payload), "frame_count": expected_frames,
                    "reconstruction_contract": contract, "counts": counts}
        if write:
            atomic_json(receipt, manifest)
            return True
        return json.loads(receipt.read_text()) == manifest
    except (OSError, ValueError, KeyError):
        return False


def quarantine_incomplete(l0):
    """Caller owns the job/scene lease. Never replace published observations."""
    l0 = Path(l0)
    if l0.exists():
        target = l0.with_name("l0.interrupted." + str(time.time_ns()))
        l0.rename(target)
        return target


def accepted_reconstruction(root, payload):
    """Publication wins over a changed cache contract; never rewrite its source."""
    accepted = committed_observation(story_root(root, payload["run_id"], payload["story_id"]),
                                     payload["shot_id"])
    if accepted is None:
        return None
    meta = json.loads((accepted / "observation.json").read_text())
    if meta.get("original_video_sha256") != payload["video_sha256"]:
        raise ValueError("accepted shot content is immutable; cannot regenerate in place")
    if meta.get("scene_id") != payload["scene_id"]:
        raise ValueError("accepted observation belongs to a different scene")
    if meta.get("event_id") != payload.get("event_id"):
        raise ValueError("accepted observation event_id is immutable; use its original event scope")
    return accepted


def accepted_provenance(root, payload, accepted):
    from atas_geometry.service.paths import scene_dir
    from atas_geometry.world.world_state import load_world
    scene = scene_dir(root, payload["run_id"], payload["story_id"], payload["scene_id"])
    row = load_world(scene)["observations"][payload["shot_id"]]
    folder = scene / ".world_versions" / row["version"]
    candidate = folder / "candidate.json"
    review = folder / "publication_review.json"
    meta = json.loads((accepted / "observation.json").read_text())
    data = json.loads(candidate.read_text()) if candidate.is_file() else {}
    audit = json.loads(review.read_text()) if review.is_file() else {}
    return {"original_candidate_job_id": row["version"], "original_event_id": meta.get("event_id"),
            "original_review_lineage": data.get("lineage", audit.get("lineage", meta.get("source_review_lineage"))),
            "original_video_sha256": row["source_sha256"]}

def revision_l0(root,payload):
    run,story,shot=(str(payload[k]) for k in ('run_id','story_id','shot_id'))
    digest=payload['video_sha256']
    if not re.fullmatch(r'[0-9a-f]{64}',digest):raise ValueError('invalid video content hash')
    accepted=committed_observation(story_root(root,run,story),shot)
    if accepted:
        meta=json.loads((accepted/'observation.json').read_text())
        if meta.get('original_video_sha256')!=digest:
            raise ValueError('accepted shot content is immutable; cannot regenerate in place')
    legacy=raw_l0_dir(root,run,story,shot)
    if payload.get("reconstruction_contract"):
        return legacy.parent/'recon_versions'/contract_digest(payload)/'l0'
    if (legacy/'cameras.npz').is_file():
        meta=json.loads((legacy/'observation.json').read_text()) if (legacy/'observation.json').is_file() else {}
        old=meta.get('original_video_sha256')
        if not old and (legacy/'video.mp4').is_file():old=file_hash(legacy/'video.mp4')
        if old==digest:return legacy
    # Old derived candidates resolve their source symlinks into the unchanged
    # old raw directory. A regenerated movie never replaces those bytes.
    return legacy.parent/'recon_versions'/digest/'l0'
