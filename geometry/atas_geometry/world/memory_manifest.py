"""Small explicit cross-machine index; dense geometry remains on the GPU host."""
import json
from pathlib import Path


def observation_manifest(l0, l1, *, payload, world_version=None, publication_status="pending_review",
                         candidate_job_id=None):
    l0 = Path(l0)
    meta_path = l0 / "observation.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    dense_path = l0 / "dense_complete.json"
    dense = json.loads(dense_path.read_text()) if dense_path.is_file() else {}
    depths = len(list((l0 / "depths").glob("*.exr")))
    masks = len(list((l0 / "dynamic_mask").glob("*.png")))
    fps = float(meta.get("source_fps") or 0)
    original = int(meta.get("original_frame_count") or depths)
    offset = max(0, (original - depths) // 2)
    return {"schema": "l0_l1_l2_manifest_v1",
            **{k: meta.get(k, payload.get(k)) for k in ("run_id", "story_id", "scene_id", "shot_id", "event_id")},
            "publication_status": publication_status, "world_version": world_version,
            "original_video_sha256": meta.get("original_video_sha256"),
            "review_lineage": payload.get("review_lineage", payload.get("lineage", meta.get("source_review_lineage"))),
            "candidate_job_id": candidate_job_id, "l0_dir": str(l0), "l1_npz": str(l1),
            "dense_manifest_path": str(dense_path) if dense_path.is_file() else None,
            "original_reconstruction_contract": dense.get("reconstruction_contract") or meta.get("source_reconstruction_contract"),
            "original_contract_digest": dense.get("contract_digest") or meta.get("source_reconstruction_contract_digest"),
            "l2": {"representation": "time_local_rgbd_masks", "frame_count": depths,
                   "original_frame_count": original, "source_fps": fps,
                   "time_start": offset / fps if fps else None,
                   "time_end_exclusive": (offset + depths) / fps if fps else None,
                   "depth_frames": depths, "dynamic_mask_frames": masks,
                   "coordinate_space": meta.get("coordinate_space", "raw_observation"),
                   "registered": meta.get("registered") is True,
                   "observation_metadata": str(meta_path),
                   "complete_per_frame": depths > 0 and depths == masks}}
