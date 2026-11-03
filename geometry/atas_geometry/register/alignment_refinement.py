"""Bounded correction, fitted and verified on disjoint source frames.

This is not an acceptance-threshold fallback. A failed correction stays pending.
"""
from pathlib import Path
import json
import numpy as np


def propose_correction(matches, tolerance):
    from atas_geometry.register.anchored_fusion import robust_sim3, apply_sim3
    by_pair = {p['pair_id']: p['src_index'] for p in matches.pair_reports}
    frames = np.array([by_pair[int(p)] for p in matches.pair_ids])
    distinct = np.unique(frames)
    result = {'accepted': False, 'method': 'bounded_sim3_disjoint_source_frames_v1'}
    if len(distinct) < 4:
        return None, dict(result, reason='insufficient_independent_source_frames')
    held = np.isin(frames, distinct[1::2])
    if min(int(held.sum()), int((~held).sum())) < 30:
        return None, dict(result, reason='insufficient_training_or_validation_matches')
    try:
        tr = robust_sim3(matches.source[~held], matches.target[~held])
    except (ValueError, np.linalg.LinAlgError):
        return None, dict(result, reason='degenerate_alignment')
    scale, rotation, translation = tr
    aligned = apply_sim3(matches.source, tr)
    error = np.linalg.norm(aligned - matches.target, axis=1)
    before = np.linalg.norm(matches.source - matches.target, axis=1)
    displacement = np.linalg.norm(aligned - matches.source, axis=1)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))
    bounded = .9 <= scale <= 1.1 and angle <= 5 and np.quantile(displacement, .95) <= 2*tolerance
    # Same residual limits as the existing check, on data never used to fit.
    verified = (np.median(error[held]) <= tolerance
                and np.mean(error[held] <= 2*tolerance) >= .65
                and np.median(error[held]) < np.median(before[held])
                and np.median(error) <= tolerance and np.mean(error <= 2*tolerance) >= .65)
    result.update(accepted=bool(bounded and verified), bounded=bool(bounded),
                  reason='verified' if bounded and verified else 'correction_not_verified',
                  scale=float(scale), rotation=rotation.tolist(), translation=translation.tolist(),
                  rotation_degrees=angle, displacement_p95=float(np.quantile(displacement,.95)),
                  training_frames=distinct[::2].tolist(), validation_frames=distinct[1::2].tolist(),
                  validation_before_median=float(np.median(before[held])),
                  validation_after_median=float(np.median(error[held])),
                  validation_within_2t=float(np.mean(error[held]<=2*tolerance)),
                  median_world_error=float(np.median(error)),
                  within_2t_fraction=float(np.mean(error<=2*tolerance)))
    return (tr if result['accepted'] else None), result


def materialize_correction(store, destination, transform, report, evidence):
    """New version only: preserve every frame/mask/time, transform depth and pose once."""
    from atas_geometry.register.anchored_fusion import apply_sim3, write_exr, write_json
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    scale, rotation, _ = transform
    for path in store.recon_dir.iterdir():
        if path.name not in {'depths', 'cameras.npz', 'anchored_reconstruction.json', 'progress.json'}:
            (destination/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (destination/'depths').mkdir()
    for i in range(store.count):
        write_exr(destination/'depths'/f'{i:05d}.exr', store.depth_at(i)*scale)
    poses = store.cam_c2w.copy()
    poses[:,:3,3] = apply_sim3(poses[:,:3,3], transform)
    poses[:,:3,:3] = rotation[None] @ poses[:,:3,:3]
    np.savez(destination/'cameras.npz', cam_c2w=poses, intrinsics=store.intrinsics)
    updated = json.loads(json.dumps(report, default=lambda v: v.tolist()))
    updated['alignment_refinement'] = evidence
    updated['unrefined_l0'] = str(store.recon_dir)
    write_json(destination/'anchored_reconstruction.json', updated)
    return updated
