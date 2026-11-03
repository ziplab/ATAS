"""Per-block validation of static write-back; never grants a full-shot L2 re-capture."""
from dataclasses import replace
import numpy as np


def propose_block_registration(matches, tolerance):
    """Independent registration is not a bounded refinement: displacement is unrestricted, but it must still pass held-out frames."""
    from atas_geometry.register.anchored_fusion import robust_sim3, apply_sim3
    pairs={p['pair_id']:p['src_index'] for p in matches.pair_reports}
    labels=np.asarray([pairs[int(p)] for p in matches.pair_ids])
    frames=np.unique(labels)
    evidence=dict(accepted=False,method='independent_block_sim3_heldout_frames_v1')
    if len(frames)<4:
        return None,dict(evidence,reason='insufficient_independent_source_frames')
    held=np.isin(labels,frames[1::2])
    if min(int(held.sum()),int((~held).sum()))<30:
        return None,dict(evidence,reason='insufficient_training_or_validation_matches')
    try:
        tr=robust_sim3(matches.source[~held],matches.target[~held])
    except (ValueError,np.linalg.LinAlgError):
        return None,dict(evidence,reason='degenerate_alignment')
    errors=np.linalg.norm(apply_sim3(matches.source,tr)-matches.target,axis=1)
    accepted=bool(np.isfinite(errors).all() and np.median(errors[held])<=tolerance
        and np.mean(errors[held]<=2*tolerance)>=.65
        and np.median(errors)<=tolerance and np.mean(errors<=2*tolerance)>=.65)
    evidence.update(accepted=accepted,scale=float(tr[0]),rotation=tr[1].tolist(),translation=tr[2].tolist(),
        training_frames=frames[::2].tolist(),validation_frames=frames[1::2].tolist(),
        validation_median=float(np.median(errors[held])),validation_within_2t=float(np.mean(errors[held]<=2*tolerance)),
        tolerance=tolerance,reason='verified' if accepted else 'heldout_alignment_failed')
    return (tr if accepted else None),evidence


def eligible_chunks(report, count):
    owner = report.get('frame_owner', [])
    if len(owner) != count or report.get('frame_count') != count:
        raise ValueError('frame ownership is missing or inconsistent')
    chunks = report.get('chunks', [])
    ids = [c['chunk'] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate chunk identity')
    result = []
    for c in chunks:
        frames = [i for i, value in enumerate(owner) if value == c['chunk']]
        if any(not c['start'] <= i < c['end'] for i in frames):
            raise ValueError('frame outside owning chunk')
        if c.get('accepted_geometry') is True and len(frames) >= 2:
            result.append((c, frames))
    return result


def validate_blocks(reference, new, report):
    from atas_geometry.register.matching import collect_sift_anchors
    from atas_geometry.register.anchored_fusion import validate_cross_shot
    eligible = eligible_chunks(report, new.count)
    samples = sorted({frames[j] for _, frames in eligible for j in
                      np.linspace(0, len(frames)-1, min(5,len(frames))).astype(int)})
    if not samples:
        return dict(schema='partial_static_v1', accepted=False, frame_indices=[], chunks=[],
                    whole_clip_recapture_accepted=False)
    # Features are extracted once; no per-chunk re-reconstruction or pose fitting.
    matches = collect_sift_anchors(new, reference, profile='pooled_static_v1', source_indices=samples)
    chunks, accepted_frames = [], []
    for chunk, frames in eligible:
        pairs = [p for p in matches.pair_reports if p['src_index'] in frames]
        mask = np.isin(matches.pair_ids, [p['pair_id'] for p in pairs])
        subset = replace(matches, source=matches.source[mask], target=matches.target[mask],
                         pair_ids=matches.pair_ids[mask], pair_reports=pairs)
        check = validate_cross_shot(reference, new, report['anchor_tolerance'], matches=subset)
        correction = None
        if not check['accepted']:
            from atas_geometry.register.anchored_fusion import apply_sim3
            transform, correction = propose_block_registration(subset, report['anchor_tolerance'])
            if transform is not None:
                subset = replace(subset, source=apply_sim3(subset.source, transform))
                check = validate_cross_shot(reference, new, report['anchor_tolerance'], matches=subset)
        # Two independent source frames, not two historical matches of one frame.
        source_checks = []
        for i in sorted({p['src_index'] for p in pairs}):
            labels = [p['pair_id'] for p in pairs if p['src_index'] == i]
            errors = np.linalg.norm(subset.source - subset.target, axis=1)[np.isin(subset.pair_ids, labels)]
            source_checks.append(dict(frame=i, count=len(errors), accepted=bool(len(errors) >= 15
                and np.median(errors) <= report['anchor_tolerance']
                and np.mean(errors <= 2*report['anchor_tolerance']) >= .65)))
        accepted = check['accepted'] and sum(c['accepted'] for c in source_checks) >= 2
        # A bad sampled frame cannot hide in the pooled median of good frames.
        accepted = accepted and all(c['accepted'] for c in source_checks if c['count'] >= 15)
        chunks.append(dict(chunk=chunk['chunk'], accepted=bool(accepted), frames=frames,
                           cross_shot=check, source_checks=source_checks, correction=correction))
        if accepted:
            accepted_frames.extend(frames)
    return dict(schema='partial_static_v1', accepted=bool(accepted_frames),
                frame_indices=sorted(accepted_frames), chunks=chunks,
                whole_clip_recapture_accepted=False)


def fusion_frames(report, count):
    partial = report.get('partial_static_validation')
    if partial is None:
        return list(range(count)) if report.get('geometry_accepted') is True else []
    if partial.get('schema') != 'partial_static_v1' or partial.get('accepted') is not True:
        return []
    frames = partial.get('frame_indices', [])
    if any(type(i) is not int or not 0 <= i < count for i in frames) or len(set(frames)) != len(frames):
        raise ValueError('invalid partial fusion frames')
    allowed = {i for _, owned in eligible_chunks(report, count) for i in owned}
    verified = {i for c in partial.get('chunks', []) if c.get('accepted') is True for i in c['frames']}
    if not set(frames) <= allowed & verified:
        raise ValueError('unverified frame in partial fusion request')
    return sorted(frames)


def frame_corrections(report):
    """Only verified static blocks are transformed; raw L2 is never rewritten."""
    result={}
    for c in report.get('partial_static_validation',{}).get('chunks',[]):
        tr=c.get('correction')
        if c.get('accepted') is True and tr and tr.get('accepted') is True:
            transform=(tr['scale'],np.asarray(tr['rotation']),np.asarray(tr['translation']))
            result.update({i:transform for i in c['frames']})
    return result


def evidence_bins(frames, total_frames, *, partial):
    if any(type(i) is not int or not 0 <= i < total_frames for i in frames):
        raise ValueError('invalid temporal evidence frames')
    if frames != sorted(set(frames)):
        raise ValueError('temporal evidence must be ordered and unique')
    return {i:min(2,rank*3//len(frames) if partial else i*3//total_frames)
            for rank,i in enumerate(frames)}
