"""Static matching and optional offline fusion of Vista4D observations.

The anchored production path uses world_scaled_static_v1 verified matches and
audits their UNFITTED world residuals. Legacy Sim3 fitting remains available to
explicit offline/v2 callers; it is not an acceptance shortcut for anchored L0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from atas_geometry.memory.fusion import FusedSceneMemory, SimilarityTransform, _umeyama
from atas_geometry.memory.vista4d_layers import ReconStore
from atas_geometry.memory.fuse import (
    ALIGN_SCALE_MAX,
    ALIGN_SCALE_MIN,
    _align_indices,
    _static_frame_points,
    _unique_voxels,
    transform_c2w,
)


@dataclass(frozen=True)
class AnchorSet:
    source: np.ndarray
    target: np.ndarray
    pair_ids: np.ndarray
    pair_reports: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _lift_uv(recon: ReconStore, index: int, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = recon.depth_at(index)
    h, w = depth.shape[:2]
    u = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    z = depth[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-5)
    fx, fy, cx, cy = [float(x) for x in recon.intrinsics[index, :4]]
    # OpenCV keypoints index pixel centres at integer coordinates; our stored
    # Pi3 calibration and dense lifter use centres at index + 0.5.
    cam = np.stack(((u + .5 - cx) * z / max(fx, 1e-6), (v + .5 - cy) * z / max(fy, 1e-6), z), axis=1)
    return cam @ recon.cam_c2w[index, :3, :3].T + recon.cam_c2w[index, :3, 3], valid


def _lift_uv_robust(
    recon: ReconStore,
    index: int,
    uv: np.ndarray,
    *,
    max_relative_depth_jump: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Sub-pixel lift while rejecting depth discontinuities and unstable pixels."""
    depth = recon.depth_at(index).astype(np.float64)
    h, w = depth.shape[:2]
    u = np.asarray(uv[:, 0], dtype=np.float64)
    v = np.asarray(uv[:, 1], dtype=np.float64)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    inside = (u0 >= 1) & (u0 < w - 2) & (v0 >= 1) & (v0 < h - 2)
    safe_u = np.clip(u0, 1, w - 3)
    safe_v = np.clip(v0, 1, h - 3)
    du = u - u0
    dv = v - v0
    z00 = depth[safe_v, safe_u]
    z10 = depth[safe_v, safe_u + 1]
    z01 = depth[safe_v + 1, safe_u]
    z11 = depth[safe_v + 1, safe_u + 1]
    z = (
        z00 * (1.0 - du) * (1.0 - dv)
        + z10 * du * (1.0 - dv)
        + z01 * (1.0 - du) * dv
        + z11 * du * dv
    )
    neighbours = np.stack(
        [depth[safe_v + dy, safe_u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
        axis=1,
    )
    finite_neighbours = np.isfinite(neighbours) & (neighbours > 1e-5)
    stable = finite_neighbours.all(axis=1)
    local_span = np.ptp(neighbours, axis=1)
    stable &= local_span <= np.maximum(z * max_relative_depth_jump, 0.01)
    valid = inside & np.isfinite(z) & (z > 1e-5) & stable
    fx, fy, cx, cy = [float(x) for x in recon.intrinsics[index, :4]]
    # Interpolate depth in OpenCV's index coordinates, then convert the ray
    # coordinate to the same pixel-centre convention as dense reconstruction.
    cam = np.stack(((u + .5 - cx) * z / max(fx, 1e-6), (v + .5 - cy) * z / max(fy, 1e-6), z), axis=1)
    world = cam @ recon.cam_c2w[index, :3, :3].T + recon.cam_c2w[index, :3, 3]
    return world, valid, {
        "input": len(uv),
        "outside": int(np.count_nonzero(~inside)),
        "depth_unstable": int(np.count_nonzero(inside & ~stable)),
        "accepted": int(np.count_nonzero(valid)),
    }


def _grid_balance_matches(matches: list[Any], keypoints: list[Any], width: int, height: int, *, cols: int = 8, rows: int = 6, per_cell: int = 20) -> list[Any]:
    buckets: dict[tuple[int, int], list[Any]] = {}
    for match in sorted(matches, key=lambda item: float(item.distance)):
        u, v = keypoints[match.queryIdx].pt
        cell = (
            min(cols - 1, max(0, int(float(u) / max(width, 1) * cols))),
            min(rows - 1, max(0, int(float(v) / max(height, 1) * rows))),
        )
        bucket = buckets.setdefault(cell, [])
        if len(bucket) < per_cell:
            bucket.append(match)
    return [match for bucket in buckets.values() for match in bucket]


def evidence_frames(count: int, samples: int = 13) -> list[int]:
    """Frames sampled across a clip for consistency evidence (dense enough that
    neighbouring samples overlap under fast camera motion)."""
    if count <= samples:
        return list(range(count))
    return sorted({int(round(i)) for i in np.linspace(0, count - 1, samples)})


def collect_sift_anchors(new: ReconStore, reference: ReconStore, *, profile: str = "legacy_v21", world_tolerance: float | None = None, source_indices=None) -> AnchorSet:
    """Collect geometrically lifted matches from several independent frame pairs."""
    import cv2

    if profile not in {"legacy_v21", "robust_static_v22", "fixed_world_v1", "world_scaled_static_v1", "pooled_static_v1"}:
        raise ValueError('unknown static matching profile')
    if profile == 'world_scaled_static_v1':
        if world_tolerance is None or not math.isfinite(world_tolerance) or world_tolerance <= 0:
            raise ValueError('positive finite world tolerance required')
        # These are strictly tighter than the final fixed-world error budget.
        # No reference metres or match-derived residual statistics set the units.
        pair_threshold, pair_median_limit = .5 * world_tolerance, .35 * world_tolerance
    else:
        pair_threshold, pair_median_limit = .035, .025

    detector = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # Cross-shot validation profiles compare 13 frames per clip: five could miss the
    # views the two clips actually share, leaving no evidence either way.
    sample = (evidence_frames if profile in {"world_scaled_static_v1", "pooled_static_v1"}
              else _align_indices)
    new_ids = sample(new.count) if source_indices is None else sorted(set(source_indices))
    if any(type(i) is not int or not 0 <= i < new.count for i in new_ids):
        raise ValueError('invalid source frame indices')
    ref_ids = sample(reference.count)

    def describe(recon: ReconStore, index: int):
        gray = cv2.cvtColor(np.asarray(recon.frames[index]), cv2.COLOR_RGB2GRAY)
        return detector.detectAndCompute(gray, None)

    new_features = {i: describe(new, i) for i in new_ids}
    ref_features = {i: describe(reference, i) for i in ref_ids}
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "profile": profile,
        "frame_pair_candidates": 0,
        "insufficient_feature_matches": 0,
        "insufficient_static_depth_matches": 0,
        "pair_ransac_failed": 0,
        "pair_quality_failed": 0,
        "ratio_matches_total": 0,
        "mutual_grid_matches_total": 0,
        "pair_details": [],
    }
    pair_id = 0
    # A sparse pair need not independently estimate a seven-DOF transform.
    # Pool its independently matched static points; the caller validates the
    # complete observation in the fixed world, without residual preselection.
    minimum_pair_matches = 3 if profile == 'pooled_static_v1' else 12
    for src_index in new_ids:
        src_kp, src_desc = new_features[src_index]
        if src_desc is None or len(src_kp) < 12:
            continue
        for dst_index in ref_ids:
            diagnostics["frame_pair_candidates"] += 1
            detail = {"src_index": int(src_index), "dst_index": int(dst_index),
                      "status": "rejected", "reasons": []}
            diagnostics["pair_details"].append(detail)
            dst_kp, dst_desc = ref_features[dst_index]
            if dst_desc is None or len(dst_kp) < 12:
                detail["reasons"] = ["insufficient_target_features"]
                continue
            forward = [p[0] for p in matcher.knnMatch(src_desc, dst_desc, k=2) if len(p) == 2 and p[0].distance < 0.72 * p[1].distance]
            diagnostics["ratio_matches_total"] += len(forward)
            if profile in {"robust_static_v22", "fixed_world_v1", "world_scaled_static_v1", "pooled_static_v1"}:
                reverse = [p[0] for p in matcher.knnMatch(dst_desc, src_desc, k=2) if len(p) == 2 and p[0].distance < 0.72 * p[1].distance]
                reverse_lookup = {match.queryIdx: match.trainIdx for match in reverse}
                forward = [match for match in forward if reverse_lookup.get(match.trainIdx) == match.queryIdx]
                h, w = new.frames[src_index].shape[:2]
                good = _grid_balance_matches(forward, src_kp, w, h)
                diagnostics["mutual_grid_matches_total"] += len(good)
            else:
                good = forward
            if len(good) < minimum_pair_matches:
                detail.update(raw_matches=len(good), reasons=["insufficient_feature_matches"])
                diagnostics["insufficient_feature_matches"] += 1
                continue
            src_uv = np.asarray([src_kp[m.queryIdx].pt for m in good], dtype=np.float64)
            dst_uv = np.asarray([dst_kp[m.trainIdx].pt for m in good], dtype=np.float64)
            if profile == "fixed_world_v1":
                # Reject image mismatches independently of the 3-D residual that
                # the caller will audit. Never refit or residual-select the world.
                cv2.setRNGSeed(1000 + diagnostics["frame_pair_candidates"])
                matrix, inliers = cv2.findFundamentalMat(src_uv, dst_uv, cv2.USAC_MAGSAC, 1.5, .999, 10000)
                epipolar = np.zeros(len(src_uv), bool) if matrix is None or inliers is None else inliers.ravel().astype(bool)
                detail.update(epipolar_inliers=int(epipolar.sum()), epipolar_threshold_pixels=1.5)
                if epipolar.sum() < 12:
                    detail["reasons"] = ["insufficient_epipolar_matches"]
                    continue
                src_uv, dst_uv = src_uv[epipolar], dst_uv[epipolar]
            if profile in {"robust_static_v22", "fixed_world_v1", "world_scaled_static_v1", "pooled_static_v1"}:
                src_xyz, src_valid, src_depth_report = _lift_uv_robust(new, src_index, src_uv)
                dst_xyz, dst_valid, dst_depth_report = _lift_uv_robust(reference, dst_index, dst_uv)
            else:
                src_xyz, src_valid = _lift_uv(new, src_index, src_uv)
                dst_xyz, dst_valid = _lift_uv(reference, dst_index, dst_uv)
                src_depth_report = dst_depth_report = None
            src_h, src_w = new.frames[src_index].shape[:2]
            dst_h, dst_w = reference.frames[dst_index].shape[:2]
            src_u = np.clip(np.rint(src_uv[:, 0]).astype(np.int64), 0, src_w - 1)
            src_v = np.clip(np.rint(src_uv[:, 1]).astype(np.int64), 0, src_h - 1)
            dst_u = np.clip(np.rint(dst_uv[:, 0]).astype(np.int64), 0, dst_w - 1)
            dst_v = np.clip(np.rint(dst_uv[:, 1]).astype(np.int64), 0, dst_h - 1)
            src_dynamic = new.dynamic_mask_at(src_index, (src_h, src_w))[src_v, src_u]
            dst_dynamic = reference.dynamic_mask_at(dst_index, (dst_h, dst_w))[dst_v, dst_u]
            keep = src_valid & dst_valid & ~src_dynamic & ~dst_dynamic
            detail.update(raw_matches=len(good), static_depth_matches=int(keep.sum()),
                          source_depth_filter=src_depth_report, target_depth_filter=dst_depth_report)
            if np.count_nonzero(keep) < minimum_pair_matches:
                detail["reasons"] = ["insufficient_static_depth_matches"]
                diagnostics["insufficient_static_depth_matches"] += 1
                continue
            src_xyz, dst_xyz = src_xyz[keep], dst_xyz[keep]
            pair_transform: SimilarityTransform | None = None
            pair_keep = np.ones(len(src_xyz), dtype=bool)
            pair_median: float | None = None
            if profile in {"robust_static_v22", "world_scaled_static_v1"}:
                try:
                    pair_transform, pair_keep = _fit_ransac(
                        src_xyz, dst_xyz, threshold=pair_threshold, seed=1000 + pair_id
                    )
                    pair_error = np.linalg.norm(pair_transform.apply(src_xyz) - dst_xyz, axis=1)
                    pair_median = float(np.median(pair_error[pair_keep])) if pair_keep.any() else None
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    detail.update(reasons=["pair_ransac_failed"], error=str(exc))
                    diagnostics["pair_ransac_failed"] += 1
                    continue
                detail.update(inliers=int(pair_keep.sum()), inlier_ratio=float(pair_keep.mean()),
                              median_error=pair_median, transform=pair_transform.to_json_dict(),
                              ransac_threshold=pair_threshold, median_limit=pair_median_limit,
                              fixed_world_median_error=float(np.median(np.linalg.norm(src_xyz-dst_xyz,axis=1))))
                reasons = []
                if int(pair_keep.sum()) < 12: reasons.append("insufficient_inliers")
                if float(pair_keep.mean()) < .25: reasons.append("low_inlier_ratio")
                if pair_median is None or pair_median > pair_median_limit: reasons.append("high_median_error")
                if not ALIGN_SCALE_MIN <= pair_transform.scale <= ALIGN_SCALE_MAX: reasons.append("scale_out_of_range")
                detail["reasons"] = reasons
                if reasons:
                    diagnostics["pair_quality_failed"] += 1
                    continue
                src_xyz, dst_xyz = src_xyz[pair_keep], dst_xyz[pair_keep]
            source.append(src_xyz)
            target.append(dst_xyz)
            labels.append(np.full(len(src_xyz), pair_id, dtype=np.int32))
            pair_report: dict[str, Any] = {"pair_id": pair_id, "src_index": src_index, "dst_index": dst_index, "raw_matches": len(good), "static_matches": len(src_xyz)}
            if pair_transform is not None:
                pair_report.update(
                    pair_inlier_ratio=float(pair_keep.mean()),
                    pair_median_error=pair_median,
                    pair_transform=pair_transform.to_json_dict(),
                    source_depth_filter=src_depth_report,
                    target_depth_filter=dst_depth_report,
                )
            reports.append(pair_report)
            detail.update(status="accepted", pair_id=pair_id)
            pair_id += 1
    if not source:
        return AnchorSet(np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=np.int32), (), diagnostics)
    diagnostics["accepted_pairs_before_consensus"] = len(reports)
    anchors = AnchorSet(np.concatenate(source), np.concatenate(target), np.concatenate(labels), tuple(reports), diagnostics)
    return (_pair_consensus(anchors, translation_tolerance=2*world_tolerance)
            if profile == 'world_scaled_static_v1' else
            _pair_consensus(anchors) if profile == "robust_static_v22" else anchors)


def _pair_consensus(anchors: AnchorSet, *, translation_tolerance: float = .12) -> AnchorSet:
    """Keep the dominant frame-pair transform cluster, not point-level accidents."""
    transforms: dict[int, SimilarityTransform] = {}
    weights: dict[int, int] = {}
    for report in anchors.pair_reports:
        raw = report.get("pair_transform")
        if not raw:
            continue
        pair_id = int(report["pair_id"])
        transforms[pair_id] = SimilarityTransform(
            float(raw["scale"]),
            np.asarray(raw["rotation"], dtype=np.float64),
            np.asarray(raw["translation"], dtype=np.float64),
        )
        weights[pair_id] = int(report["static_matches"])
    if len(transforms) < 2:
        return anchors
    clusters: dict[int, set[int]] = {}
    for candidate_id, candidate in transforms.items():
        clusters[candidate_id] = {
            other_id
            for other_id, other in transforms.items()
            if (
                transform_disagreement(candidate, other)["scale_log"] <= math.log(1.12)
                and transform_disagreement(candidate, other)["rotation_deg"] <= 8.0
                and transform_disagreement(candidate, other)["translation"] <= translation_tolerance
            )
        }
    winner = max(clusters, key=lambda key: sum(weights[item] for item in clusters[key]))
    keep_ids = clusters[winner]
    keep = np.asarray([int(pair) in keep_ids for pair in anchors.pair_ids], dtype=bool)
    reports = tuple(report for report in anchors.pair_reports if int(report["pair_id"]) in keep_ids)
    diagnostics = dict(anchors.diagnostics)
    diagnostics.update(
        consensus_pair_ids=sorted(keep_ids),
        accepted_pairs_after_consensus=len(reports),
        anchors_after_consensus=int(keep.sum()),
    )
    return AnchorSet(anchors.source[keep], anchors.target[keep], anchors.pair_ids[keep], reports, diagnostics)


def _fit_ransac(source: np.ndarray, target: np.ndarray, *, threshold: float, seed: int = 7) -> tuple[SimilarityTransform, np.ndarray]:
    if len(source) < 3:
        raise ValueError("at least three anchors are required")
    rng = np.random.default_rng(seed)
    best: tuple[int, float, SimilarityTransform, np.ndarray] | None = None
    iterations = min(1600, max(400, len(source) * 3))
    for _ in range(iterations):
        ids = rng.choice(len(source), 3, replace=False)
        try:
            transform = _umeyama(source[ids], target[ids])
        except (ValueError, np.linalg.LinAlgError):
            continue
        error = np.linalg.norm(transform.apply(source) - target, axis=1)
        keep = error <= threshold
        score = (int(keep.sum()), -float(np.median(error[keep])) if keep.any() else -math.inf)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], transform, keep)
    if best is None or best[0] < 8:
        raise RuntimeError("SIFT Sim3 RANSAC found no consensus")
    keep = best[3]
    transform = _umeyama(source[keep], target[keep])
    return transform, np.linalg.norm(transform.apply(source) - target, axis=1) <= threshold


def fit_camera_prior(source_c2w: np.ndarray, target_c2w: np.ndarray) -> SimilarityTransform | None:
    """Fit reconstructed camera motion to the already-authored target trajectory."""
    if len(source_c2w) < 2 or len(target_c2w) < 2:
        return None
    ids = np.rint(np.linspace(0, len(target_c2w) - 1, len(source_c2w))).astype(np.int64)
    target = target_c2w[ids]
    source_center = source_c2w[:, :3, 3]
    target_center = target[:, :3, 3]
    baseline = max(float(np.ptp(target_center, axis=0).max()), 0.1)
    # Axis endpoints keep locked/short trajectories from becoming an arbitrary Sim3.
    source_aug = [source_center]
    target_aug = [target_center]
    for axis in (0, 1, 2):
        source_aug.append(source_center + source_c2w[:, :3, axis] * baseline)
        target_aug.append(target_center + target[:, :3, axis] * baseline)
    try:
        return _umeyama(np.concatenate(source_aug), np.concatenate(target_aug))
    except (ValueError, np.linalg.LinAlgError):
        return None


def _rotation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    r = a @ b.T
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))))


def transform_disagreement(a: SimilarityTransform, b: SimilarityTransform) -> dict[str, float]:
    return {
        "scale_log": abs(float(math.log(max(a.scale, 1e-9) / max(b.scale, 1e-9)))),
        "rotation_deg": _rotation_degrees(a.rotation, b.rotation),
        "translation": float(np.linalg.norm(a.translation - b.translation)),
    }


def camera_prior_quality(
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
    prior: SimilarityTransform,
) -> dict[str, Any]:
    """Decide whether an authored camera path is strong enough to veto SIFT.

    Wan may not follow the point-cloud condition. In that case fitting the
    reconstructed trajectory to the authored one can produce a mathematically
    valid but physically absurd Sim3. Such a path is useful evidence of camera
    follow failure, not evidence that static geometry is misregistered.
    """
    ids = np.rint(np.linspace(0, len(target_c2w) - 1, len(source_c2w))).astype(np.int64)
    target = target_c2w[ids]
    source_centers = np.asarray(source_c2w[:, :3, 3], dtype=np.float64)
    target_centers = np.asarray(target[:, :3, 3], dtype=np.float64)
    source_travel = float(np.linalg.norm(np.diff(source_centers, axis=0), axis=1).sum())
    target_travel = float(np.linalg.norm(np.diff(target_centers, axis=0), axis=1).sum())
    predicted_centers = prior.apply(source_centers)
    center_error = np.linalg.norm(predicted_centers - target_centers, axis=1)
    center_rmse = float(np.sqrt(np.mean(center_error * center_error)))
    rotation_error = np.asarray(
        [
            _rotation_degrees(prior.rotation @ source_c2w[index, :3, :3], target[index, :3, :3])
            for index in range(len(source_c2w))
        ],
        dtype=np.float64,
    )
    rotation_mean = float(rotation_error.mean())
    reasons: list[str] = []
    if source_travel < 0.02 or target_travel < 0.02:
        reasons.append("insufficient_camera_baseline")
    if not ALIGN_SCALE_MIN <= float(prior.scale) <= ALIGN_SCALE_MAX:
        reasons.append("prior_scale_out_of_range")
    normalized_center_rmse = center_rmse / max(target_travel, 0.02)
    if normalized_center_rmse > 0.25:
        reasons.append("camera_center_fit_residual")
    if rotation_mean > 15.0:
        reasons.append("camera_rotation_fit_residual")
    usable = not reasons
    return {
        "status": "usable" if usable else "prior_unusable",
        "veto_enabled": usable,
        "prior_unusable": not usable,
        "camera_follow_mismatch": not usable,
        "reasons": reasons,
        "source_travel": source_travel,
        "target_travel": target_travel,
        "center_rmse": center_rmse,
        "normalized_center_rmse": normalized_center_rmse,
        "rotation_error_degrees_mean": rotation_mean,
        "rotation_error_degrees_max": float(rotation_error.max()),
    }


def refine_rigid_icp(source: np.ndarray, target: np.ndarray, initial: SimilarityTransform, *, max_points: int = 40000, iterations: int = 8) -> tuple[SimilarityTransform, dict[str, Any]]:
    """Small, reversible point-to-point refinement on a held-out geometry split."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(11)
    sampled = source[rng.choice(len(source), min(len(source), max_points), replace=False)]
    permutation = rng.permutation(len(sampled))
    split = max(30, int(len(sampled) * 0.75))
    src = sampled[permutation[:split]]
    evaluation = sampled[permutation[split:]]
    if len(evaluation) < 30:
        evaluation = src
    dst = target[rng.choice(len(target), min(len(target), max_points), replace=False)]
    tree = cKDTree(dst)
    current = initial
    used = 0

    def metrics(transform: SimilarityTransform) -> tuple[float, float]:
        distances, _ = tree.query(transform.apply(evaluation), k=1, workers=-1)
        return float(np.median(distances)), float(np.percentile(distances, 75))

    before_median, before_p75 = metrics(initial)
    for _ in range(iterations):
        moved = current.apply(src)
        distances, indices = tree.query(moved, k=1, workers=-1)
        finite = np.isfinite(distances)
        cutoff = float(np.percentile(distances[finite], 45.0)) if finite.any() else 0.0
        keep = finite & (distances <= cutoff)
        used = int(keep.sum())
        if used < 30:
            break
        source_fit = moved[keep]
        target_fit = dst[indices[keep]]
        source_center = source_fit.mean(axis=0)
        target_center = target_fit.mean(axis=0)
        covariance = (source_fit - source_center).T @ (target_fit - target_center)
        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = target_center - source_center @ rotation.T
        delta = SimilarityTransform(1.0, rotation, translation)
        if _rotation_degrees(delta.rotation, np.eye(3)) > 3.0:
            break
        current = SimilarityTransform(
            scale=float(current.scale),
            rotation=delta.rotation @ current.rotation,
            translation=current.translation @ delta.rotation.T + delta.translation,
        )
    after_median, after_p75 = metrics(current)
    accepted = bool(
        after_median < before_median * 0.98
        and after_p75 <= before_p75
    )
    chosen = current if accepted else initial
    return chosen, {
        "attempted": True,
        "accepted": accepted,
        "reason": "heldout_geometry_improved" if accepted else "heldout_geometry_not_improved",
        "fit_sample_count": len(src),
        "evaluation_sample_count": len(evaluation),
        "correspondences": used,
        "median_before": before_median,
        "median_after": after_median,
        "p75_before": before_p75,
        "p75_after": after_p75,
    }


def adaptive_overlap_radius(voxel_size: float, heldout_error: np.ndarray) -> float:
    """Registration uncertainty controls duplicate-surface suppression."""
    residual_p75 = float(np.percentile(heldout_error, 75)) if len(heldout_error) else 0.0
    return float(np.clip(max(voxel_size * 1.5, residual_p75 * 1.25), voxel_size * 1.5, 0.05))


def validate_heldout(error: np.ndarray, voxel_size: float) -> dict[str, Any]:
    median = float(np.median(error)) if len(error) else math.inf
    tolerance = max(0.03, voxel_size * 5.0)
    ratio = float(np.mean(error <= tolerance)) if len(error) else 0.0
    median_limit = max(0.025, voxel_size * 4.0)
    return {
        "median_error": median,
        "inlier_ratio": ratio,
        "inlier_tolerance": tolerance,
        "median_limit": median_limit,
        "accepted": bool(median <= median_limit and ratio >= 0.65),
    }


def classify_candidate_rays(
    existing_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    camera_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    *,
    depth_tolerance: float,
    cell_pixels: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    """Allow writes only through rays where the current L1 has no observation."""
    width, height = image_size
    grid_w = int(math.ceil(width / cell_pixels))
    grid_h = int(math.ceil(height / cell_pixels))
    w2c = np.linalg.inv(np.asarray(camera_c2w, dtype=np.float64))
    fx, fy, cx, cy = [float(value) for value in intrinsics[:4]]

    def project(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera = xyz.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
        depth = camera[:, 2]
        u = fx * camera[:, 0] / np.maximum(depth, 1e-9) + cx
        v = fy * camera[:, 1] / np.maximum(depth, 1e-9) + cy
        cell_u = np.floor(u / cell_pixels).astype(np.int64)
        cell_v = np.floor(v / cell_pixels).astype(np.int64)
        valid = (
            np.isfinite(camera).all(axis=1)
            & (depth > 1e-5)
            & (cell_u >= 0)
            & (cell_u < grid_w)
            & (cell_v >= 0)
            & (cell_v < grid_h)
        )
        return cell_v * grid_w + cell_u, depth, valid

    existing_pixel, existing_depth, existing_valid = project(existing_xyz)
    z_buffer = np.full(grid_h * grid_w, np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, existing_pixel[existing_valid], existing_depth[existing_valid])
    candidate_pixel, candidate_depth, candidate_valid = project(candidate_xyz)
    observed_depth = np.full(len(candidate_xyz), np.inf, dtype=np.float64)
    observed_depth[candidate_valid] = z_buffer[candidate_pixel[candidate_valid]]
    covered = candidate_valid & np.isfinite(observed_depth)
    overlap = covered & (np.abs(candidate_depth - observed_depth) <= depth_tolerance)
    conflict = covered & ~overlap
    unknown = candidate_valid & ~covered
    return unknown, {
        "candidate_points": int(len(candidate_xyz)),
        "outside_camera": int(np.count_nonzero(~candidate_valid)),
        "unknown_ray_points": int(np.count_nonzero(unknown)),
        "covered_overlap_points": int(np.count_nonzero(overlap)),
        "covered_conflict_points": int(np.count_nonzero(conflict)),
        "nearer_conflicts": int(np.count_nonzero(conflict & (candidate_depth < observed_depth))),
        "farther_conflicts": int(np.count_nonzero(conflict & (candidate_depth > observed_depth))),
    }


def _collect_novel_frames(
    recon: ReconStore,
    frames: Iterable[int],
    transform: SimilarityTransform,
    existing: FusedSceneMemory,
    *,
    device: Any,
    overlap_radius: float,
    temporal_bin_count: int = 3,
    min_temporal_bins: int = 3,
    support_stats_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream frames so a dense clip is never concatenated in host memory."""
    from atas_geometry.memory.vista4d_layers import _voxel_keys
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    accepted_keys: set[int] = set()
    # Python integer bitsets keep per-cell support exact without millions of
    # tiny set objects on dense 1080p observations.
    support_frames: dict[tuple[int, int, int], int] = {}
    support_bins: dict[tuple[int, int, int], int] = {}
    point_cells: list[np.ndarray] = []
    try:
        from scipy.spatial import cKDTree

        existing_tree: Any = cKDTree(np.asarray(existing.xyz, dtype=np.float32))
    except ImportError:
        existing_tree = None
    candidates = 0
    static_pixels = 0
    ray_totals = {
        "candidate_points": 0,
        "outside_camera": 0,
        "unknown_ray_points": 0,
        "covered_overlap_points": 0,
        "covered_conflict_points": 0,
        "nearer_conflicts": 0,
        "farther_conflicts": 0,
    }
    frame_list = [int(index) for index in frames]
    if temporal_bin_count < 1 or not 1 <= min_temporal_bins <= temporal_bin_count:
        raise ValueError("temporal support requires 1 <= min_temporal_bins <= temporal_bin_count")
    for frame_position, index in enumerate(frame_list):
        temporal_bin = min(
            temporal_bin_count - 1,
            int(frame_position * temporal_bin_count / max(len(frame_list), 1)),
        )
        xyz, rgb = _static_frame_points(recon, int(index))
        static_pixels += len(xyz)
        if not len(xyz):
            continue
        xyz = transform.apply(xyz).astype(np.float32)
        novel_xyz, novel_rgb, frame_report = select_novel_points(
            xyz,
            rgb,
            existing,
            device=device,
            existing_tree=existing_tree,
            novelty_radius=overlap_radius,
        )
        candidates += int(frame_report["candidate_unique_points"])
        if not len(novel_xyz):
            continue
        aligned_camera = transform_c2w(
            recon.cam_c2w[index : index + 1],
            scale=transform.scale,
            rotation=transform.rotation,
            translation=transform.translation,
        )[0]
        native_h, native_w = recon.frames[index].shape[:2]
        unknown_ray, ray_report = classify_candidate_rays(
            np.asarray(existing.xyz),
            novel_xyz,
            aligned_camera,
            recon.intrinsics[index],
            (native_w, native_h),
            depth_tolerance=max(overlap_radius, float(existing.voxel_size) * 2.0),
        )
        for key in ray_totals:
            ray_totals[key] += int(ray_report[key])
        novel_xyz, novel_rgb = novel_xyz[unknown_ray], novel_rgb[unknown_ray]
        if not len(novel_xyz):
            continue
        keys = _voxel_keys(novel_xyz, np.asarray(existing.voxel_origin), float(existing.voxel_size))
        support_size = max(float(existing.voxel_size) * 4.0, overlap_radius)
        all_cells = np.floor(
            (novel_xyz.astype(np.float64) - np.asarray(existing.voxel_origin)) / support_size
        ).astype(np.int64)
        # Count a support cell at most once per source frame. This is computed
        # before fine-voxel dedup so a perfectly aligned new wall still records
        # repeated observation rather than disappearing after frame one.
        for row in np.unique(all_cells, axis=0):
            cell = tuple(int(v) for v in row)
            support_frames[cell] = support_frames.get(cell, 0) | (1 << frame_position)
            support_bins[cell] = support_bins.get(cell, 0) | (1 << temporal_bin)
        keep = np.asarray([int(key) not in accepted_keys for key in keys], dtype=bool)
        if keep.any():
            kept_xyz = novel_xyz[keep]
            xyz_parts.append(kept_xyz)
            rgb_parts.append(novel_rgb[keep])
            cells = all_cells[keep]
            point_cells.append(cells)
            accepted_keys.update(int(key) for key in keys[keep])
    if not xyz_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), {"static_pixels": static_pixels, "candidate_unique_points": candidates, "ray_classification": ray_totals, "novel_points": 0, "novelty_radius": overlap_radius, "temporal_bin_count": temporal_bin_count, "min_temporal_bins": min_temporal_bins}
    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    cells = np.concatenate(point_cells)
    supported = {
        cell for cell, bins in support_bins.items() if bins.bit_count() >= min_temporal_bins
    }
    coherent: set[tuple[int, int, int]] = set()
    for cell in supported:
        if any(
            (cell[0] + dx, cell[1] + dy, cell[2] + dz) in supported
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        ):
            coherent.add(cell)
    keep = np.asarray([tuple(int(v) for v in row) in coherent for row in cells.tolist()], dtype=bool)
    xyz, rgb = xyz[keep], rgb[keep]
    ordered_cells = sorted(support_frames)
    frame_counts = np.asarray([support_frames[cell].bit_count() for cell in ordered_cells], dtype=np.int16)
    bin_counts = np.asarray([support_bins[cell].bit_count() for cell in ordered_cells], dtype=np.int8)
    accepted_cells = np.asarray([cell in coherent for cell in ordered_cells], dtype=bool)
    if support_stats_path is not None:
        support_stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            support_stats_path,
            cells=np.asarray(ordered_cells, dtype=np.int64),
            frame_support=frame_counts,
            bin_support=bin_counts,
            accepted=accepted_cells,
            temporal_bin_count=np.asarray(temporal_bin_count, dtype=np.int16),
            min_temporal_bins=np.asarray(min_temporal_bins, dtype=np.int16),
        )
    frame_hist = np.bincount(frame_counts.astype(np.int64), minlength=len(frame_list) + 1)
    bin_hist = np.bincount(bin_counts.astype(np.int64), minlength=temporal_bin_count + 1)
    return xyz, rgb, {
        "static_pixels": static_pixels,
        "candidate_unique_points": candidates,
        "ray_classification": ray_totals,
        "far_points_before_consistency": int(len(keep)),
        "multi_frame_support_cells": len(supported),
        "coherent_support_cells": len(coherent),
        "candidate_cell_count": len(ordered_cells),
        "frame_support_histogram": {str(i): int(value) for i, value in enumerate(frame_hist) if value},
        "bin_support_histogram": {str(i): int(value) for i, value in enumerate(bin_hist) if value},
        "temporal_bin_count": temporal_bin_count,
        "min_temporal_bins": min_temporal_bins,
        "support_stats_path": str(support_stats_path) if support_stats_path else None,
        "novel_points": len(xyz),
        "novelty_radius": overlap_radius,
        "policy": "fine L1 voxels retained only inside adjacent support cells persistent across required early/mid/late bins",
    }


def select_novel_points(xyz: np.ndarray, rgb: np.ndarray, existing: FusedSceneMemory, *, device: Any, existing_tree: Any = None, novelty_radius: float | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deduplicate at L1 resolution; no 16x coarse grid and no 27-neighbour veto."""
    voxel = float(existing.voxel_size)
    if device is None:
        # Pure-numpy path makes the policy independently testable. Production
        # passes a CUDA device and retains the existing GPU unique operation.
        origin = np.asarray(existing.voxel_origin)
        cells = np.floor((xyz.astype(np.float64) - origin) / voxel).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        first.sort()
        uniq_xyz, uniq_rgb = xyz[first], rgb[first]
    else:
        uniq_xyz, uniq_rgb, _ = _unique_voxels(xyz, rgb, np.asarray(existing.voxel_origin), voxel, device)
    try:
        from scipy.spatial import cKDTree

        tree = existing_tree or cKDTree(np.asarray(existing.xyz, dtype=np.float32))
        distance, _ = tree.query(uniq_xyz, k=1, workers=-1)
    except ImportError:
        # Small synthetic/minimal environments only. Real scenes should install
        # scipy so this does not become a quadratic operation.
        reference = np.asarray(existing.xyz, dtype=np.float32)
        chunks = []
        for start in range(0, len(uniq_xyz), 1024):
            delta = uniq_xyz[start : start + 1024, None, :] - reference[None, :, :]
            chunks.append(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
        distance = np.concatenate(chunks) if chunks else np.empty(0)
    threshold = float(novelty_radius if novelty_radius is not None else voxel * 1.25)
    novel = distance > threshold
    return uniq_xyz[novel], uniq_rgb[novel], {
        "candidate_unique_points": int(len(uniq_xyz)),
        "novelty_radius": float(threshold),
        "novel_points": int(novel.sum()),
        "nearest_median": float(np.median(distance)) if len(distance) else None,
        "nearest_p90": float(np.percentile(distance, 90)) if len(distance) else None,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
