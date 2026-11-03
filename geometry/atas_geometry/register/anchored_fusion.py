"""History-conditioned Pi3X reconstruction and append-only static writeback.

Accepted RGB-D and world memory are immutable inputs. The output is a separate
dense RGB-D observation plus a reviewable L1 candidate; this module never
promotes a candidate or rewrites an accepted observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist()) + "\n")
    os.replace(temporary, path)


def frame_windows(count, size=24, overlap=4):
    if count < 1 or size <= overlap or overlap < 0:
        raise ValueError("invalid frame window parameters")
    start = 0
    while start < count:
        # The last window is moved back to full size instead of being a short
        # remainder: a few frames give Pi3X too little motion context, and the
        # extra overlap is re-predicted, never re-owned.
        if start + size > count and start > 0:
            start = max(1, count - size)
        end = min(count, start + size)
        yield start, end
        if end == count:
            break
        start = end - overlap


class AnchorFitError(ValueError):
    """This reference cannot establish a transform; not a worker crash."""


def fit_sim3(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 6 or source.shape != target.shape:
        raise AnchorFitError("not enough paired anchor points")
    a, b = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(b.T @ a / len(a))
    if singular[1] < max(singular[0], 1e-12) * 1e-7:
        raise AnchorFitError("degenerate anchor geometry")
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    r = u @ np.diag(sign) @ vt
    scale = float((singular * sign).sum() / np.mean(np.sum(a * a, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise AnchorFitError("invalid anchor scale")
    return scale, r, target.mean(0) - scale * (r @ source.mean(0))


def apply_sim3(points, transform):
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def robust_sim3(source, target):
    keep = np.ones(len(source), bool)
    for _ in range(8):
        tr = fit_sim3(source[keep], target[keep])
        error = np.linalg.norm(apply_sim3(source, tr) - target, axis=1)
        updated = error <= max(np.quantile(error, .65), 1e-7)
        if np.array_equal(keep, updated):
            break
        keep = updated
    return fit_sim3(source[keep], target[keep])


def lift_depth(depth, pose, intrinsic):
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    fx, fy, cx, cy = intrinsic
    local = np.stack(((x + .5 - cx) / fx * depth,
                      (y + .5 - cy) / fy * depth, depth), -1)
    return local @ pose[:3, :3].T + pose[:3, 3]


def fit_intrinsics(local):
    """Fit Pi3X's predicted rays without adding an inconsistent depth shift."""
    h, w = local.shape[:2]
    y, x = np.mgrid[:h, :w]
    rays = local[..., :2] / np.maximum(local[..., 2:3], 1e-8)
    values, residuals = [], []
    for axis, pixels in ((0, x + .5), (1, y + .5)):
        r = rays[..., axis].ravel()
        p = pixels.ravel()
        valid = np.isfinite(r) & (np.abs(r) < 10)
        # Same all-pixel least-squares solution without allocating an Nx2
        # design matrix and running an SVD for every frame and axis.
        rv, pv = r[valid].astype(np.float64), p[valid].astype(np.float64)
        rm, pm = rv.mean(), pv.mean()
        centered = rv - rm
        variance = np.dot(centered, centered)
        if not np.isfinite(variance) or variance <= 1e-20:
            raise ValueError("degenerate predicted rays")
        slope = np.dot(centered, pv - pm) / variance
        coeff = np.array([slope, pm - slope * rm])
        if coeff[0] <= 0 or not np.isfinite(coeff).all():
            raise ValueError("invalid predicted pinhole intrinsics")
        values.append(coeff)
        residuals.append(float(np.median(np.abs(r[valid] * coeff[0] + coeff[1] - p[valid]))))
    return np.array([values[0][0], values[1][0], values[0][1], values[1][1]]), max(residuals)


def strict_store(path, duration=None):
    import cv2
    from atas_geometry.memory.fuse import _install_depth_reader
    from atas_geometry.memory.vista4d_layers import ReconStore
    _install_depth_reader()
    cap = cv2.VideoCapture(str(Path(path) / "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError("source video has invalid FPS")
    if duration is None:
        duration = max(0, frame_count - 1) / fps
    recon = ReconStore(Path(path), duration=duration, exclude_sky=True, mask_dilate=8)
    counts = [len(recon.frames), len(recon.depth_files), len(recon.dynamic_files),
              len(recon.sky_files), len(recon.cam_c2w), len(recon.intrinsics)]
    if not counts[0] or len(set(counts)) != 1:
        raise ValueError(f"incomplete dense L0: {path}, counts={counts}")
    return recon


def static_depth(recon, index):
    import cv2
    from atas_geometry.memory.vista4d_layers import _mask
    depth = recon.depth_at(index).copy()
    valid = np.isfinite(depth) & (depth > 1e-5)
    bad = recon.dynamic_mask_at(index, depth.shape) | _mask(recon.sky_files[index], depth.shape)
    # Depth boundaries and actor edges are weak geometric anchors.
    lo = cv2.erode(np.where(valid, depth, 0), np.ones((3, 3), np.uint8))
    hi = cv2.dilate(np.where(valid, depth, 0), np.ones((3, 3), np.uint8))
    valid &= ~bad & ((hi - lo) < .04 * np.maximum(depth, 1e-5))
    depth[~valid] = 0
    return depth


def select_history(recon, new, count=6):
    """Retrieve overlap using masked image features; retain temporal diversity."""
    import cv2
    detector = cv2.SIFT_create(nfeatures=1800)
    def desc(store, idx):
        rgb = store.frames[idx]
        mask = (~store.dynamic_mask_at(idx, rgb.shape[:2])).astype(np.uint8) * 255
        return detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)[1]
    probes = [desc(new, int(i)) for i in np.linspace(0, new.count - 1, 3)]
    candidates = np.unique(np.linspace(0, recon.count - 1, min(18, recon.count)).astype(int))
    scores = []
    matcher = cv2.BFMatcher()
    for i in candidates:
        d = desc(recon, int(i))
        score = 0
        if d is not None and len(d) > 1:
            for nd in probes:
                if nd is not None and len(nd) > 1:
                    score += sum(len(p) == 2 and p[0].distance < .72 * p[1].distance
                                 for p in matcher.knnMatch(nd, d, k=2))
        scores.append((score, int(i)))
    chosen = []
    separation = max(1, recon.count // (count * 2))
    for score, i in sorted(scores, reverse=True):
        if all(abs(i - j) >= separation for j in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    for _, i in sorted(scores, reverse=True):
        if len(chosen) >= count:
            break
        if i not in chosen:
            chosen.append(i)
    if len(chosen) < 4:
        raise AnchorFitError("need >=4 distinct history frames for conditioning and validation")
    return sorted(chosen), [{"frame": i, "static_matches": s} for s, i in scores]


def write_exr(path, depth):
    import OpenEXR
    import Imath
    h, w = depth.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    output = OpenEXR.OutputFile(str(path), header)
    try:
        output.writePixels({"Y": np.asarray(depth, np.float32).tobytes()})
    finally:
        output.close()


def overlap_prior_frames(previous_tail, start, end):
    """Alternate reliable overlap views: one for conditioning, one for testing.

    At least one overlap view must remain unseen by the prior inputs. A rejected
    previous chunk is never placed in previous_tail by the caller.
    """
    shared = sorted(i for i in previous_tail if start <= i < end)
    return shared[::2] if len(shared) >= 2 else []


def reconstruct_conditioned(reference, new, output, *, model_id, vista_repo,
                            pixel_limit=255000, chunk_size=24, overlap=4,
                            local_repair=True, repair_from=None):
    import cv2
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(vista_repo))
    from utils.recon_and_seg.recon_pi3 import init_pi3, preprocess_frames_pi3
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite derived L0: {output}")
    output.mkdir(parents=True)
    (output / "depths").mkdir()
    (output / "confidence").mkdir()
    # Share immutable original pixels/masks; never concatenate videos for SAM.
    for name in ("video.mp4", "dynamic_mask", "sky_mask", "entity_masks.json", "entity_masks"):
        src = new.recon_dir / name
        if src.exists():
            (output / name).symlink_to(src.resolve())
    baseline = None
    if repair_from is not None:
        baseline=json.loads((Path(repair_from)/'anchored_reconstruction.json').read_text())
        if (Path(baseline['source_l0']).resolve()!=new.recon_dir.resolve()
                or Path(baseline['reference_l0']).resolve()!=reference.recon_dir.resolve()
                or baseline['frame_count']!=new.count):
            raise ValueError('local repair baseline source/reference/frame count mismatch')
        anchor_ids,retrieval=baseline['anchor_frames'],baseline['retrieval']
    else:
        anchor_ids, retrieval = select_history(reference, new)
    n_anchor = len(anchor_ids)
    # Alternating held-out views receive RGB only (no depth, intrinsics or pose).
    heldout = np.arange(n_anchor) % 3 == 2
    prior = ~heldout
    h, w = new.frames[0].shape[:2]
    if reference.frames[0].shape[:2] != (h, w):
        raise ValueError("history/new RGB resolution differs; explicit calibration resize required")
    anchor_rgb, (ph, pw) = preprocess_frames_pi3(np.stack([reference.frames[i] for i in anchor_ids]), pixel_limit)
    history_depth = []
    history_points = []
    history_k = []
    for i in anchor_ids:
        d = cv2.resize(static_depth(reference, i), (pw, ph), interpolation=cv2.INTER_NEAREST)
        k = reference.intrinsics[i].copy() * np.array([pw/w, ph/h, pw/w, ph/h])
        history_depth.append(d)
        history_points.append(lift_depth(d, reference.cam_c2w[i], k))
        history_k.append(k)
    history_depth, history_points = np.stack(history_depth), np.stack(history_points)
    if any(np.count_nonzero(d) < 100 for d in history_depth):
        raise AnchorFitError("selected history frame has too little valid static depth")
    scene_depth = float(np.median(history_depth[history_depth > 0]))
    # All thresholds are relative to this world's arbitrary reconstruction units.
    anchor_tolerance = .02 * scene_depth
    # Pi3X's output heads explicitly disable autocast and consume float32.
    # Keep master weights float32; mixed precision belongs to the forward pass.
    needs_model=baseline is None or any(not c['accepted_geometry'] for c in baseline['chunks'])
    model = init_pi3(model_id, device="cuda") if needs_model else None
    poses_out = np.zeros((new.count, 4, 4), np.float32)
    k_out = np.zeros((new.count, 4), np.float32)
    owner = np.full(new.count, -1, int)
    reports = []
    if baseline is not None:
        if not np.isclose(anchor_tolerance,baseline['anchor_tolerance'],rtol=1e-6):
            raise ValueError('local repair changed historical tolerance')
        with np.load(Path(repair_from)/'cameras.npz') as data:
            poses_out=data['cam_c2w'].copy();k_out=data['intrinsics'].copy()
        owner=np.asarray(baseline['frame_owner'],int)
        if len(owner)!=new.count or len(poses_out)!=new.count or len(k_out)!=new.count:
            raise ValueError('local repair baseline frame arrays are incomplete')
        reports=baseline['chunks']
        for kind,suffix in [('depths','exr'),('confidence','npy')]:
            for path in (Path(repair_from)/kind).glob('*.'+suffix):
                try:
                    os.link(path,output/kind/path.name)
                except OSError as exc:
                    import errno,shutil
                    if exc.errno!=errno.EXDEV:raise
                    shutil.copy2(path,output/kind/path.name)
        write_json(output/'repair_baseline.json',dict(source=str(Path(repair_from).resolve()),
            geometry_accepted=baseline['geometry_accepted'],chunks=baseline['chunks']))
    previous_tail = {}
    previous_valid = False
    windows=[] if baseline is not None else frame_windows(new.count,chunk_size,overlap)
    for chunk_index, (start, end) in enumerate(windows):
        tick = time.perf_counter()
        rgb, _ = preprocess_frames_pi3(np.stack(new.frames[start:end]), pixel_limit)
        images = torch.cat((anchor_rgb, rgb), 0)[None].to("cuda", dtype=torch.bfloat16)
        n = images.shape[1]
        depths = np.zeros((n, ph, pw), np.float32)
        depths[:n_anchor] = history_depth
        # Held-out depth must not leak into Pi3X's pre-mask normalization.
        depths[np.flatnonzero(heldout)] = 0
        poses = np.repeat(np.eye(4)[None], n, axis=0).astype(np.float32)
        poses[:n_anchor] = reference.cam_c2w[anchor_ids]
        intrinsics = np.repeat(np.eye(3)[None], n, axis=0).astype(np.float32)
        for j, k in enumerate(history_k):
            intrinsics[j] = [[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]
        mask = np.zeros((1, n), bool)
        mask[0, :n_anchor] = prior
        seam_prior_ids = overlap_prior_frames(previous_tail, start, end) if previous_valid else []
        seam_test_ids = sorted(i for i in previous_tail if previous_valid and start <= i < end and i not in seam_prior_ids)
        for frame in seam_prior_ids:
            tail = previous_tail[frame]
            j = n_anchor + frame - start
            depths[j] = np.where(tail['valid'], tail['depth'], 0)
            poses[j] = tail['pose']
            fx, fy, cx, cy = tail['intrinsic']
            intrinsics[j] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            mask[0, j] = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(images,
                depths=torch.from_numpy(depths[None]).to("cuda"),
                poses=torch.from_numpy(poses[None]).to("cuda"),
                intrinsics=torch.from_numpy(intrinsics[None]).to("cuda"),
                mask_add_depth=torch.from_numpy(mask.copy()).to("cuda"),
                mask_add_pose=torch.from_numpy(mask.copy()).to("cuda"),
                mask_add_ray=torch.from_numpy(mask.copy()).to("cuda"), with_prior=True)
        local = prediction["local_points"][0].float().cpu().numpy()
        world = prediction["points"][0].float().cpu().numpy()
        c2w = prediction["camera_poses"][0].float().cpu().numpy()
        confidence = prediction["conf"][0].sigmoid().float().cpu().numpy()[..., 0]
        del prediction, images
        src, dst, test_src, test_dst, test_slots = [], [], [], [], []
        for j in range(n_anchor):
            valid = (history_depth[j] > 0) & np.isfinite(world[j]).all(-1) & (confidence[j] > .1)
            # Sampling is for fitting only. Every new RGB-D frame remains dense.
            ids = np.flatnonzero(valid)[::8]
            s, t = world[j].reshape(-1, 3)[ids], history_points[j].reshape(-1, 3)[ids]
            (test_src if heldout[j] else src).append(s)
            (test_dst if heldout[j] else dst).append(t)
            if heldout[j]:
                test_slots.append((int(anchor_ids[j]), len(s)))
        src, dst, test_src, test_dst = map(np.concatenate, (src, dst, test_src, test_dst))
        try:
            tr = robust_sim3(src, dst)
        except AnchorFitError as exc:
            record = dict(chunk=chunk_index,start=start,end=end,accepted_geometry=False,
                          anchor_fit_error=str(exc),seconds=time.perf_counter()-tick)
            reports.append(record)
            previous_tail={};previous_valid=False
            write_json(output/'progress.json',dict(chunks=reports,completed_frames=int((owner>=0).sum())))
            print(json.dumps(record),flush=True)
            continue
        fit_error = np.linalg.norm(apply_sim3(src, tr) - dst, axis=1)
        test_error = np.linalg.norm(apply_sim3(test_src, tr) - test_dst, axis=1)
        # Per held-out history frame: tells a bad reference frame from a bad new clip.
        heldout_frames, offset = [], 0
        for frame, size in test_slots:
            part = test_error[offset:offset + size]; offset += size
            heldout_frames.append(dict(frame=frame, count=int(size),
                                       median=float(np.median(part)) if size else None))
        valid_chunk = (len(test_error) >= 100 and np.median(test_error) <= anchor_tolerance
                       and np.mean(test_error <= 2 * anchor_tolerance) >= .65)
        scale, rotation, translation = tr
        c2w[:, :3, 3] = apply_sim3(c2w[:, :3, 3], tr)
        c2w[:, :3, :3] = rotation[None] @ c2w[:, :3, :3]
        seam_error = []
        current_tail = {}
        ray_errors = []
        for offset, frame in enumerate(range(start, end)):
            j = n_anchor + offset
            intrinsic, ray_error = fit_intrinsics(local[j])
            ray_errors.append(ray_error)
            world_depth = local[j, ..., 2] * scale
            k_low = intrinsic.copy()
            low_points = lift_depth(world_depth, c2w[j], k_low)
            # Never certify a seam using the exact views supplied as priors.
            if previous_valid and frame in previous_tail and frame not in seam_prior_ids:
                old_points, old_valid = previous_tail[frame]['points'], previous_tail[frame]['valid']
                good = old_valid & (confidence[j] > .2) & np.isfinite(low_points).all(-1)
                if good.sum() > 100:
                    seam_error.append(float(np.median(np.linalg.norm(low_points[good] - old_points[good], axis=-1))))
            if frame >= end - overlap:
                low_static = cv2.resize((static_depth(new, frame) > 0).astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
                current_tail[frame] = dict(points=low_points,
                    valid=low_static & (confidence[j] > .2) & np.isfinite(world_depth) & (world_depth > 0),
                    depth=world_depth, pose=c2w[j].copy(), intrinsic=k_low)
            if owner[frame] >= 0:
                continue
            depth = F.interpolate(torch.from_numpy(world_depth)[None, None], (h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
            write_exr(output / "depths" / f"{frame:05d}.exr", depth)
            np.save(output / "confidence" / f"{frame:05d}.npy", confidence[j].astype(np.float16))
            poses_out[frame] = c2w[j]
            k_out[frame] = intrinsic * np.array([w/pw, h/ph, w/pw, h/ph])
            owner[frame] = chunk_index
        seam_median = float(np.median(seam_error)) if seam_error else None
        if seam_prior_ids and not seam_error:
            valid_chunk = False
        if seam_median is not None and seam_median > anchor_tolerance * 2:
            valid_chunk = False
        if max(ray_errors) > 2.0:
            valid_chunk = False
        previous_tail = current_tail
        previous_valid = bool(valid_chunk)
        record = {"chunk": chunk_index, "start": start, "end": end,
                  "accepted_geometry": bool(valid_chunk), "fit_median": float(np.median(fit_error)),
                  "heldout_median": float(np.median(test_error)), "heldout_p90": float(np.quantile(test_error, .9)),
                  "heldout_within_2t": float(np.mean(test_error <= 2 * anchor_tolerance)),
                  "heldout_frames": heldout_frames,
                  "seam_median": seam_median, "pinhole_residual_pixels": max(ray_errors),
                  "seam_prior_frames": seam_prior_ids,
                  "seam_test_frames": seam_test_ids,
                  "scale": scale, "rotation": rotation, "translation": translation,
                  "seconds": time.perf_counter() - tick}
        reports.append(record)
        write_json(output / "progress.json", {"chunks": reports, "completed_frames": int((owner >= 0).sum())})
        print(json.dumps({k: v for k, v in record.items() if k not in ("rotation", "translation")}), flush=True)
    repair_report = dict(policy='disabled',attempts=[],repaired_chunks=[])
    if local_repair and any(not c['accepted_geometry'] for c in reports):
        from atas_geometry.reconstruct.temporal_chunk_repair import repair_conditioned_chunks
        repair_report = repair_conditioned_chunks(reference=reference,new=new,output=output,
            reports=reports,owner=owner,poses_out=poses_out,k_out=k_out,model=model,
            preprocess=preprocess_frames_pi3,anchor_rgb=anchor_rgb,anchor_ids=anchor_ids,
            history_depth=history_depth,history_points=history_points,history_k=history_k,
            heldout=heldout,tolerance=anchor_tolerance,pixel_limit=pixel_limit)
    np.savez(output / "cameras.npz", cam_c2w=poses_out, intrinsics=k_out)
    cap = cv2.VideoCapture(str(new.recon_dir / "video.mp4"))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
    report = {"method": "pi3x_history_geometry_conditioned_local_repair_v3", "source_l0": str(new.recon_dir),
              "reference_l0": str(reference.recon_dir), "anchor_frames": anchor_ids,
              "conditioned_anchor_slots": np.flatnonzero(prior).tolist(),
              "heldout_anchor_slots": np.flatnonzero(heldout).tolist(), "retrieval": retrieval,
              "frame_count": new.count, "frame_stride": 1, "source_fps": fps,
              "source_indices": list(range(new.count)), "frame_owner": owner.tolist(),
              "source_timestamps_seconds": [i / fps for i in range(new.count)],
              "world_scale": "reference reconstruction units (not calibrated metres)",
              "scene_depth": scene_depth, "anchor_tolerance": anchor_tolerance,
              "resolution": [w, h], "inference_resolution": [pw, ph], "chunks": reports,
              "local_repair": repair_report,
              "unreconstructed_frames": np.flatnonzero(owner<0).tolist(),
              "geometry_accepted": bool(np.all(owner>=0) and all(c["accepted_geometry"] for c in reports)),
              "l2_policy": "all original frames and masks, depth scaled with camera translations exactly once"}
    write_json(output / "anchored_reconstruction.json", report)
    del model
    torch.cuda.empty_cache()
    return report


def validate_cross_shot(reference, new, tolerance, *, matches=None):
    """Check actual old/new static matches in world coordinates, with no refit.

    History self-consistency alone cannot establish that an unrelated new shot
    occupies the right place. This check does not move the new reconstruction.
    """
    from atas_geometry.register.matching import collect_sift_anchors
    # Pair fitting only rejects inconsistent correspondences; the fitted pose
    # is NEVER applied to this observation or to the final residual below.
    # Use the scene's existing error budget, not a hard-coded metre-like unit.
    if matches is None:
        matches = collect_sift_anchors(new, reference, profile="world_scaled_static_v1",world_tolerance=tolerance)
    error = np.linalg.norm(matches.source - matches.target, axis=1)
    n = len(error)
    pairs = len(np.unique(matches.pair_ids))
    fraction = float(np.mean(error <= 2 * tolerance)) if n else 0.
    median = float(np.median(error)) if n else None
    return {"accepted": bool(n >= 30 and pairs >= 2 and median <= tolerance and fraction >= .65),
            "correspondences": n, "frame_pairs": pairs, "median_world_error": median,
            "within_2t_fraction": fraction, "tolerance_world_units": tolerance,
            "method": "static_multiframe_correspondences_in_world_without_refitting",
            "diagnostics": matches.diagnostics,
            "pair_reports": list(matches.pair_reports)}


def historical_visibility(points, views, tolerance):
    """Classify against observed static surfaces, not against occupied 2D rays.

    Behind a historical surface is unobserved/occluded, not a contradiction.
    In front of a confidently observed surface contradicts historical free space.
    """
    surface = np.zeros(len(points), bool)
    free_space = np.zeros(len(points), np.int16)
    for depth, pose, k in views:
        h, w = depth.shape
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        z = local[:, 2]
        u = k[0] * local[:, 0] / np.maximum(z, 1e-8) + k[2]
        v = k[1] * local[:, 1] / np.maximum(z, 1e-8) + k[3]
        valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ix = np.flatnonzero(valid)
        d = depth[v[ix].astype(int), u[ix].astype(int)]
        known = d > 0
        residual = z[ix] - d
        surface[ix[known & (np.abs(residual) <= tolerance)]] = True
        free_space[ix[known & (residual < -tolerance)]] += 1
    return surface, free_space


def fuse_candidate(reference, new, memory, output, report, *, min_bins=2):
    import time
    started = time.perf_counter()
    import cv2
    from scipy.spatial import cKDTree
    from atas_geometry.memory.vista4d_layers import _voxel_keys, mask_area_report
    from atas_geometry.memory.fuse import _memory_from_voxels
    from atas_geometry.memory.gpu_voxel_index import IncrementalKeys, TemporalSupport, unique_first
    from atas_geometry.memory.bounded_neighbors import outside_radius
    output = Path(output)
    from atas_geometry.register.partial_static import fusion_frames, frame_corrections, evidence_bins
    selected_frames = fusion_frames(report, new.count)
    corrections = frame_corrections(report)
    partial_mode = 'partial_static_validation' in report
    if not selected_frames:
        return {"merged": False, "reason": "history_or_chunk_consistency_failed", "new_points": 0}
    tree = cKDTree(memory.xyz)
    voxel, origin = float(memory.voxel_size), memory.voxel_origin
    tolerance = max(3 * voxel, min(report["anchor_tolerance"],
                    2 * max(c["heldout_median"] for c in report["chunks"] if c.get('accepted_geometry'))))
    views = [(static_depth(reference, i), reference.cam_c2w[i], reference.intrinsics[i])
             for i in report["anchor_frames"]]
    seen = IncrementalKeys(_voxel_keys(memory.xyz, origin, voxel))
    coarse_tree = None
    coarse_error = 0.
    if partial_mode:
        cell_size = max(voxel, tolerance / 4)
        representatives, _ = unique_first(_voxel_keys(memory.xyz, origin, cell_size), seen.device)
        coarse_tree = cKDTree(memory.xyz[representatives])
        coarse_error = np.sqrt(3.) * cell_size * (1 + 1e-6)
    from atas_geometry.memory.fusion_projection import Projection
    projection = Projection(views, seen.device)
    support_size = max(voxel * 4, tolerance)
    support = TemporalSupport(device=seen.device)
    output.mkdir(parents=True,exist_ok=True)
    parts_dir=output/'fusion_parts';parts_dir.mkdir(exist_ok=True)
    parts, colors, cell_parts = [], [], []
    totals = dict(static_pixels=0, low_confidence=0, duplicate=0, historical_surface=0, free_space_conflict=0)
    excluded = set(mask_area_report(new)["excluded_frame_indices"])
    setup_seconds = time.perf_counter() - started
    stages = dict(lift=0., unique=0., nearest=0., visibility=0., incremental_index=0., checkpoint=0.)
    # Three evidence intervals belong to the usable observation, not to frames
    # already rejected for bad coordinates. Preserve the two-bin support rule.
    usable_frames = [i for i in selected_frames if i not in excluded]
    bins = evidence_bins(usable_frames, new.count, partial=partial_mode)
    for i in usable_frames:
        if i in excluded:
            continue
        tick = time.perf_counter()
        depth = static_depth(new, i)
        conf = np.load(new.recon_dir / "confidence" / f"{i:05d}.npy").astype(np.float32)
        conf = cv2.resize(conf, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = depth > 0
        totals["static_pixels"] += int(valid.sum())
        totals["low_confidence"] += int((valid & (conf < .2)).sum())
        valid &= conf >= .2
        pose = new.cam_c2w[i]
        if i in corrections:
            scale, rotation, translation = corrections[i]
            pose = pose.copy()
            pose[:3,3] = scale * rotation @ pose[:3,3] + translation
            pose[:3,:3] = rotation @ pose[:3,:3]
            depth = depth * scale
        xyz = projection.lift(depth, pose, new.intrinsics[i], valid)
        rgb = new.frames[i][valid]
        stages['lift'] += time.perf_counter() - tick
        tick = time.perf_counter()
        unique, _ = unique_first(_voxel_keys(xyz, origin, voxel),seen.device)
        xyz, rgb = xyz[unique], rgb[unique]
        stages['unique'] += time.perf_counter() - tick
        tick = time.perf_counter()
        surface, conflict = projection.visibility(xyz, tolerance)
        totals["historical_surface"] += int(surface.sum())
        totals["free_space_conflict"] += int((conflict >= 2).sum())
        keep = ~surface & (conflict < 2)
        xyz, rgb = xyz[keep], rgb[keep]
        stages['visibility'] += time.perf_counter() - tick
        if not len(xyz):
            continue
        # Identical conjunction of predicates, cheaper order: discard known
        # surfaces/conflicts on GPU before asking the CPU tree about novelty.
        tick = time.perf_counter()
        # A registration-uncertainty-sized neighbourhood of an existing world
        # surface is overlap, not an automatically novel parallel wall.
        keep = outside_radius(tree,xyz,max(voxel*2,tolerance),
            coarse_tree=coarse_tree,coarse_error=coarse_error)
        totals['duplicate'] += int((~keep).sum())
        xyz, rgb = xyz[keep], rgb[keep]
        stages['nearest'] += time.perf_counter() - tick
        if not len(xyz):
            continue
        tick = time.perf_counter()
        cell_keys = _voxel_keys(xyz, origin, support_size)
        bin_id = bins[i]
        support.add(cell_keys,bin_id)
        keys = _voxel_keys(xyz, origin, voxel)
        fresh = ~seen.contains(keys)
        if fresh.any():
            parts.append(xyz[fresh]); colors.append(rgb[fresh]); cell_parts.append(cell_keys[fresh])
            seen.add(keys[fresh])
            stages['incremental_index'] += time.perf_counter() - tick
            tick = time.perf_counter()
            # Preserve computed candidates before final capacity/I/O work.
            # These are unaccepted parts, not visible world revisions.
            part=parts_dir/f'{i:05d}.npz'
            temporary=part.with_suffix('.partial.npz')
            np.savez(temporary,xyz=xyz[fresh],rgb=rgb[fresh],cells=cell_keys[fresh])
            os.replace(temporary,part)
            stages['checkpoint'] += time.perf_counter() - tick
        else:
            stages['incremental_index'] += time.perf_counter() - tick
        if i % 20 == 0:
            print(f"fusion frame {i}/{new.count}, candidate voxels {sum(map(len, parts))}", flush=True)
            write_json(output/'fusion_progress.json',{'frame':i,'total_frames':new.count,
                'candidate_points':sum(map(len,parts)),'published':False,
                'index':'gpu_incremental_voxels_v2','device':str(seen.device)})
    if not parts:
        return {"merged": False, "reason": "no_supported_novel_surface", "new_points": 0, "counts": totals}
    finalize_started = time.perf_counter()
    xyz, rgb, cells = np.concatenate(parts), np.concatenate(colors), np.concatenate(cell_parts)
    keep = support.mask(cells,min_bins=min_bins)
    xyz, rgb = xyz[keep], rgb[keep]
    if not len(xyz):
        return {"merged": False, "reason": "no_temporally_supported_novel_surface", "new_points": 0, "counts": totals}
    # Persistent world size is not a GPU residency limit. The renderer streams
    # large worlds through one z-buffer; never discard a valid computed scene.
    added = _memory_from_voxels(xyz, rgb, voxel_size=voxel, origin=origin, shot_id=new.recon_dir.parent.name)
    candidate = append_static(memory, added)
    if not np.array_equal(candidate.xyz[:len(memory.xyz)], memory.xyz):
        raise RuntimeError("append-only invariant violated")
    candidate.save(output / "l1_candidate.partial.npz", metadata={"method": "anchored_surface_incremental_v2", "official_l1_unchanged": True})
    os.replace(output / 'l1_candidate.partial.npz', output / 'l1_candidate.npz')
    np.savez(output / "new_static_points.npz", xyz=xyz, rgb=rgb)
    return {"merged": bool(len(xyz)), "new_points": len(xyz), "before_points": len(memory.xyz),
            "after_points": len(candidate.xyz), "candidate": str(output / "l1_candidate.npz"),
            "counts": totals, "temporal_bins": 3, "min_bins": min_bins,
            "temporal_support_scope": "reliable_frames" if partial_mode else "whole_clip",
            "world_overlap_radius": max(voxel*2,tolerance) if partial_mode else voxel*2,
            "excluded_mask_frames": sorted(excluded), "overlap_tolerance_world_units": tolerance,
            "fine_voxel_size": voxel, "append_only_prefix_verified": True,
            "timing_seconds": dict(total=time.perf_counter()-started, setup=setup_seconds,
                     finalize=time.perf_counter()-finalize_started, **stages),
            "index_backend": str(seen.device), "points_discarded_for_budget": 0}


def append_static(memory, added):
    """Preserve all old fields/provenance; novel points already passed dedup."""
    fields = ("xyz", "rgb", "world_time", "confidence", "foreground", "weight",
              "point_count", "source_count", "conflict_score")
    values = {name: np.concatenate((getattr(memory, name), getattr(added, name))) for name in fields}
    values["provenance_offsets"] = np.concatenate((memory.provenance_offsets,
            added.provenance_offsets[1:] + memory.provenance_offsets[-1]))
    values["provenance_input_indices"] = np.concatenate((memory.provenance_input_indices, added.provenance_input_indices))
    values["camera_ids"] = memory.camera_ids + added.camera_ids
    return replace(memory, **values)


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # resident servers reuse the process
    p = argparse.ArgumentParser()
    p.add_argument("--reference-l0", type=Path, required=True)
    p.add_argument("--new-l0", type=Path, required=True)
    p.add_argument("--l1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vista-repo", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--chunk-size", type=int, default=24)
    p.add_argument("--reuse-reconstruction", action="store_true")
    p.add_argument("--partial-static", action="store_true", help="Validate blocks separately; emit an L1 candidate only, without certifying whole-clip L2")
    p.add_argument("--no-local-repair", action="store_true", help="Baseline comparison only; keep acceptance thresholds unchanged")
    p.add_argument("--repair-conditioned-l0", type=Path, help="Repair only rejected intervals of an existing derived L0 into a new output")
    p.add_argument("--reuse-conditioned-l0", type=Path,
                   help="Read an existing dense reconstruction without altering it; fit/fuse into a new output")
    a = p.parse_args()
    if a.repair_conditioned_l0 and (a.reuse_conditioned_l0 or a.reuse_reconstruction or a.no_local_repair):
        p.error('repair-conditioned-l0 cannot be combined with reuse or no-local-repair')
    from atas_geometry.memory.fusion import FusedSceneMemory
    a.output.mkdir(parents=True, exist_ok=True)
    before_hash = sha256(a.l1)
    reference, new = strict_store(a.reference_l0), strict_store(a.new_l0)
    output_l0 = a.reuse_conditioned_l0 or a.output / "conditioned_l0"
    try:
        report = (json.loads((output_l0 / "anchored_reconstruction.json").read_text()) if a.reuse_reconstruction or a.reuse_conditioned_l0
                  else reconstruct_conditioned(reference, new, output_l0, model_id=a.model_id,
                                                vista_repo=a.vista_repo, chunk_size=a.chunk_size,
                                                local_repair=not a.no_local_repair,
                                                repair_from=a.repair_conditioned_l0))
    except AnchorFitError as exc:
        # A deterministic reference rejection must reach bounded alternate-
        # reference recovery, not masquerade as an interrupted GPU process.
        # Partial dense outputs are retained but NEVER certified or published.
        write_json(a.output/'fusion_report.json',dict(merged=False,new_points=0,
            reason='anchor_fit_failed',error=str(exc),source_l0=str(a.new_l0),
            reference_l0=str(a.reference_l0),original_l1_sha256=before_hash,
            original_l1_unchanged=sha256(a.l1)==before_hash,
            reconstruction=dict(geometry_accepted=False,chunks=[],error=str(exc)),
            cross_shot_validation=dict(accepted=False,status='not_run',reason='anchor_fit_failed')))
        return
    if (Path(report['source_l0']).resolve() != a.new_l0.resolve()
            or Path(report['reference_l0']).resolve() != a.reference_l0.resolve()):
        raise ValueError('conditioned reconstruction source/reference mismatch')
    if a.partial_static:
        from atas_geometry.register.partial_static import validate_blocks
        tick = time.perf_counter()
        derived = strict_store(output_l0, new.duration)
        partial = validate_blocks(reference, derived, report)
        validation_seconds = time.perf_counter() - tick
        report = dict(report, partial_static_validation=partial)
        tick = time.perf_counter()
        fusion = fuse_candidate(reference, derived, FusedSceneMemory.load(a.l1), a.output, report) if partial['accepted'] else dict(
            merged=False, new_points=0, reason='no_independently_registered_block')
        fusion.update(method='partial_static_candidate_v1', source_l0=str(a.new_l0), reference_l0=str(a.reference_l0),
            original_l1=str(a.l1), original_l1_sha256=before_hash, original_l1_unchanged=sha256(a.l1)==before_hash,
            conditioned_l0=str(output_l0), reconstruction=report,
            partial_static_validation=partial, static_validation_seconds=validation_seconds,
            fusion_wall_seconds=time.perf_counter()-tick,
            cross_shot_validation=dict(accepted=False, status='partial_static_only'),
            publication_status='offline_candidate', whole_clip_recapture_accepted=False)
        write_json(a.output/'fusion_report.json', fusion)
        print(json.dumps({k:v for k,v in fusion.items() if k not in ('reconstruction','partial_static_validation')}), flush=True)
        return
    if not report['geometry_accepted']:
        # Cross-shot alignment cannot repair a broken internal reconstruction.
        # Do not read the multi-million-point world or repeat SIFT on this path.
        cross = dict(accepted=False,status='not_run',reason='internal_reconstruction_inconsistent')
        write_json(a.output/'cross_shot_validation.json',cross)
        fusion=dict(merged=False,new_points=0,reason='internal_reconstruction_inconsistent',
            method='pi3x_history_conditioned_fusion_v1',source_l0=str(a.new_l0),
            reference_l0=str(a.reference_l0),original_l1=str(a.l1),original_l1_sha256=before_hash,
            original_l1_unchanged=sha256(a.l1)==before_hash,conditioned_l0=str(output_l0),
            reconstruction=report,cross_shot_validation=cross)
        write_json(a.output/'fusion_report.json',fusion)
        write_json(a.output/'pending_submap.json',dict(status='pending_internal_reconstruction',
            source_l0=str(a.new_l0),conditioned_l0=str(output_l0),
            source_video_sha256=sha256(a.new_l0/'video.mp4'),reference_l0=str(a.reference_l0),
            world_sha256=before_hash,fusion_report=str(a.output/'fusion_report.json'),
            frame_count=report.get('frame_count',new.count),published=False,requires_new_alignment_evidence=True))
        return
    derived = strict_store(output_l0, new.duration)
    from atas_geometry.register.matching import collect_sift_anchors
    matches = collect_sift_anchors(derived, reference, profile="world_scaled_static_v1",
                                   world_tolerance=report["anchor_tolerance"])
    if len(matches.source) < 30:
        pooled = collect_sift_anchors(derived, reference, profile='pooled_static_v1')
        source_frames = {p['src_index'] for p in pooled.pair_reports}
        if len(pooled.source) >= 30 and len(source_frames) >= 3:
            matches = pooled
    cross = validate_cross_shot(reference, derived, report["anchor_tolerance"], matches=matches)
    if matches.diagnostics.get('profile') == 'pooled_static_v1':
        cross['method'] = 'pooled_static_multiframe_fixed_world_validation_no_pairwise_refit'
    if not cross['accepted'] and report['geometry_accepted']:
        from atas_geometry.register.alignment_refinement import propose_correction, materialize_correction
        tr, correction = propose_correction(matches, report['anchor_tolerance'])
        write_json(a.output/'alignment_refinement.json', correction)
        if tr is not None:
            original_cross = cross
            report = materialize_correction(derived, a.output/'refined_l0', tr, report, correction)
            output_l0 = a.output/'refined_l0'
            derived = strict_store(output_l0, new.duration)
            # Transform the cached correspondences too. No repeated SIFT/reconstruction.
            from dataclasses import replace as dc_replace
            corrected_matches = dc_replace(matches, source=apply_sim3(matches.source, tr))
            cross = validate_cross_shot(reference, derived, report['anchor_tolerance'], matches=corrected_matches)
            cross.update(method='bounded_correction_with_disjoint_frame_validation',
                         before_correction=original_cross, refinement=correction)
    write_json(a.output / "cross_shot_validation.json", cross)
    fusion = (fuse_candidate(reference, derived, FusedSceneMemory.load(a.l1), a.output, report) if cross["accepted"]
              else {"merged": False, "reason": "cross_shot_static_alignment_failed", "new_points": 0})
    fusion.update(method="pi3x_history_conditioned_fusion_v1", source_l0=str(a.new_l0),
                  reference_l0=str(a.reference_l0), original_l1=str(a.l1),
                  original_l1_sha256=before_hash, original_l1_unchanged=sha256(a.l1) == before_hash,
                  conditioned_l0=str(output_l0), reconstruction=report, cross_shot_validation=cross)
    import torch
    if torch.cuda.is_available():
        fusion["gpu_reserved_peak_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    write_json(a.output / "fusion_report.json", fusion)
    if not cross['accepted'] or not report['geometry_accepted']:
        write_json(a.output/'pending_submap.json', dict(status='pending_alignment',
            source_l0=str(a.new_l0), conditioned_l0=str(output_l0),
            source_video_sha256=sha256(a.new_l0/'video.mp4'), reference_l0=str(a.reference_l0),
            world_sha256=before_hash, fusion_report=str(a.output/'fusion_report.json'),
            frame_count=derived.count, published=False, requires_new_alignment_evidence=True))
    print(json.dumps({k:v for k,v in fusion.items() if k != "reconstruction"}), flush=True)


if __name__ == "__main__":
    main()
