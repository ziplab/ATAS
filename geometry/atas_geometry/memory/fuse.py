from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from atas_geometry.memory.fusion import FusedSceneMemory, MemoryPointBatch, _umeyama, fuse_memory_points
from atas_geometry.memory.vista4d_layers import (
    ReconStore,
    _voxel_keys,
    mask_area_report,
    pixel_footprint_voxel_size,
)
from atas_geometry.service.paths import remember_scene_shot
MASK_ROOM_RATIO = 0.85
GROWTH_FAIL_RATIO = 2.0
# World capacity is not a render working-set budget. Render residency/chunk
# budgets live in render.py; geometry quality gates below remain unchanged.
HOME_K = 2.0
ALIGN_SCALE_MIN = 0.5
ALIGN_SCALE_MAX = 2.0
NOVEL_FRAC_SKIP = 0.50
CLOSEUP_MASK_SKIP = 0.40
TRUSTED_ALIGN_MIN_INLIERS = 100
TRUSTED_ALIGN_MAX_MEDIAN_ERR = 0.03


def _trusted_alignment(align_info: dict[str, Any] | None) -> bool:
    """Strong SIFT registration proves same-room, so high novelty is real new
    coverage (the thing we want to fuse), not a misaligned clip. The novelty
    and growth gates guard weak SIFT alignments that cleared the floor."""
    if not align_info or align_info.get("method") != "sift":
        return False
    median_err = align_info.get("median_err")
    if median_err is None:
        return False
    return (
        int(align_info.get("inliers") or 0) >= TRUSTED_ALIGN_MIN_INLIERS
        and float(median_err) <= TRUSTED_ALIGN_MAX_MEDIAN_ERR
        and ALIGN_SCALE_MIN <= float(align_info["scale"]) <= ALIGN_SCALE_MAX
    )


def find_previous_l0(current_l0: Path, l1_path: Path) -> Path | None:
    """Last L0 recon'd into **this scene**, not the previous clip on the timeline.

    shot_10 revisiting a palace must SIFT against the last palace shot (or that
    room's Home), never the bedroom that happened to be shot_09.
    """
    shots_root = current_l0.parent.parent
    current_shot = current_l0.parent.name
    scene = l1_path.parent
    if not shots_root.is_dir():
        return None
    ids: list[str] = []
    home = ""
    home_marker = scene / "home_shot_id.txt"
    if home_marker.is_file():
        home = home_marker.read_text(encoding="utf-8").strip()
        if home:
            ids.append(home)
    log = scene / "shot_ids.txt"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            sid = line.strip()
            if sid and sid not in ids:
                ids.append(sid)
    prev_aligned: Path | None = None
    prev_any: Path | None = None
    for sid in ids:
        if sid == current_shot:
            continue
        candidate = shots_root / sid / "l0"
        if not (candidate / "cameras.npz").is_file():
            continue
        prev_any = candidate
        if sid == home or (candidate / "cameras_pi3_raw.npz").is_file():
            prev_aligned = candidate
    return prev_aligned or prev_any


def apply_sim3(
    xyz: np.ndarray, *, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return (xyz.astype(np.float64) * scale) @ rotation.T + translation


def transform_c2w(
    c2w: np.ndarray, *, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[:, :3, :3] = rotation @ out[:, :3, :3]
    out[:, :3, 3] = (out[:, :3, 3] * scale) @ rotation.T + translation
    return out.astype(np.float32)


def write_aligned_cameras(l0_dir: Path, aligned_c2w: np.ndarray) -> None:
    path = l0_dir / "cameras.npz"
    raw = l0_dir / "cameras_pi3_raw.npz"
    data = dict(np.load(path))
    if not raw.is_file():
        np.savez(raw, **data)
    data["cam_c2w"] = aligned_c2w.astype(np.float32)
    np.savez(path, **data)


def _align_indices(count: int) -> list[int]:
    if count <= 5:
        return list(range(count))
    return sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})


def _lift_uv(recon: ReconStore, index: int, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = recon.depth_at(index)
    height, width = depth.shape[:2]
    us = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, width - 1)
    vs = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, height - 1)
    z = depth[vs, us].astype(np.float64)
    fx, fy, cx, cy = [float(value) for value in recon.intrinsics[index, :4]]
    valid = np.isfinite(z) & (z > 1e-5)
    points_cam = np.stack(
        [(us - cx) / max(fx, 1e-6) * z, (vs - cy) / max(fy, 1e-6) * z, z],
        axis=1,
    )
    world = points_cam @ recon.cam_c2w[index, :3, :3].T + recon.cam_c2w[index, :3, 3]
    return world, valid


def _ransac_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.08,
    iters: int = 400,
):
    rng = np.random.default_rng(1)
    count = int(source.shape[0])
    best_inliers = 0
    best_median = 1e9
    best = None
    for _ in range(iters):
        sample = rng.choice(count, size=3, replace=False)
        try:
            transform = _umeyama(source[sample], target[sample])
        except ValueError:
            continue
        error = np.linalg.norm(transform.apply(source) - target, axis=1)
        inliers = int(np.count_nonzero(error < threshold))
        median = float(np.median(error))
        if inliers > best_inliers or (inliers == best_inliers and median < best_median):
            best_inliers = inliers
            best_median = median
            best = transform
    if best is None:
        raise RuntimeError("Sim3 RANSAC failed")
    error = np.linalg.norm(best.apply(source) - target, axis=1)
    keep = error < threshold * 1.5
    if int(np.count_nonzero(keep)) >= 8:
        best = _umeyama(source[keep], target[keep])
        error = np.linalg.norm(best.apply(source) - target, axis=1)
    inlier_mask = error < threshold
    return best, inlier_mask, error


def estimate_sim3_sift(
    new_recon: ReconStore,
    prev_recon: ReconStore,
) -> dict[str, Any] | None:
    import cv2

    new_ids = _align_indices(new_recon.count)
    prev_ids = _align_indices(prev_recon.count)
    detector = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def describe(recon: ReconStore, index: int):
        gray = cv2.cvtColor(np.asarray(recon.frames[index]), cv2.COLOR_RGB2GRAY)
        return detector.detectAndCompute(gray, None)

    new_feat = {index: describe(new_recon, index) for index in new_ids}
    prev_feat = {index: describe(prev_recon, index) for index in prev_ids}
    best: dict[str, Any] | None = None
    for src_index, (src_kp, src_desc) in new_feat.items():
        if src_desc is None or len(src_kp) < 12:
            continue
        for dst_index, (dst_kp, dst_desc) in prev_feat.items():
            if dst_desc is None or len(dst_kp) < 12:
                continue
            knn = matcher.knnMatch(src_desc, dst_desc, k=2)
            good = [
                pair[0]
                for pair in knn
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
            ]
            if len(good) < 16:
                continue
            src_uv = np.array([src_kp[match.queryIdx].pt for match in good], dtype=np.float64)
            dst_uv = np.array([dst_kp[match.trainIdx].pt for match in good], dtype=np.float64)
            src_xyz, src_valid = _lift_uv(new_recon, src_index, src_uv)
            dst_xyz, dst_valid = _lift_uv(prev_recon, dst_index, dst_uv)
            keep = src_valid & dst_valid
            if int(np.count_nonzero(keep)) < 16:
                continue
            transform, inlier_mask, error = _ransac_umeyama(src_xyz[keep], dst_xyz[keep])
            inliers = int(np.count_nonzero(inlier_mask))
            median = float(np.median(error))
            pair = {
                "scale": float(transform.scale),
                "rotation": np.asarray(transform.rotation, dtype=np.float64),
                "translation": np.asarray(transform.translation, dtype=np.float64),
                "inliers": inliers,
                "matches": int(np.count_nonzero(keep)),
                "median_err": median,
                "src_index": int(src_index),
                "dst_index": int(dst_index),
                "method": "sift",
            }
            better = best is None or inliers > int(best["inliers"]) or (
                inliers == int(best["inliers"]) and median < float(best["median_err"])
            )
            if better:
                best = pair
    if best is None:
        return None
    ratio = float(best["inliers"]) / max(int(best["matches"]), 1)
    if int(best["inliers"]) < 30 or ratio < 0.3 or float(best["median_err"]) > 0.2:
        return None
    return best


def estimate_sim3(
    new_recon: ReconStore,
    previous_l0: Path,
    *,
    duration: float,
) -> dict[str, Any] | None:
    """SIFT-only alignment. No camera_continue heuristic: a guessed transform
    writes points into the wrong part of the room, which is worse than not
    fusing at all. Chaining no longer depends on this -- it uses our own
    rendered last-frame poses, which are already in the scene frame."""
    prev_recon = ReconStore(previous_l0, duration=float(duration), exclude_sky=True)
    return estimate_sim3_sift(new_recon, prev_recon)


def _align_public(info: dict[str, Any] | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return {
        "method": info.get("method"),
        "scale": float(info["scale"]),
        "inliers": int(info.get("inliers") or 0),
        "matches": int(info.get("matches") or 0),
        "median_err": info.get("median_err"),
        "src_index": info.get("src_index"),
        "dst_index": info.get("dst_index"),
    }


def _write_chain_cameras_if_trusted(
    l0_dir: Path,
    recon: ReconStore,
    align_info: dict[str, Any] | None,
) -> None:
    """Persist the aligned cameras when a SIFT alignment is trustworthy.

    Merge may be skipped (close-up, mostly-novel frame) while the alignment
    itself is still good; writing the aligned cameras keeps this L0 usable as
    a chained continue start even though it never entered L1.
    """
    if not _trusted_alignment(align_info):
        return
    scale = float(align_info["scale"])
    write_aligned_cameras(
        l0_dir,
        transform_c2w(
            recon.cam_c2w,
            scale=scale,
            rotation=np.asarray(align_info["rotation"], dtype=np.float64),
            translation=np.asarray(align_info["translation"], dtype=np.float64),
        ),
    )


def _read_depth_any(path: Path) -> np.ndarray:
    try:
        imath = importlib.import_module("Imath")
        openexr = importlib.import_module("OpenEXR")
        pixel_type = cast(Any, imath).PixelType(cast(Any, imath).PixelType.FLOAT)
        file = cast(Any, openexr).InputFile(str(path))
        try:
            data_window = file.header()["dataWindow"]
            width = data_window.max.x - data_window.min.x + 1
            height = data_window.max.y - data_window.min.y + 1
            header = file.header()
            channels = header["channels"]
            name = "Y" if "Y" in channels else next(iter(channels))
            channel = file.channel(name, pixel_type)
            return np.frombuffer(channel, dtype=np.float32).reshape(height, width)
        finally:
            file.close()
    except Exception:
        import imageio.v3 as iio

        depth = iio.imread(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return np.asarray(depth, dtype=np.float32)


def _install_depth_reader() -> None:
    import atas_geometry.memory.vista4d_layers as layers

    layers._read_exr_depth = _read_depth_any


def _select_keyframes(c2w: np.ndarray, voxel_size: float) -> list[int]:
    count = int(c2w.shape[0])
    if count <= 2:
        return list(range(count))
    chosen = [0]
    last = c2w[0]
    min_move = max(float(voxel_size) * 8.0, 0.03)
    min_deg = 5.0
    for index in range(1, count):
        current = c2w[index]
        travel = float(np.linalg.norm(current[:3, 3] - last[:3, 3]))
        cosine = float(np.clip(np.sum(current[:3, 2] * last[:3, 2]), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        if travel >= min_move or angle >= min_deg:
            chosen.append(index)
            last = current
    mid = count // 2
    if mid not in chosen:
        chosen.append(mid)
    if chosen[-1] != count - 1:
        chosen.append(count - 1)
    return sorted(set(chosen))


def _unique_voxels(
    xyz: np.ndarray,
    rgb: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return xyz, rgb, empty_i
    keys = _voxel_keys(xyz, origin, voxel_size)
    packed = torch.from_numpy(np.asarray(keys, dtype=np.int64)).to(device)
    inverse = packed.new_zeros(packed.shape[0], dtype=torch.long)
    uniq, inverse = torch.unique(packed, return_inverse=True)
    order = torch.arange(packed.shape[0], device=device)
    first = torch.full((uniq.shape[0],), packed.shape[0], dtype=torch.long, device=device)
    first.scatter_reduce_(0, inverse, order, reduce="amin", include_self=True)
    keep = first.detach().cpu().numpy()
    uniq_np = uniq.detach().cpu().numpy()
    return xyz[keep], rgb[keep], uniq_np


def _memory_from_voxels(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    voxel_size: float,
    origin: np.ndarray,
    shot_id: str,
) -> FusedSceneMemory:
    count = int(xyz.shape[0])
    batch = MemoryPointBatch(
        xyz=xyz.astype(np.float32),
        rgb=rgb.astype(np.uint8),
        world_time=np.zeros(count, dtype=np.float32),
        confidence=np.full(count, 0.7, dtype=np.float32),
        input_index=np.arange(count, dtype=np.int32),
        camera_index=np.zeros(count, dtype=np.int16),
        foreground=np.zeros(count, dtype=np.float32),
        camera_ids=(shot_id,),
    )
    return fuse_memory_points(batch, voxel_size=voxel_size, time_bin_size=1.0, voxel_origin=origin)


def _skip_merge_existing(
    *,
    existing: FusedSceneMemory,
    l0_dir: Path,
    align_info: dict[str, Any] | None,
    first_novel_frac: float | None,
    median: float,
    extra: str,
) -> dict[str, Any]:
    return {
        "l1_point_count": int(existing.xyz.shape[0]),
        "voxel_size": float(existing.voxel_size),
        "merged": False,
        "skip_merge": extra,
        "static_pixels_seen": 0,
        "novel_points": 0,
        "mask_area_median": median,
        "align_scale": float(align_info["scale"]) if align_info else 1.0,
        "previous_l0": None,
        "first_novel_frac": first_novel_frac,
        "align": _align_public(align_info),
        "l0_dir": str(l0_dir),
    }


def _static_frame_points(recon: ReconStore, index: int) -> tuple[np.ndarray, np.ndarray]:
    batch = recon.unproject(index)
    keep = batch.foreground < 0.5
    xyz = np.asarray(batch.xyz[keep])
    rgb = np.asarray(batch.rgb[keep])
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    if xyz.shape[0] == 0:
        return xyz, rgb
    cam = recon.cam_c2w[index, :3, 3]
    dist = np.linalg.norm(xyz - cam, axis=1)
    near = float(np.percentile(dist, 20.0))
    keep_near = dist < max(near * 25.0, 1.0)
    return xyz[keep_near], rgb[keep_near]


def _accumulate_home_voxels(
    recon: ReconStore,
    *,
    voxel: float,
    frames: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    origin: np.ndarray | None = None
    acc_xyz: list[np.ndarray] = []
    acc_rgb: list[np.ndarray] = []
    from atas_geometry.memory.gpu_voxel_index import IncrementalKeys
    acc_keys = IncrementalKeys(device=device)
    static_pixels = 0
    for index in frames:
        xyz, rgb = _static_frame_points(recon, index)
        static_pixels += int(xyz.shape[0])
        if xyz.shape[0] == 0:
            continue
        if origin is None:
            origin = np.floor(np.percentile(xyz.astype(np.float64), 1.0, axis=0) / voxel) * voxel
        uniq_xyz, uniq_rgb, keys = _unique_voxels(xyz, rgb, origin, voxel, device)
        # Exact membership on the assigned device. _unique_voxels already
        # retains each voxel's first sample; this mask preserves its order and
        # the earliest frame's RGB without per-point Python objects.
        novel = ~acc_keys.contains(keys)
        if not np.any(novel):
            continue
        acc_xyz.append(uniq_xyz[novel])
        acc_rgb.append(uniq_rgb[novel])
        acc_keys.add(keys[novel])
    if origin is None or not acc_xyz:
        raise RuntimeError("no static points to fuse for Home L1")
    return np.concatenate(acc_xyz, axis=0), np.concatenate(acc_rgb, axis=0), origin, static_pixels


def _fuse_home_scene(
    recon: ReconStore,
    l1_path: Path,
    *,
    shot_id: str,
    report: dict[str, Any],
    gpu_index: int,
) -> dict[str, Any]:
    """One GPU pass at HOME_K on keyframes. No k=1.0 numpy sweep of every frame."""
    skipped = {int(index) for index in report.get("excluded_frame_indices") or ()}
    from atas_geometry.service.device_mapping import device_index
    device = torch.device(f"cuda:{device_index(gpu_index)}")
    torch.cuda.set_device(device)
    chosen_k = HOME_K
    voxel = float(pixel_footprint_voxel_size(recon, k=chosen_k))
    frames = [index for index in _select_keyframes(recon.cam_c2w, voxel) if index not in skipped]
    new_xyz, new_rgb, origin, static_pixels = _accumulate_home_voxels(
        recon, voxel=voxel, frames=frames, device=device
    )
    memory = _memory_from_voxels(
        new_xyz, new_rgb, voxel_size=voxel, origin=origin, shot_id=shot_id
    )
    l1_path.parent.mkdir(parents=True, exist_ok=True)
    temporary=l1_path.with_name(l1_path.stem+".partial.npz")
    memory.save(temporary)
    temporary.replace(l1_path)
    return {
        "l1_point_count": int(memory.xyz.shape[0]),
        "voxel_size": float(memory.voxel_size),
        "merged": True,
        "home": True,
        "k": chosen_k,
        "keyframes": frames,
        "static_pixels_seen": static_pixels,
        "novel_points": int(memory.xyz.shape[0]),
        "mask_area_median": float(report["mask_area_median"]),
        "align_scale": 1.0,
        "previous_l0": None,
        "first_novel_frac": None,
        "align": None,
        "gpu_index": gpu_index,
    }


def fuse_l0_to_l1(
    l0_dir: Path,
    l1_path: Path,
    *,
    duration: float,
    shot_id: str,
    gpu_index: int,
) -> dict[str, Any]:
    _install_depth_reader()
    recon = ReconStore(l0_dir, duration=float(duration), exclude_sky=True)
    if recon.count <= 0:
        raise RuntimeError(f"empty recon: {l0_dir}")
    remember_scene_shot(l1_path.parent, shot_id)
    report = mask_area_report(recon)
    median = float(report["mask_area_median"])
    if median >= MASK_ROOM_RATIO:
        raise RuntimeError(
            f"dynamic mask paints the room (median={median:.2f}); refusing L1 writeback"
        )
    if not l1_path.is_file():
        return _fuse_home_scene(
            recon, l1_path, shot_id=shot_id, report=report, gpu_index=gpu_index
        )
    voxel_size = float(pixel_footprint_voxel_size(recon, k=1.5))
    existing: FusedSceneMemory | None = None
    origin: np.ndarray | None = None
    existing_keys: set[int] = set()
    overlap_voxel = voxel_size
    if l1_path.is_file():
        existing = FusedSceneMemory.load(l1_path)
        origin = np.asarray(existing.voxel_origin, dtype=np.float64)
        voxel_size = float(existing.voxel_size)
        overlap_voxel = max(voxel_size * 16.0, 0.06)
        existing_keys = set(
            _voxel_keys(np.asarray(existing.xyz), origin, overlap_voxel).tolist()
        )
    assert existing is not None
    raw_cameras = l0_dir / "cameras_pi3_raw.npz"
    if raw_cameras.is_file():
        recon.cam_c2w = np.load(raw_cameras)["cam_c2w"].astype(np.float64)
    align_scale = 1.0
    align_rotation = np.eye(3, dtype=np.float64)
    align_translation = np.zeros(3, dtype=np.float64)
    align_info: dict[str, Any] | None = None
    aligned = existing is not None
    previous_l0 = find_previous_l0(l0_dir, l1_path) if aligned else None
    if aligned and previous_l0 is None:
        return {
            **_skip_merge_existing(
                existing=existing,
                l0_dir=l0_dir,
                align_info=None,
                first_novel_frac=None,
                median=median,
                extra="no_previous_l0_to_align",
            ),
            "gpu_index": gpu_index,
        }
    if previous_l0 is not None:
        align_info = estimate_sim3(recon, previous_l0, duration=duration)
        if align_info is None:
            return {
                **_skip_merge_existing(
                    existing=existing,
                    l0_dir=l0_dir,
                    align_info=None,
                    first_novel_frac=None,
                    median=median,
                    extra="sift_align_failed",
                ),
                "gpu_index": gpu_index,
                "previous_l0": str(previous_l0),
            }
        align_scale = float(align_info["scale"])
        align_rotation = np.asarray(align_info["rotation"], dtype=np.float64)
        align_translation = np.asarray(align_info["translation"], dtype=np.float64)
        if not (ALIGN_SCALE_MIN <= align_scale <= ALIGN_SCALE_MAX):
            return {
                **_skip_merge_existing(
                    existing=existing,
                    l0_dir=l0_dir,
                    align_info=align_info,
                    first_novel_frac=None,
                    median=median,
                    extra=f"align_scale_{align_scale:.3f}",
                ),
                "gpu_index": gpu_index,
                "previous_l0": str(previous_l0),
            }
        if median >= CLOSEUP_MASK_SKIP:
            _write_chain_cameras_if_trusted(l0_dir, recon, align_info)
            return {
                **_skip_merge_existing(
                    existing=existing,
                    l0_dir=l0_dir,
                    align_info=align_info,
                    first_novel_frac=None,
                    median=median,
                    extra=f"closeup_mask_{median:.2f}",
                ),
                "gpu_index": gpu_index,
                "previous_l0": str(previous_l0),
            }
    frames = _select_keyframes(recon.cam_c2w, voxel_size)
    skipped = set(int(index) for index in report.get("excluded_frame_indices") or [])
    from atas_geometry.service.device_mapping import device_index
    device = torch.device(f"cuda:{device_index(gpu_index)}")
    torch.cuda.set_device(device)
    acc_xyz: list[np.ndarray] = []
    acc_rgb: list[np.ndarray] = []
    acc_keys: set[int] = set(existing_keys)
    static_pixels = 0
    first_novel_frac: float | None = None
    for index in frames:
        if index in skipped:
            continue
        batch = recon.unproject(index)
        keep = batch.foreground < 0.5
        xyz = np.asarray(batch.xyz[keep])
        rgb = np.asarray(batch.rgb[keep])
        finite = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[finite], rgb[finite]
        if xyz.shape[0]:
            cam = recon.cam_c2w[index, :3, 3]
            dist = np.linalg.norm(xyz - cam, axis=1)
            near = float(np.percentile(dist, 20.0))
            keep_near = dist < max(near * 25.0, 1.0)
            xyz, rgb = xyz[keep_near], rgb[keep_near]
        if previous_l0 is not None and xyz.shape[0]:
            xyz = apply_sim3(
                xyz,
                scale=align_scale,
                rotation=align_rotation,
                translation=align_translation,
            ).astype(np.float32)
        static_pixels += int(xyz.shape[0])
        if xyz.shape[0] == 0:
            continue
        if origin is None:
            origin = np.floor(np.percentile(xyz.astype(np.float64), 1.0, axis=0) / voxel_size) * voxel_size
        uniq_xyz, uniq_rgb, keys = _unique_voxels(xyz, rgb, origin, overlap_voxel, device)
        novel = [i for i, key in enumerate(keys.tolist()) if int(key) not in acc_keys]
        if (
            previous_l0 is not None
            and first_novel_frac is None
            and uniq_xyz.shape[0]
            and not _trusted_alignment(align_info)
        ):
            first_novel_frac = float(len(novel) / max(int(uniq_xyz.shape[0]), 1))
            if first_novel_frac > NOVEL_FRAC_SKIP:
                _write_chain_cameras_if_trusted(l0_dir, recon, align_info)
                return {
                    **_skip_merge_existing(
                        existing=existing,
                        l0_dir=l0_dir,
                        align_info=align_info,
                        first_novel_frac=first_novel_frac,
                        median=median,
                        extra=f"first_frame_novel_{first_novel_frac:.2f}",
                    ),
                    "gpu_index": gpu_index,
                    "previous_l0": str(previous_l0),
                    "keyframes": frames,
                }
        if not novel:
            continue
        acc_xyz.append(uniq_xyz[novel])
        acc_rgb.append(uniq_rgb[novel])
        acc_keys.update(int(keys[i]) for i in novel)
    if origin is None or not acc_xyz:
        if existing is not None:
            if previous_l0 is not None:
                write_aligned_cameras(
                    l0_dir,
                    transform_c2w(
                        recon.cam_c2w,
                        scale=align_scale,
                        rotation=align_rotation,
                        translation=align_translation,
                    ),
                )
            return {
                "l1_point_count": int(existing.xyz.shape[0]),
                "voxel_size": float(existing.voxel_size),
                "keyframes": frames,
                "static_pixels_seen": static_pixels,
                "novel_points": 0,
                "align_scale": float(align_scale) if previous_l0 is not None else 1.0,
                "previous_l0": str(previous_l0) if previous_l0 is not None else None,
                "first_novel_frac": first_novel_frac,
                "align": _align_public(align_info),
            }
        raise RuntimeError("no static points to fuse")
    new_xyz = np.concatenate(acc_xyz, axis=0)
    new_rgb = np.concatenate(acc_rgb, axis=0)
    if existing is None:
        memory = _memory_from_voxels(
            new_xyz, new_rgb, voxel_size=voxel_size, origin=origin, shot_id=shot_id
        )
        novel = int(memory.xyz.shape[0])
    else:
        trusted = _trusted_alignment(align_info)
        over_growth = int(new_xyz.shape[0]) > max(int(existing.xyz.shape[0]), 1) * GROWTH_FAIL_RATIO
        if over_growth and not trusted:
            reason = f"growth_{new_xyz.shape[0]}_onto_{existing.xyz.shape[0]}"
            return {
                **_skip_merge_existing(
                    existing=existing,
                    l0_dir=l0_dir,
                    align_info=align_info,
                    first_novel_frac=first_novel_frac,
                    median=median,
                    extra=reason,
                ),
                "gpu_index": gpu_index,
                "previous_l0": str(previous_l0) if previous_l0 is not None else None,
                "keyframes": frames,
            }
        merged_xyz = np.concatenate((existing.xyz, new_xyz), axis=0)
        merged_rgb = np.concatenate((existing.rgb, new_rgb), axis=0)
        memory = _memory_from_voxels(
            merged_xyz,
            merged_rgb,
            voxel_size=voxel_size,
            origin=origin,
            shot_id=shot_id,
        )
        novel = int(new_xyz.shape[0])
    l1_path.parent.mkdir(parents=True, exist_ok=True)
    memory.save(l1_path)
    if previous_l0 is not None:
        write_aligned_cameras(
            l0_dir,
            transform_c2w(
                recon.cam_c2w,
                scale=align_scale,
                rotation=align_rotation,
                translation=align_translation,
            ),
        )
    return {
        "l1_point_count": int(memory.xyz.shape[0]),
        "voxel_size": float(memory.voxel_size),
        "merged": True,
        "keyframes": frames,
        "static_pixels_seen": static_pixels,
        "novel_points": novel,
        "mask_area_median": median,
        "gpu_index": gpu_index,
        "align_scale": float(align_scale) if previous_l0 is not None else 1.0,
        "previous_l0": str(previous_l0) if previous_l0 is not None else None,
        "first_novel_frac": first_novel_frac,
        "align": _align_public(align_info),
    }


