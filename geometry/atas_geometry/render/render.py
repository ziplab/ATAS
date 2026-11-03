"""GPU point rendering of the static scene memory (L1), optionally with
time-local dynamic points (L2) composited through one z-buffer."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from atas_geometry.memory.fusion import FusedSceneMemory
from atas_geometry.memory.observation import CameraCalibration
GREY = 127


def _project_entity_exclusion_mask(
    entity_point_sets: tuple[np.ndarray, ...],
    camera: CameraCalibration,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Project verified L2 entities into a conservative image-space mask.

    This mask is an audit aid only: it never changes the L1 render.  A separate
    padded bounding box is used for each requested entity so that empty pixels
    where the dynamic subject is expected do not count against the static
    condition, while the space between two actors remains auditable.
    """
    excluded = np.zeros((height, width), dtype=bool)
    w2c = np.asarray(camera.world_to_camera, dtype=np.float64)
    scale_x = width / float(camera.image_size[0])
    scale_y = height / float(camera.image_size[1])
    fx = float(camera.intrinsic[0, 0]) * scale_x
    fy = float(camera.intrinsic[1, 1]) * scale_y
    cx = float(camera.intrinsic[0, 2]) * scale_x
    cy = float(camera.intrinsic[1, 2]) * scale_y
    for raw_points in entity_point_sets:
        points = np.asarray(raw_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        cam_xyz = points @ w2c[:3, :3].T + w2c[:3, 3]
        front = np.isfinite(cam_xyz).all(axis=1) & (cam_xyz[:, 2] > 1e-5)
        cam_xyz = cam_xyz[front]
        if not len(cam_xyz):
            continue
        u = fx * cam_xyz[:, 0] / cam_xyz[:, 2] + cx
        v = fy * cam_xyz[:, 1] / cam_xyz[:, 2] + cy
        visible = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[visible]
        v = v[visible]
        if not len(u):
            continue
        # Trim isolated reconstruction outliers, then pad the observed extent.
        # Bounding each entity independently is deliberately conservative but
        # avoids masking the full interval between actors in a two-shot.
        u0, u1 = np.percentile(u, (1.0, 99.0)) if len(u) >= 20 else (u.min(), u.max())
        v0, v1 = np.percentile(v, (1.0, 99.0)) if len(v) >= 20 else (v.min(), v.max())
        pad_x = max(2.0, 0.05 * max(float(u1 - u0), 1.0))
        pad_y = max(2.0, 0.05 * max(float(v1 - v0), 1.0))
        x0 = max(0, int(math.floor(float(u0) - pad_x)))
        x1 = min(width, int(math.ceil(float(u1) + pad_x)) + 1)
        y0 = max(0, int(math.floor(float(v0) - pad_y)))
        y1 = min(height, int(math.ceil(float(v1) + pad_y)) + 1)
        if x0 < x1 and y0 < y1:
            excluded[y0:y1, x0:x1] = True
    return excluded


def _coverage_outside_mask(covered: np.ndarray, excluded: np.ndarray) -> float:
    eligible = ~np.asarray(excluded, dtype=bool)
    denominator = int(eligible.sum())
    if denominator == 0:
        return 0.0
    return float((np.asarray(covered, dtype=bool) & eligible).sum() / denominator)


# Peak transient bytes per resident point while projecting and z-buffering one
# frame (xyz/rgb, camera-space coordinates, uv, depth, masks, sort keys).
BYTES_PER_RESIDENT_POINT = 100


def render_point_budget(device) -> tuple[int, int]:
    """Points kept resident on the GPU, and the streaming batch beyond that.

    Defaults follow the card's currently free memory (at most 40% of it), so
    paper-scale scenes (tens of millions of points) stay resident on large cards
    and stream in batches on small ones. Both values can be pinned by env.
    """
    import os
    resident = os.environ.get('GEOMETRY_RENDER_RESIDENT_POINTS')
    batch = os.environ.get('GEOMETRY_RENDER_POINT_BATCH')
    if resident is None:
        free, _ = torch.cuda.mem_get_info(device)
        resident_points = max(1_000_000, int(0.4 * free / BYTES_PER_RESIDENT_POINT))
    else:
        resident_points = int(resident)
    batch_points = int(batch) if batch is not None else max(1_000_000, resident_points // 4)
    return resident_points, batch_points


def render_l1_gpu(
    memory: FusedSceneMemory,
    cameras: list[Any],
    *,
    width: int,
    height: int,
    device: torch.device,
    splat_mode: str = "pixel",
    point_footprint: float | None = None,
    max_point_radius: int = 1,
    entity_exclusion_points: tuple[np.ndarray, ...] | None = None,
    dynamic_frames=None,
    dynamic_splat_mode: str = "pixel",
    max_dynamic_radius: int = 2,
    frame_sink=None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Render static and optional dynamic points with one depth buffer.

    Low-level/legacy callers retain pixel mode. Semantic cameras explicitly use
    bounded_square with a projected static voxel footprint and a two-pixel cap.
    Dynamic expansion is opt-in here and uses per-source-pixel support, not L1.
    """
    if splat_mode not in {"pixel", "bounded_square"}:
        raise ValueError(f"unknown point splat mode {splat_mode!r}")
    if splat_mode == "bounded_square" and (point_footprint is None
            or not math.isfinite(point_footprint) or point_footprint <= 0):
        raise ValueError("bounded_square requires a finite positive point_footprint")
    if isinstance(max_point_radius, bool) or not isinstance(max_point_radius, int) or not 0 <= max_point_radius <= 8:
        raise ValueError("max_point_radius must be an integer in 0..8")
    from atas_geometry.render.dynamic_footprint import dynamic_render_options
    dynamic_render_options({"dynamic_splat_mode":dynamic_splat_mode,"max_dynamic_radius":max_dynamic_radius})
    resident_budget, point_batch = render_point_budget(device)
    if resident_budget<1 or point_batch<1:raise ValueError('render working-set budgets must be positive')
    if len(memory.xyz)>resident_budget:
        from atas_geometry.render.streamed_render import render_streamed
        return render_streamed(memory,cameras,width=width,height=height,device=device,
            splat_mode=splat_mode,point_footprint=point_footprint,max_point_radius=max_point_radius,
            entity_exclusion_points=entity_exclusion_points,dynamic_frames=dynamic_frames,
            dynamic_splat_mode=dynamic_splat_mode,max_dynamic_radius=max_dynamic_radius,
            frame_sink=frame_sink,point_batch_size=point_batch)
    torch.cuda.set_device(device)
    xyz = torch.from_numpy(np.asarray(memory.xyz, dtype=np.float32)).to(device)
    rgb = torch.from_numpy(np.asarray(memory.rgb, dtype=np.uint8)).to(device)
    static_xyz, static_rgb = xyz, rgb
    dynamic_iterator = iter(dynamic_frames) if dynamic_frames is not None else None
    dynamic_histogram={str(r):0 for r in range(3)}
    dynamic_pixel_counts=[]
    dynamic_bounds=[]
    frames: list[np.ndarray] = []
    coverages: list[float] = []
    point_center_coverages: list[float] = []
    center_coverages: list[float] = []
    static_condition_coverages: list[float] = []
    static_condition_center_coverages: list[float] = []
    entity_excluded_fractions: list[float] = []
    radius_histogram = {str(r): 0 for r in range(max_point_radius + 1)}
    splat_fragments = 0
    visible_points = 0
    splat_ms = 0.0
    start_all = torch.cuda.Event(enable_timing=True)
    end_all = torch.cuda.Event(enable_timing=True)
    start_all.record()
    n_pixels = width * height
    for camera in cameras:
        if dynamic_iterator is not None:
            dynamic_data=next(dynamic_iterator)
            dynamic_xyz, dynamic_rgb = dynamic_data[:2]
            if dynamic_splat_mode=="source_pixel":
                if len(dynamic_data)!=4: raise ValueError("source_pixel needs per-point support and boundary caps")
                support,limits=map(np.asarray,dynamic_data[2:])
                if (support.shape!=(len(dynamic_xyz),) or limits.shape!=support.shape
                        or not np.isfinite(support).all() or np.any(support<=0)
                        or not np.isfinite(limits).all() or np.any((limits<0)|(limits>2))
                        or np.any(limits!=np.floor(limits))):
                    raise ValueError("invalid dynamic source-pixel support")
            xyz = torch.cat((static_xyz, torch.as_tensor(dynamic_xyz, dtype=torch.float32, device=device)))
            rgb = torch.cat((static_rgb, torch.as_tensor(dynamic_rgb, dtype=torch.uint8, device=device)))
        w2c = torch.from_numpy(np.asarray(camera.world_to_camera, dtype=np.float32)).to(device)
        cam_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        depth = cam_xyz[:, 2]
        scale_x = width / float(camera.image_size[0])
        scale_y = height / float(camera.image_size[1])
        fx = float(camera.intrinsic[0, 0]) * scale_x
        fy = float(camera.intrinsic[1, 1]) * scale_y
        cx = float(camera.intrinsic[0, 2]) * scale_x
        cy = float(camera.intrinsic[1, 2]) * scale_y
        u = fx * cam_xyz[:, 0] / depth.clamp(min=1e-6) + cx
        v = fy * cam_xyz[:, 1] / depth.clamp(min=1e-6) + cy
        # ReconStore lifts pixel centres at (x + .5, y + .5). Raster bins
        # are [x, x + 1): rounding centres creates collisions and grid holes.
        center_u = u.floor().to(torch.int64)
        center_v = v.floor().to(torch.int64)
        finite_front = (
            torch.isfinite(cam_xyz).all(dim=1)
            & (depth > 1e-5)
        )
        zbuf = torch.full((n_pixels,), float("inf"), device=device, dtype=torch.float32)
        color = torch.full((n_pixels, 3), GREY, device=device, dtype=torch.uint8)
        dynamic_owner=torch.zeros(n_pixels,device=device,dtype=torch.bool)
        point_is_dynamic=torch.arange(len(xyz),device=device)>=len(static_xyz)
        centers_inside = (finite_front & (center_u >= 0) & (center_u < width)
                          & (center_v >= 0) & (center_v < height))
        if dynamic_iterator is not None:
            selected=centers_inside&point_is_dynamic
            if bool(selected.any()):
                uv=torch.stack((u[selected]/width,v[selected]/height),dim=1)
                bounds=torch.quantile(uv,torch.tensor([.01,.99],device=device),dim=0)
                dynamic_bounds.append(bounds.flatten().cpu().tolist())
            else: dynamic_bounds.append(None)
        center_support = torch.zeros(n_pixels, device=device, dtype=torch.bool)
        center_support[center_v[centers_inside] * width + center_u[centers_inside]] = True
        point_center_coverages.append(float(center_support.float().mean().item()))
        if splat_mode == "pixel" and (dynamic_iterator is None or dynamic_splat_mode=="pixel"):
            valid = (
                finite_front
                & (center_u >= 0)
                & (center_u < width)
                & (center_v >= 0)
                & (center_v < height)
            )
            pix = center_v[valid] * width + center_u[valid]
            d = depth[valid]
            c = rgb[valid]
            visible_points += int(pix.numel())
            splat_fragments += int(pix.numel())
            radius_histogram["0"] += int(pix.numel())
            if pix.numel() > 0:
                zbuf.scatter_reduce_(0, pix, d, reduce="amin", include_self=True)
                winners = torch.abs(d - zbuf[pix]) <= 1e-4
                if bool(winners.any()):
                    color[pix[winners]] = c[winners]
                    dynamic_owner[pix[winners]]=point_is_dynamic[valid][winners]
            dynamic_histogram["0"]+=int((valid&point_is_dynamic).sum().item())
        else:
            focal = 0.5 * (fx + fy)
            edge = (float(point_footprint) * focal / depth.clamp(min=1e-6)
                    if splat_mode=="bounded_square" else torch.ones_like(depth))
            radii = torch.ceil((edge - 1.0) * 0.5).clamp(
                min=0, max=max(0, int(max_point_radius))
            ).to(torch.int64)
            if dynamic_iterator is not None:
                if dynamic_splat_mode=="source_pixel":
                    dynamic_edge=torch.as_tensor(support,device=device)*focal/depth[len(static_xyz):].clamp(min=1e-6)
                    dynamic_radii=torch.ceil((dynamic_edge-1.)*.5).clamp(0,max_dynamic_radius).to(torch.int64)
                    radii[len(static_xyz):]=torch.minimum(dynamic_radii,torch.as_tensor(limits,device=device,dtype=torch.int64))
                else: radii[len(static_xyz):]=0
            maximum=max(max_point_radius,max_dynamic_radius if dynamic_iterator is not None and dynamic_splat_mode=="source_pixel" else 0)
            for radius in range(maximum + 1):
                selected = (
                    finite_front
                    & (radii == radius)
                    & (center_u >= -radius)
                    & (center_u < width + radius)
                    & (center_v >= -radius)
                    & (center_v < height + radius)
                )
                count = int(selected.sum().item())
                visible_points += count
                radius_histogram[str(radius)] = radius_histogram.get(str(radius), 0) + count
                if radius<=2: dynamic_histogram[str(radius)]+=int((selected&point_is_dynamic).sum().item())
                selected_u = center_u[selected]
                selected_v = center_v[selected]
                selected_depth = depth[selected]
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        uu = selected_u + dx
                        vv = selected_v + dy
                        inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                        if not bool(inside.any()):
                            continue
                        pix = vv[inside] * width + uu[inside]
                        source_depth = selected_depth[inside]
                        splat_fragments += int(pix.numel())
                        zbuf.scatter_reduce_(
                            0, pix, source_depth, reduce="amin", include_self=True
                        )
                selected_rgb = rgb[selected]
                selected_dynamic=point_is_dynamic[selected]
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        uu = selected_u + dx
                        vv = selected_v + dy
                        inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                        if not bool(inside.any()):
                            continue
                        pix = vv[inside] * width + uu[inside]
                        source_depth = selected_depth[inside]
                        source_rgb = selected_rgb[inside]
                        winners = torch.abs(source_depth - zbuf[pix]) <= 1e-4
                        if bool(winners.any()):
                            color[pix[winners]] = source_rgb[winners]
                            dynamic_owner[pix[winners]]=selected_dynamic[inside][winners]
        image = color.view(height, width, 3).detach().cpu().numpy()
        covered = torch.isfinite(zbuf) & (zbuf < 1e20)
        coverages.append(float(covered.float().mean().item()))
        if dynamic_iterator is not None: dynamic_pixel_counts.append(int(dynamic_owner.sum().item()))
        covered_2d = covered.view(height, width)
        y0, y1 = height // 4, height - height // 4
        x0, x1 = width // 4, width - width // 4
        center_coverages.append(
            float(covered_2d[y0:y1, x0:x1].float().mean().item())
        )
        if entity_exclusion_points is not None:
            excluded = _project_entity_exclusion_mask(
                entity_exclusion_points,
                camera,
                width=width,
                height=height,
            )
            covered_np = covered_2d.detach().cpu().numpy()
            static_condition_coverages.append(
                _coverage_outside_mask(covered_np, excluded)
            )
            static_condition_center_coverages.append(
                _coverage_outside_mask(
                    covered_np[y0:y1, x0:x1], excluded[y0:y1, x0:x1]
                )
            )
            entity_excluded_fractions.append(float(excluded.mean()))
        if frame_sink is None:
            frames.append(image)
        else:
            frame_sink(image)
    end_all.record()
    torch.cuda.synchronize()
    splat_ms = float(start_all.elapsed_time(end_all))
    stats = {
        "gpu_splat_ms": splat_ms,
        "mean_coverage": float(np.mean(coverages) if coverages else 0.0),
        "mean_point_center_coverage": float(np.mean(point_center_coverages) if point_center_coverages else 0.0),
        "point_center_coverages": point_center_coverages,
        "min_coverage": float(np.min(coverages) if coverages else 0.0),
        "mean_center_coverage": float(
            np.mean(center_coverages) if center_coverages else 0.0
        ),
        "min_center_coverage": float(
            np.min(center_coverages) if center_coverages else 0.0
        ),
        "coverages": coverages,
        "point_count": int(xyz.shape[0]),
        "splat_mode": splat_mode,
        "dynamic_splat_mode": dynamic_splat_mode if dynamic_iterator is not None else None,
        "dynamic_radius_histogram":dynamic_histogram,
        "dynamic_visible_pixels_per_frame":dynamic_pixel_counts,
        "dynamic_screen_bounds_per_frame":dynamic_bounds,
        "max_dynamic_radius":max_dynamic_radius if dynamic_splat_mode=="source_pixel" else 0,
        "point_footprint": point_footprint,
        "max_point_radius": int(max_point_radius),
        "radius_histogram": radius_histogram,
        "splat_fragment_count": int(splat_fragments),
        "mean_fragments_per_visible_point": float(
            splat_fragments / max(1, visible_points)
        ),
        "occlusion": "single_nearest_depth_per_splat_fragment",
    }
    if entity_exclusion_points is not None:
        stats["static_condition_coverage"] = {
            "mean": float(np.mean(static_condition_coverages)),
            "min": float(np.min(static_condition_coverages)),
            "mean_center": float(np.mean(static_condition_center_coverages)),
            "min_center": float(np.min(static_condition_center_coverages)),
            "mean_entity_excluded_fraction": float(
                np.mean(entity_excluded_fractions)
            ),
            "mask_policy": "per_entity_projected_l2_bbox_p01_p99_pad5pct_min2px",
        }
    return frames, stats


def _rescale_camera(
    camera: CameraCalibration, width: int, height: int
) -> CameraCalibration:
    """Stored chain poses carry the resolution they were rendered at."""
    src_w, src_h = camera.image_size
    if (src_w, src_h) == (width, height):
        return camera
    scale_x = width / float(src_w)
    scale_y = height / float(src_h)
    intrinsic = np.asarray(camera.intrinsic, dtype=np.float64).copy()
    intrinsic[0, :] *= scale_x
    intrinsic[1, :] *= scale_y
    return CameraCalibration(
        camera_id=camera.camera_id,
        world_to_camera=np.asarray(camera.world_to_camera, dtype=np.float64),
        intrinsic=intrinsic,
        image_size=(int(width), int(height)),
    )
