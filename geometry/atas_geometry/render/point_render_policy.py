from __future__ import annotations

import math


def resolve_point_render_mode(value: object) -> str:
    """Resolve an explicit per-render experiment switch."""
    mode = str(value or "pixel").strip().lower()
    if mode not in {"pixel", "bounded_square"}:
        raise ValueError(f"unknown point render mode {mode!r}")
    return mode


def semantic_point_render_options(payload: dict, voxel_size: float) -> dict:
    """Static points default to pixel centres; expansion is explicit opt-in.

    Dynamic source-pixel support is configured separately and is unchanged.
    """
    mode = resolve_point_render_mode(payload.get("point_render_mode") or "pixel")
    radius = payload.get("max_point_radius", 2)
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 2:
        raise ValueError("semantic max_point_radius must be an integer in 0..2")
    if mode == "bounded_square" and (not math.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("bounded static rendering requires a finite positive voxel size")
    return {"splat_mode": mode,
            "point_footprint": float(voxel_size) if mode == "bounded_square" else None,
            "max_point_radius": radius if mode == "bounded_square" else 0}


def bounded_projected_radius(
    footprint_world: float,
    focal_px: float,
    depth: float,
    *,
    max_radius: int = 1,
) -> int:
    """Return the experimental point radius without permitting large disks.

    The projected L1 voxel edge still controls whether a point needs more
    than one pixel, but the radius is capped.  A larger missing footprint is
    a camera-coverage problem, not something the rasterizer should conceal by
    drawing a conspicuous circle over unobserved pixels.
    """
    if footprint_world < 0.0 or focal_px <= 0.0 or depth <= 0.0:
        raise ValueError("footprint, focal length and depth must describe a valid view")
    edge_px = float(footprint_world) * float(focal_px) / float(depth)
    radius = max(0, int(math.ceil((edge_px - 1.0) * 0.5)))
    return min(radius, max(0, int(max_radius)))
