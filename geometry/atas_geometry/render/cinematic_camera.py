"""Ground and execute one authored camera path; report issues for explicit repair.

This is not a global identity tracker. VLM regions are frame-local evidence;
they never become identity_verified masks or permanent world-memory labels.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial.transform import Rotation, Slerp

from atas_geometry.memory.camera_query import (
    _look_at_camera,
    _normalize,
    _rotate_axis,
    check_camera_query,
    generate_camera_query,
    sparse_geometry_sample,
)

MOVES = {"locked", "push_in", "pull_back", "orbit", "arc_left", "arc_right", "truck_left", "truck_right", "crane_up", "crane_down", "push_in_with_arc", "pan_left", "pan_right", "tilt_up", "tilt_down", "handheld", "track_left", "track_right", "track_forward", "track_back"}
ROTATIONAL_MOVES = {"pan_left", "pan_right", "tilt_up", "tilt_down", "handheld"}
TRACK_MODES = {"track_left":"truck_left", "track_right":"truck_right",
               "track_forward":"push_in", "track_back":"pull_back"}


def numbers(values, n, name, lo=None, hi=None):
    a = np.asarray(values, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all() or (lo is not None and np.any(a < lo)) or (hi is not None and np.any(a > hi)):
        raise ValueError(f"invalid {name}")
    return a


def validate_plan(plan):
    if plan.get("schema_version") != 1 or plan.get("status") != "ok":
        raise ValueError("camera planner could not resolve the requested shot")
    subjects = plan.get("subjects")
    if not isinstance(subjects, list) or not 1 <= len(subjects) <= 4:
        raise ValueError("camera intent needs one to four explicitly selected subjects")
    for item in subjects:
        if item.get("framing_role", "focal") not in {"focal", "context"}:
            raise ValueError("subject framing_role must be focal or context")
        if not str(item.get("label", "")).strip() or item.get("kind") not in {"dynamic", "static"}:
            raise ValueError("subject needs a label and dynamic/static kind")
        if item.get('part') not in {None, 'face', 'head', 'face/head', 'upper_body', 'full_body', 'body', 'object'}:
            raise ValueError('subject region has unsupported part; replan the visible focal part')
        box = numbers(item.get("bbox"), 4, "subject bbox", 0, 1)
        point = numbers(item.get("anchor_uv"), 2, "subject anchor", 0, 1)
        if np.any(box[2:] - box[:2] <= .005) or np.any(point < box[:2]) or np.any(point > box[2:]):
            raise ValueError("subject anchor must lie inside a nonempty bbox")
        confidence = float(item.get("confidence", 0))
        if not math.isfinite(confidence) or confidence < .75 or confidence > 1 or item.get("visible") is not True:
            raise ValueError("subject visually unresolved; do not substitute an anonymous cluster")
    framing = plan["framing"]
    numbers(framing.get("screen_position"), 2, "screen position", .1, .9)
    numbers([framing.get("vertical_fraction"), framing.get("horizontal_fraction")], 2, "frame fractions", .1, .9)
    numbers([framing.get("side_degrees"), framing.get("elevation_degrees")], 2, "view angles", -85, 85)
    motion = plan["motion"]
    if motion.get("name") not in MOVES:
        raise ValueError("unsupported camera motion")
    numbers([motion.get("degrees")], 1, "motion degrees", 0, 100)
    numbers([motion.get("travel_ratio")], 1, "travel ratio", 0, .7)
    if motion['name'] in ROTATIONAL_MOVES:
        if float(motion['degrees']) <= 0 or float(motion['travel_ratio']) != 0:
            raise ValueError('rotational motion requires positive degrees and zero travel')
        if motion['name']=='handheld' and float(motion['degrees'])>3:
            raise ValueError('handheld sway is limited to 3 degrees')
    if motion['name'] in TRACK_MODES and (float(motion['travel_ratio'])<=0 or float(motion['degrees'])!=0):
        raise ValueError('tracking requires positive travel and zero arc')
    framing["screen_position"] = [float(v) for v in framing["screen_position"]]
    for key in ("vertical_fraction", "horizontal_fraction", "side_degrees", "elevation_degrees"):
        framing[key] = float(framing[key])
    for key in ("degrees", "travel_ratio"):
        motion[key] = float(motion[key])
    for item in subjects:
        item["bbox"] = [float(v) for v in item["bbox"]]
        item["anchor_uv"] = [float(v) for v in item["anchor_uv"]]
        item["confidence"] = float(item["confidence"])
    return plan


def ground_subjects(recon, index, plan, *, subject_masks=None):
    """Use seed-local depth inside the VLM region; reject missing/ambiguous depth."""
    validate_plan(plan)
    depth = recon.depth_at(index)
    h, w = depth.shape
    observation = recon.unproject(index)
    # Reconstruct the same valid-pixel ordering as ReconStore, including sky.
    from atas_geometry.memory.vista4d_layers import _mask
    valid = np.isfinite(depth) & (depth > 1e-5)
    if recon.exclude_sky and index < len(recon.sky_files):
        valid &= ~_mask(recon.sky_files[index], depth.shape)
    ys, xs = np.nonzero(valid)
    uv = np.column_stack(((xs + .5) / w, (ys + .5) / h))
    if len(uv) != len(observation.xyz):
        raise ValueError("source pixel/depth indexing mismatch")
    sets, audits = [], []
    def unavailable_context(item, reason):
        if item.get("framing_role", "focal") != "context":
            return False
        # Empty groups preserve the subjects/groups/audits positional contract.
        # Unknown context is not geometry evidence and never affects focal fit.
        sets.append(np.empty((0, 3), dtype=np.float32))
        audits.append({"label": item["label"], "part": item.get("part"), "points": 0,
                       "framing_role": "context", "status": "unknown", "reason": reason,
                       "evidence": "unavailable", "identity_verified": False,
                       "part_semantics_verified": False})
        return True
    for item in plan["subjects"]:
        box, seed = np.asarray(item["bbox"]), np.asarray(item["anchor_uv"])
        inside = np.all((uv >= box[:2]) & (uv <= box[2:]), axis=1)
        desired = observation.foreground >= .5 if item["kind"] == "dynamic" else observation.foreground < .5
        identity_mask = (subject_masks or {}).get(item['label'])
        instance_fraction = None
        if identity_mask is not None:
            identity_mask = np.asarray(identity_mask, dtype=bool)
            if identity_mask.shape != depth.shape:
                raise ValueError('subject region identity mask/depth shape mismatch')
            # A seed proves one pixel's identity, not every foreground pixel in
            # the box. Constrain the entire selected region to that same track.
            desired &= identity_mask[ys, xs]
            py, px = np.nonzero(identity_mask)
            puv = np.column_stack(((px + .5) / w, (py + .5) / h))
            contained = np.all((puv >= box[:2]) & (puv <= box[2:]), axis=1)
            instance_fraction = float(contained.mean()) if len(puv) else 0.
            clipped = (identity_mask[0].any() or identity_mask[-1].any()
                       or identity_mask[:,0].any() or identity_mask[:,-1].any())
            # This is a contradiction check, NOT an anatomical face detector.
            # A cropped source may show only a face; do not reject that case.
            if item.get('part') in {'face','head','face/head','upper_body'} and not clipped and instance_fraction >= .9:
                raise ValueError(f"subject region for {item['label']} claims {item['part']} but contains "
                                 f"{instance_fraction:.1%} of the complete visible instance; replan the actual part bbox")
        inside &= desired
        near = inside & (np.linalg.norm((uv - seed) / np.maximum(box[2:] - box[:2], .01), axis=1) < .2)
        if np.count_nonzero(near) < 16:
            if unavailable_context(item, "no_reliable_seed_depth"):
                continue
            raise ValueError(f"no reliable seed depth for {item['label']}")
        z = depth[ys, xs]
        median = float(np.median(z[near]))
        mad = float(np.median(np.abs(z[near] - median)))
        if mad / median > .15:
            if unavailable_context(item, "ambiguous_seed_depth"):
                continue
            raise ValueError(f"ambiguous seed depth for {item['label']}")
        selected = inside & (np.abs(z - median) <= max(4 * mad, .15 * median))
        points = np.asarray(observation.xyz)[selected]
        if len(points) < 32:
            if unavailable_context(item, "insufficient_grounded_region"):
                continue
            raise ValueError(f"insufficient grounded region for {item['label']}")
        sets.append(points)
        audits.append({"label": item["label"], "part": item.get("part"), "points": len(points),
            "evidence": "verified_instance_intersected_vlm_region_seed_depth" if identity_mask is not None else "vlm_region_seed_depth",
            "identity_verified": identity_mask is not None, "confidence": item["confidence"],
            "instance_fraction_in_region": instance_fraction,
            "part_semantics_verified": False,
            "part_verification_policy": "anatomy_requires_visual_review; occupancy_is_not_semantic_acceptance"})
    # Keep every region for diagnostics, but never use scenery to size a close-up.
    if not any(len(p) for p in sets):
        raise ValueError("camera intent has no grounded focal subject")
    focus = np.mean([np.median(p, axis=0) for p in sets if len(p)], axis=0)
    all_dynamic = np.asarray(observation.xyz)[observation.foreground >= .5]
    return np.concatenate(sets), focus, all_dynamic, audits


def project(points, camera):
    p = np.asarray(points) @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    uvh = p @ camera.intrinsic.T
    uv = uvh[:, :2] / np.maximum(p[:, 2:3], 1e-8)
    return uv / np.asarray(camera.image_size), p[:, 2]


def aimed_camera(reference, position, focus, screen, up):
    """Aim the desired subject at an image location without changing intrinsics."""
    c2w = _look_at_camera(position, focus, up)
    w, h = reference.image_size
    k = reference.intrinsic
    ray = _normalize(np.array([(screen[0] * w - k[0, 2]) / k[0, 0], (screen[1] * h - k[1, 2]) / k[1, 1], 1.]))
    target = np.array([0., 0, 1.])
    cross = np.cross(ray, target)
    if np.linalg.norm(cross) > 1e-9:
        rotation = Rotation.from_rotvec(_normalize(cross) * math.acos(float(np.clip(ray @ target, -1, 1)))).as_matrix()
        c2w[:3, :3] = c2w[:3, :3] @ rotation
    return replace(reference, camera_id="semantic_framing", world_to_camera=np.linalg.inv(c2w))


def fit_reference(reference, points, focus, framing, side):
    up = _normalize(-reference.camera_to_world[:3, 1])
    offset = reference.center_world - focus
    direction = _normalize(offset - up * (offset @ up))
    direction = _rotate_axis(direction, up, math.radians(side))
    elevation = math.radians(float(framing["elevation_degrees"]))
    direction = direction * math.cos(elevation) + up * math.sin(elevation)
    screen = np.asarray(framing["screen_position"])
    extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-4)
    low, high = extent * .05, max(extent * 20, np.linalg.norm(offset) * 4)
    for _ in range(24):
        distance = (low + high) / 2
        candidate = aimed_camera(reference, focus + direction * distance, focus, screen, up)
        uv, z = project(points, candidate)
        bounds = np.percentile(uv, [2, 98], axis=0)
        size = bounds[1] - bounds[0]
        fits = (np.min(z) > 0 and size[0] <= framing["horizontal_fraction"]
            and size[1] <= framing["vertical_fraction"] and np.all(bounds[0] >= .03) and np.all(bounds[1] <= .97))
        if fits:
            high = distance
        else:
            low = distance
    return aimed_camera(reference, focus + direction * high, focus, screen, up)


def coverage_risk(points, cameras, excluded_points, subject_points=None):
    """Coarse projected support, NOT a calibrated final-pixel coverage gate."""
    values = []
    for camera in cameras:
        def mask_for(xyz):
            uv, z = project(xyz, camera)
            valid = (z > 0) & np.all((uv >= 0) & (uv < 1), axis=1)
            pixels = np.floor(uv[valid] * [96, 54]).astype(int)
            mask = np.zeros((54, 96), dtype=bool)
            mask[pixels[:, 1], pixels[:, 0]] = True
            return mask
        occupied = mask_for(points)
        excluded = binary_dilation(mask_for(excluded_points), iterations=1) if len(excluded_points) else np.zeros_like(occupied)
        evaluated = ~excluded
        unknown = evaluated & ~occupied
        components, _ = label(unknown)
        sizes = np.bincount(components.ravel())[1:]
        large = float(sizes.max(initial=0) / max(1, evaluated.sum()))
        center = np.zeros_like(occupied)
        center[13:41, 24:72] = True
        central = center & evaluated
        neighborhood = center.copy()
        if subject_points is not None and len(subject_points):
            uv, z = project(subject_points, camera)
            visible = uv[z > 0]
            if len(visible):
                bounds = np.percentile(visible, [2, 98], axis=0)
                bounds[0] -= .12
                bounds[1] += .12
                lo = np.floor(np.clip(bounds[0], 0, 1) * [96,54]).astype(int)
                hi = np.ceil(np.clip(bounds[1], 0, 1) * [96,54]).astype(int)
                neighborhood[:] = False
                neighborhood[lo[1]:hi[1],lo[0]:hi[0]] = True
        values.append({"unknown_fraction": float(unknown.sum() / max(1, evaluated.sum())),
            "largest_unknown_region": large,
            "central_unknown_fraction": float((unknown & center).sum() / max(1, central.sum())),
            "subject_neighborhood_unknown_fraction": float((unknown & neighborhood).sum() / max(1, (neighborhood & evaluated).sum())),
            "excluded_dynamic_fraction": float(excluded.mean())})
    summary = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
    summary["worst_largest_unknown_region"] = max(v["largest_unknown_region"] for v in values)
    summary["risk_score"] = summary["unknown_fraction"] + summary["subject_neighborhood_unknown_fraction"] + 2 * summary["worst_largest_unknown_region"]
    summary["policy"] = "soft_ranking_within_same_intent_not_generation_veto"
    summary["estimator"] = "96x54_sparse_geometry_support_not_final_pixel_coverage"
    return summary


def focal_geometry(points, plan, groups):
    if groups is None:
        return points, None
    selected = [group for item, group in zip(plan["subjects"], groups)
                if item.get("framing_role", "focal") == "focal"]
    if not selected:
        raise ValueError("subject framing requires at least one focal region")
    return np.concatenate(selected), np.mean([np.median(p, axis=0) for p in selected], axis=0)


def framing_check(points, cameras, framing, *, continuous=False, locked=False):
    """Geometry framing validity, separate from collision safety and coverage ranking.

    Sample the path, not only its end. Continuous opening poses are immutable;
    their initial crop may recover, but may not get worse while approaching target.
    """
    samples = []
    # Deterministic spatially dense subset for the repeated projection check only.
    sample = points[::max(1, len(points)//16000)]
    for index in sorted(set(np.linspace(0, len(cameras)-1, min(9, len(cameras))).astype(int))):
        uv, z = project(sample, cameras[index])
        bounds = np.percentile(uv, [2, 98], axis=0)
        size = bounds[1]-bounds[0]
        clipping = float(np.mean((z <= 0) | np.any((uv < 0) | (uv > 1), axis=1)))
        samples.append({"frame_index": int(index), "clipped_fraction": clipping,
                        "bbox": bounds.reshape(-1).tolist(), "size": size.tolist()})
    end = samples[-1]
    size = np.asarray(end["size"])
    requested = np.array([framing["horizontal_fraction"], framing["vertical_fraction"]])
    fill = float(np.max(size / requested))
    allowed_crop = max(.05, samples[0]["clipped_fraction"]) if continuous else .05
    failures = []
    if any(s["clipped_fraction"] > allowed_crop + .005 for s in samples):
        failures.append("focal subject leaves frame during the camera path")
    if not (continuous and locked):
        if end["clipped_fraction"] > .05:
            failures.append("focal subject clipped at the framing target")
        if not .70 <= fill <= 1.25:
            failures.append("focal framing scale does not match requested part/shot size")
    return {"passed": not failures, "failures": failures, "samples": samples,
            "target_fill_ratio": fill, "policy": "focal_path_v2"}


def rotational_cameras(reference, times, move, degrees, *, continuous):
    """True camera-center rotation; no orbit, translation, zoom or 2D warp.

    A cut is fitted at the end of a pan/tilt. A continuous shot begins with the
    exact inherited pose. Handheld is deterministic low-frequency angular sway.
    """
    times=np.asarray(times,dtype=float)
    if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('rotational camera requires increasing frame timestamps')
    phase=(times-times[0])/(times[-1]-times[0])
    angle=math.radians(degrees)
    cameras=[]
    for i,p in enumerate(phase):
        if continuous and i==0:
            cameras.append(reference);continue
        progress=.5-.5*math.cos(math.pi*p)
        if move=='handheld':
            t=times[i]-times[0]
            rotvec=angle*np.array([.45*math.sin(2*math.pi*.7*t),
                                  .65*math.sin(2*math.pi*.43*t),
                                  .25*math.sin(2*math.pi*.31*t)])
        else:
            amount=angle*(progress if continuous else progress-1)
            sign=-1 if move in {'pan_left','tilt_down'} else 1
            axis=np.array([1.,0,0]) if move.startswith('tilt_') else np.array([0.,1,0])
            rotvec=sign*amount*axis
        pose=reference.camera_to_world.copy()
        pose[:3,:3]=pose[:3,:3]@Rotation.from_rotvec(rotvec).as_matrix()
        cameras.append(replace(reference,camera_id=move,world_to_camera=np.linalg.inv(pose)))
    return tuple(cameras)


def plan_query(memory, reference, points, focus, dynamic_points, plan, times, relation, *, subject_groups=None,
               framing_policy="strict"):
    validate_plan(plan)
    if framing_policy not in {"strict", "advisory"}:
        raise ValueError("framing_policy must be strict or advisory")
    requested_framing_policy = framing_policy
    if relation == "recapture":
        framing_policy = "strict"
    points, focal_focus = focal_geometry(points, plan, subject_groups)
    if focal_focus is not None:
        focus = focal_focus
    if not len(points) or not np.isfinite(points).all() or not np.isfinite(focus).all():
        raise ValueError("camera requires finite nonempty grounded focal geometry")
    framing, motion = plan["framing"], plan["motion"]
    move = motion["name"]
    path_move = TRACK_MODES.get(move, move)
    continuous = relation == "continuous"
    side = float(framing["side_degrees"])
    geometry = sparse_geometry_sample(memory)
    # One authored trajectory. Corrections are explicit replans, never hidden search.
    start = reference if continuous else fit_reference(reference, points, focus, framing, side)
    base_travel = float(motion["travel_ratio"])
    if continuous and move in {"push_in", "pull_back"}:
        # Match the end framing by solving distance, not by a fixed tiny dolly.
        offset = reference.center_world - focus
        up = _normalize(-reference.camera_to_world[:3, 1])
        elevation = math.degrees(math.asin(float(np.clip(_normalize(offset) @ up, -1, 1))))
        fitted = fit_reference(reference, points, focus, {**framing, "elevation_degrees": elevation}, 0.)
        required = 1 - np.linalg.norm(fitted.center_world - focus) / np.linalg.norm(offset)
        if (move == "push_in" and required > 0) or (move == "pull_back" and required < 0):
            base_travel = min(abs(float(required)), .9)
    sign = -1 if path_move in {"arc_left", "truck_left", "crane_down"} else 1
    mode = ("orbit" if path_move in {"orbit", "arc_left", "arc_right"} else "dolly" if path_move in {"push_in", "pull_back"}
        else "truck" if path_move.startswith("truck") else "crane" if path_move.startswith("crane") else "cinematic_arc" if path_move == "push_in_with_arc" else "locked")
    travel = base_travel
    if path_move == "pull_back":
        travel = -travel
    elif mode in {"truck", "crane"}:
        travel *= sign
    trajectory_start = start
    if not continuous and path_move == "push_in":
        # Fit the close-up at the END, then move into it. Fitting the start
        # and advancing further was pushing already-fitted subjects out.
        position = focus + (start.center_world-focus) / (1-travel)
        trajectory_start = aimed_camera(start, position, focus, framing["screen_position"],
                                         _normalize(-reference.camera_to_world[:3, 1]))
    query = generate_camera_query(memory, times, trajectory_start, mode=mode,
        focus_world=focus, azimuth_degrees=0, orbit_degrees=sign * float(motion["degrees"]),
        travel_ratio=travel, lift_ratio=.04 if mode == "cinematic_arc" else 0,
        truck_ratio=0)
    up = _normalize(-reference.camera_to_world[:3, 1])
    cameras = []
    for i, camera in enumerate(query.cameras):
        desired = aimed_camera(start, camera.center_world, focus, framing["screen_position"], up)
        if continuous:
            if move == "locked" or i == 0:
                desired = reference
            else:
                progress = .5 - .5 * math.cos(math.pi * i / (len(times) - 1))
                rotations = Rotation.from_matrix(np.stack([reference.camera_to_world[:3, :3], desired.camera_to_world[:3, :3]]))
                pose = desired.camera_to_world.copy()
                pose[:3, :3] = Slerp([0, 1], rotations)(progress).as_matrix()
                desired = replace(desired, world_to_camera=np.linalg.inv(pose))
        cameras.append(desired)
    query = replace(query, cameras=tuple(cameras))
    if move in ROTATIONAL_MOVES:
        query=replace(query,cameras=rotational_cameras(start,times,move,
            float(motion['degrees']),continuous=continuous))
    if any(not np.isfinite(camera.world_to_camera).all() or not np.isfinite(camera.intrinsic).all()
           for camera in query.cameras):
        raise ValueError("authored camera path has nonfinite calibration")
    safety = check_camera_query(query, geometry, min_end_distance_ratio=.05, min_surface_clearance_ratio=.02)
    if not safety["passed"]:
        raise ValueError("no geometrically safe same-intent authored trajectory; locally revise it: " + str(safety["failures"]))
    framing_validity = framing_check(points, query.cameras, framing,
                                     continuous=continuous, locked=move == "locked")
    if not framing_validity["passed"] and framing_policy == "strict":
        raise ValueError("no geometrically safe same-intent authored trajectory with valid focal framing; "
                         "locally revise it: " + str(framing_validity))
    observed = [query.cameras[i] for i in sorted({0, len(times)//2, len(times)-1})]
    risk = coverage_risk(geometry, observed, dynamic_points, points)
    # Framing remains meaningful while a continuous shot approaches its end target.
    uv, z = project(points, query.cameras[-1])
    bounds = np.percentile(uv, [2, 98], axis=0)
    size = bounds[1] - bounds[0]
    framing_error = abs(float(size[1]) - framing["vertical_fraction"]) + float(np.linalg.norm(np.mean(bounds, axis=0) - framing["screen_position"]))
    clipping = float(np.mean((z <= 0) | np.any((uv < 0) | (uv > 1), axis=1)))
    selected = {"side_degrees": side if not continuous else None,
        "motion_scale": 1., "solved_travel_ratio": travel, "framing_error": framing_error,
        "subject_clipped_fraction": clipping,
        "framing_warnings": list(framing_validity["failures"]) if not framing_validity["passed"] else [],
        "coverage": {**risk, "policy": "diagnostic_only_not_trajectory_selection"},
        "geometry_safety": safety, "framing_validity": framing_validity}
    return query, {"contract": "semantic-camera-v1", "relation": relation, "move": move,
        "framing_policy": framing_policy, "requested_framing_policy": requested_framing_policy,
        "framing_warnings": selected["framing_warnings"],
        "subject_evidence": "frame_local_vlm_regions_not_persistent_identity",
        "selected": selected, "candidate_count": 1, "rejected_candidates": [],
        "ranking": "none_single_authored_path", "execution_policy": "single_authored_path_v1", "frame0_inherited": continuous,
        "requested_motion": motion, "requested_framing": framing,
        "candidates": [selected]}
