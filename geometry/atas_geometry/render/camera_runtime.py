"""Production bridge for semantic-camera-v1, shared by L1 and event L1+L2."""
from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np
import torch
from atas_geometry.service.device_mapping import device_index
from atas_geometry.memory.fusion import FusedSceneMemory
from atas_geometry.memory.observation import CameraCalibration
from atas_geometry.memory.vista4d_layers import ReconStore, _reference_camera_at
from atas_geometry.memory.layer_writeback import write_camera_query_npz
from atas_geometry.render.camera_context import camera_context
from atas_geometry.render.cinematic_camera import ground_subjects, plan_query, project
from atas_geometry.memory.fuse import _install_depth_reader
from atas_geometry.service.paths import l0_dir, l1_path, last_camera_path, resolve_home_shot, resolve_start_shot
from atas_geometry.world.recapture import SceneEventRecon
from atas_geometry.render.render import render_l1_gpu, _rescale_camera
from atas_geometry.render.point_render_policy import semantic_point_render_options
from atas_geometry.render.dynamic_footprint import dynamic_frame, dynamic_render_options
def effective_framing_policy(payload):
    requested = payload.get("framing_policy", "strict")
    if requested not in {"strict", "advisory"}:
        raise ValueError("framing_policy must be strict or advisory")
    # A new-time L1 preview is an advisory reference, not the future actor pose.
    # Event recapture must never weaken time-indexed L2 framing by this switch.
    if (payload.get("render_mode") == "recapture" or payload.get("camera_relation") == "recapture"
            or payload.get("geometry_layers") != "l1"):
        return "strict"
    return requested


def registered_endpoint_camera(root, run, story, scene, shot, width, height):
    from atas_geometry.service.paths import story_root
    from atas_geometry.world.world_state import committed_observation
    source = committed_observation(story_root(root, run, story), shot)
    if source is None:
        return None
    meta = json.loads((source/'observation.json').read_text())
    if meta.get('registered') is not True or meta.get('scene_id') != scene:
        return None
    observed = SceneEventRecon(source, meta)
    return _reference_camera_at(observed, width, height, observed.count - 1)


def render_semantic_camera(root, payload, out_mp4, width, height, gpu_index):
    framing_policy = effective_framing_policy(payload)
    _install_depth_reader()
    context = camera_context(root, payload)
    plan = payload["camera_plan"]
    if plan.get("context", {}).get("context_id") != context["context_id"]:
        raise ValueError("camera context changed; replan against the current observation/world")
    run, story, scene = (context[k] for k in ("run_id", "story_id", "scene_id"))
    source = l0_dir(root, run, story, context["source_shot_id"])
    metadata = source / "observation.json"
    meta = json.loads(metadata.read_text()) if metadata.is_file() else {}
    if meta.get("registered") is True:
        recon = SceneEventRecon(source, meta)
    else:
        # Only Home may have legacy metadata: it defines the scene coordinate frame.
        if context["source_shot_id"] != resolve_home_shot(root, run, story, scene):
            raise ValueError("unregistered camera anchor observation")
        recon = ReconStore(source, duration=context["recon_frame_count"] / context["source_fps"], exclude_sky=True, mask_dilate=0)
    index = context["frame_index"]
    reference = _reference_camera_at(recon, width, height, index)
    relation = str(payload.get("camera_relation"))
    continuous = relation == "continuous"
    role, start, fallback = "independent_cut", context["source_shot_id"], False
    pose_basis = 'grounding_observation'
    if continuous:
        home = resolve_home_shot(root, run, story, scene)
        requested = str(payload.get("previous_shot_id") or payload.get("start_from_shot_id") or "")
        start, role, fallback = resolve_start_shot(root, run, story, scene, requested=requested, home=home, rendering_shot_id=payload["shot_id"])
        if role == "previous_shot":
            data = json.loads(last_camera_path(root, run, story, start).read_text())
            reference = _rescale_camera(CameraCalibration(data["camera_id"], np.asarray(data["world_to_camera"]), np.asarray(data["intrinsic"]), tuple(data["image_size"])), width, height)
            pose_basis = 'previous_planned_endpoint'
            if payload.get('continuous_pose_policy') == 'registered_observation_end_v1':
                observed = registered_endpoint_camera(root,run,story,scene,start,width,height)
                if observed is not None:
                    reference = observed
                    pose_basis = 'previous_registered_observation_endpoint'
        elif context["source_shot_id"] != home:
            home_recon = ReconStore(l0_dir(root, run, story, home), duration=1, exclude_sky=True)
            reference = _reference_camera_at(home_recon, width, height, home_recon.count - 1)
        else:
            # Grounding may use an earlier observation; it cannot rewind the
            # physical start of a continuous camera shot.
            reference = _reference_camera_at(recon, width, height, recon.count - 1)
        if role != 'previous_shot':pose_basis = 'home_observation_endpoint'
    elif relation not in {"cut", "cut_recompose", "recapture"}:
        raise ValueError("unknown semantic camera relation")
    memory = FusedSceneMemory.load(l1_path(root, run, story, scene))
    point_options = semantic_point_render_options(payload, memory.voxel_size)
    # Subjects are grounded frame-locally from VLM regions and depth; they are
    # never promoted to persistent identity masks.
    points, focus, dynamics, anchor_audit = ground_subjects(recon, index, plan)
    fps, duration = int(payload.get("fps", 24)), float(payload["duration"])
    if not 1 <= fps <= 120 or not np.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError("invalid output duration/fps")
    frames = max(2, round(duration * fps))
    event = payload.get("render_mode") == "recapture"
    expected_layers = "l1+l2" if event else "l1"
    if payload.get("geometry_layers") != expected_layers:
        raise ValueError("semantic camera layers do not match render mode")
    beginning = float(payload.get("source_time_start", 0)) if event else 0.
    times = tuple(beginning + i / fps for i in range(frames))
    groups = np.split(points, np.cumsum([a["points"] for a in anchor_audit])[:-1])
    query, planning = plan_query(memory, reference, points, focus, dynamics, plan, times,
                                "continuous" if continuous else relation, subject_groups=groups,
                                framing_policy=framing_policy)
    # Only small screen-space metadata crosses machines, never raw points.
    composition=[]; cursor=0
    for subject in anchor_audit:
        group=points[cursor:cursor+subject["points"]]; cursor+=subject["points"]
        uv,z=project(group,query.cameras[0]); visible=uv[z>0]
        if len(visible):
            bounds=np.clip(np.percentile(visible,[2,98],axis=0),0,1)
            composition.append({"label":subject["label"],"part":subject["part"],
                "bbox":bounds.reshape(-1).tolist()})
    composition_samples=[]
    for frame_index in np.unique(np.linspace(0,frames-1,min(7,frames)).astype(int)):
        regions=[]; cursor=0
        for subject in anchor_audit:
            group=points[cursor:cursor+subject["points"]];cursor+=subject["points"]
            uv,z=project(group,query.cameras[int(frame_index)]);visible=uv[z>0]
            bounds=np.percentile(visible,[2,98],axis=0).reshape(-1).tolist() if len(visible) else None
            regions.append({"label":subject["label"],"part":subject["part"],"bbox_unclipped":bounds})
        composition_samples.append({"frame_index":int(frame_index),"regions":regions})
    indices, counts = [], []
    def dynamic_frames():
        offset = (int(meta["original_frame_count"]) - recon.count) // 2
        for time in times:
            if not offset / float(meta["source_fps"]) <= time < (offset + recon.count) / float(meta["source_fps"]):
                raise ValueError("event time is outside reconstructed frames")
            source_index = min(recon.count - 1, round(time * float(meta["source_fps"])) - offset)
            observation = recon.unproject(source_index)
            mask = observation.foreground >= .5
            indices.append(source_index)
            counts.append(int(mask.sum()))
            yield dynamic_frame(recon,source_index,observation)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_mp4.with_name("memory_input.partial.mp4")
    # Splatted points are high-frequency; an unconstrained encode reached ~90 Mbit/s
    # and exceeded the generator's 100 MB input limit. A 25 Mbit/s cap keeps a 15 s
    # request near 50 MB, ample for a conditioning video.
    with imageio.get_writer(str(temporary), fps=fps, codec="libx264", macro_block_size=1, quality=None,
                            output_params=["-crf", "18", "-maxrate", "25M", "-bufsize", "25M"]) as writer:
        _, stats = render_l1_gpu(memory, list(query.cameras), width=width, height=height,
            device=torch.device(f"cuda:{device_index(gpu_index)}"), dynamic_frames=dynamic_frames() if event else None,
            frame_sink=writer.append_data, **point_options, **dynamic_render_options(payload))
    if event and not any(counts):
        raise ValueError("event contains no reconstructed dynamic points")
    query.save(out_mp4.with_name("camera_query.json"))
    write_camera_query_npz(query, out_mp4.with_name("target_camera.npz"))
    report = {"geometry_layers": expected_layers, "camera_contract": "semantic-camera-v1",
        "gpu_reserved_peak_mib": round(torch.cuda.max_memory_reserved(device_index(gpu_index)) / 2**20),
        "camera_runtime_revision": "film-first-framing-v1",
        "continuous_pose_basis": pose_basis,
        "framing_policy": framing_policy, "requested_framing_policy": payload.get("framing_policy", "strict"),
        "framing_warnings": planning["framing_warnings"],
        "camera_planning": planning, "anchor_audit": anchor_audit, "source_context": context,
        "first_frame_subject_regions":composition,
        "sampled_subject_regions":composition_samples,
        "subject_region_semantics":"projection_of_reference_anchors_not_predicted_actor_motion",
        "start_from_shot_id": start, "start_from_role": role, "chained_fallback_to_home": fallback,
        "requested_camera_semantic": plan["motion"]["name"], "executed_camera_semantic": plan["motion"]["name"],
        "camera_rejected": False, "camera_relation": relation, "source_frame_indices": indices,
        "dynamic_points_per_frame": counts, "fps": fps, "frames": frames,
        **{k: v for k, v in stats.items() if k != "coverages"}}
    out_mp4.with_name("geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out_mp4)
    return report
