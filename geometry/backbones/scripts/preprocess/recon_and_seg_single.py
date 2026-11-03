from argparse import ArgumentParser
from os import path
import json
import time
from pathlib import Path

import numpy as np
import torch

from utils.media import (
    concat_2d, crop_and_resize_video, depths_to_disparity_video, load_video, mask_to_video,
    save_cameras, save_clips, save_depths, save_masks, save_mp4_with_gif, save_video, slice_center_frames,
)
from utils.misc import cleanup


@torch.no_grad()
def main(args):
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    Path(args.output_folder, 'reconstruction_effective.json').write_text(json.dumps({
        'pi3_pixel_limit': args.pi3_pixel_limit,
        'pi3_vo_chunk_size': args.pi3_vo_chunk_size,
        'pi3_vo_overlap': args.pi3_vo_overlap,
        'width': args.width, 'height': args.height,
        'num_frames': args.num_frames, 'low_vram': args.low_vram,
    }, indent=2))
    phase_started = time.perf_counter()
    phase_times = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    def phase(name):
        nonlocal phase_started
        record = {'stage': name}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Peak memory this process reserved during the phase (not card-level usage).
            record['gpu_reserved_peak_mib'] = round(torch.cuda.max_memory_reserved() / 2**20)
            torch.cuda.reset_peak_memory_stats()
        now = time.perf_counter()
        phase_times[name] = now-phase_started
        record['seconds'] = phase_times[name]
        print('AVWM_RECON_PHASE '+json.dumps(record),flush=True)
        phase_started = now
    # Load source video (center-sliced to args.num_frames, cropped/resized to target resolution)
    video_src, fps = load_video(args.video_path)
    num_src_input_frames, height_src_input, width_src_input, _ = video_src.shape
    assert num_src_input_frames >= args.num_frames,\
        f"Source video must have at least args.num_frames={args.num_frames} frames, got {num_src_input_frames} frames."

    video_src = slice_center_frames(video_src, args.num_frames)
    video_src = crop_and_resize_video(video_src, args.height, args.width, resample="lanczos")
    print(
        f"Sliced and resized source video from {num_src_input_frames}x{width_src_input}x{height_src_input} "
        f"to {args.num_frames}x{args.width}x{args.height}",
    )

    # Optionally load DSE (dynamic scene expansion) casual scene capture, subsample evenly, and concatenate
    clips = None
    if args.dse_video_path is not None:
        video_dse, _ = load_video(args.dse_video_path)  # DSE fps ignored, combined video uses source fps
        num_dse_input_frames, height_dse_input, width_dse_input, _ = video_dse.shape
        num_dse_frames = args.num_dse_frames if args.num_dse_frames is not None else num_dse_input_frames
        assert num_dse_input_frames >= num_dse_frames,\
            f"DSE video must have at least num_dse_frames={num_dse_frames} frames, got {num_dse_input_frames} frames."

        dse_indices = np.linspace(0, num_dse_input_frames - 1, num_dse_frames).round().astype(int)
        video_dse = video_dse[dse_indices]
        video_dse = crop_and_resize_video(video_dse, args.height, args.width, resample="lanczos")
        print(
            f"Subsampled and resized DSE video from {num_dse_input_frames}x{width_dse_input}x{height_dse_input} "
            f"to {num_dse_frames}x{args.width}x{args.height}",
        )
        video = np.concatenate([video_src, video_dse], axis=0)
        clips = {"src": (0, args.num_frames), "dse": (args.num_frames, args.num_frames + num_dse_frames)}
    else:
        video = video_src

    phase('video_decode_resize')

    if args.recon_method == "da3":
        raise SystemExit("this snapshot ships only the Pi3 reconstruction path; use --recon_method pi3")

    elif args.recon_method == "pi3":
        from utils.recon_and_seg.recon_pi3 import init_pi3, run_pi3
        # Keep the original FP32 checkpoint. Memory is controlled by VO
        # chunking, CPU accumulation, reduced input pixels, and autocast.
        pi3_model = init_pi3(model_id=args.pi3_model_id, device="cuda")
        phase('pi3_import_and_load')
        depths, cam_c2w, intrinsics = run_pi3(
            video, model=pi3_model, pixel_limit=args.pi3_pixel_limit, resize_output=True,
            vo_chunk_size=args.pi3_vo_chunk_size, vo_overlap=args.pi3_vo_overlap,
            dtype=torch.bfloat16, device="cuda",
        )
        del pi3_model
        phase('pi3_inference_and_rgbd')

    if args.scene_scale != 1.0:
        depths *= args.scene_scale
        cam_c2w[:, :3, 3] *= args.scene_scale

    cleanup()
    phase('geometry_cleanup')

    # Dynamic mask segmentation with Segment Anything 3
    num_frames, height, width, _ = video.shape
    seg_frames = [[] for _ in range(num_frames)]
    if len(args.seg_keywords) == 0:
        dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
        sky_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    elif "_all_" in args.seg_keywords:
        dynamic_mask = np.ones((num_frames, height, width), dtype=np.bool_)
        sky_mask = np.zeros((num_frames, height, width), dtype=np.bool_)

    elif args.sam3_implementation == "official":
        from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video
        sam3_video_predictor = init_sam3_video(
            checkpoint_path=args.sam3_checkpoint_path,
            offload_video_to_cpu=args.sam3_offload_video_to_cpu,
            offload_output_to_cpu_for_eval=args.sam3_offload_output_to_cpu,
            image_size=args.sam3_image_size,
            dtype=torch.float32,
        )
        phase('sam3_import_and_load')
        dynamic_mask, seg_frames = run_sam3_video(
            video,
            sam3_video_predictor,
            args.seg_keywords,
            collect_instances=(args.save_vis or args.save_entity_masks),
        )
        phase('sam3_dynamic')
        if args.recon_method == "pi3":  # Unlike DA3, Pi3 doesn't segment the sky, so we have to do it with SAM3
            sky_mask, _ = run_sam3_video(
                video,
                sam3_video_predictor,
                ["sky"],
                collect_instances=False,
            )
            phase('sam3_sky')
        del sam3_video_predictor

    elif args.sam3_implementation == "transformers":
        # Requires transformers >= 5.0.0, which xfuser==0.45.0 doesn't support
        from utils.recon_and_seg.seg_sam3_transformers import init_sam3_video, run_sam3_video
        dtype = torch.bfloat16
        device = "cuda"
        sam3_model, sam3_processor = init_sam3_video(args.sam3_model_id, dtype=dtype, device=device)
        dynamic_mask, seg_frames =\
            run_sam3_video(video, sam3_model, sam3_processor, args.seg_keywords, dtype=dtype, device=device)
        if args.recon_method == "pi3":  # Unlike DA3, Pi3 doesn't segment the sky, so we have to do it with SAM3
            sky_mask, _ = run_sam3_video(video, sam3_model, sam3_processor, ["sky"], dtype=dtype, device=device)
        del sam3_model, sam3_processor

    cleanup()
    phase('segmentation_cleanup')

    # Save results
    save_video(path.join(args.output_folder, "video.mp4"), video, fps=fps, quality=9)
    phase('save_rgb_video')
    save_depths(path.join(args.output_folder, "depths"), depths, dtype=np.float16)
    phase('save_depths')
    save_cameras(path.join(args.output_folder, "cameras.npz"), cam_c2w, intrinsics)
    save_masks(path.join(args.output_folder, "dynamic_mask"), dynamic_mask)
    phase('save_cameras_dynamic_mask')
    # AVWM_ENTITY_MASK_EXPORT_V1
    if args.save_entity_masks:
        from scripts.preprocess.avwm_entity_mask_export import export_entity_masks
        export_entity_masks(seg_frames, __import__('pathlib').Path(args.output_folder),
                            frame_shape=(height, width))
        phase('save_entity_masks')
    save_masks(path.join(args.output_folder, "sky_mask"), sky_mask)
    phase('save_sky_masks')
    Path(args.output_folder,'reconstruction_timings.json').write_text(json.dumps(phase_times,indent=2))
    if clips is not None:
        save_clips(path.join(args.output_folder, "clips.json"), clips)

    if args.save_vis:  # Save visualization as video
        depths_video = depths_to_disparity_video(depths, ~sky_mask, use_colormap=True)
        sky_mask_video = mask_to_video(sky_mask)

        if len(args.seg_keywords) == 0 or "_all_" in args.seg_keywords:
            seg_overlay_video = video
        else:
            from utils.recon_and_seg.seg_sam3_utils import overlay_instances_on_video
            seg_overlay_video = overlay_instances_on_video(video, seg_frames, alpha=0.5)
        dynamic_mask_video = mask_to_video(dynamic_mask)

        vis_video = concat_2d([[depths_video, sky_mask_video], [seg_overlay_video, dynamic_mask_video]])
        save_mp4_with_gif(
            path.join(args.output_folder, "vis"), vis_video, fps=fps, quality=5,
            gif_folder="" if args.save_vis_gif else None,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--dse_video_path", default=None, type=str)  # Optional casual scene capture for DSE
    parser.add_argument("--output_folder", required=True, type=str)
    parser.add_argument("--seg_keywords", type=str, nargs="*", default=["_all_"])

    parser.add_argument("--save_entity_masks", action="store_true", help="persist unverified per-keyword SAM3 instance masks")
    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--save_vis_gif", action="store_true", default=False)

    parser.add_argument("--recon_method", default="pi3", type=str, choices=("pi3", "da3"))
    parser.add_argument("--pi3_model_id", default="yyfz233/Pi3X", type=str)
    parser.add_argument("--da3_model_id", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1", type=str)
    parser.add_argument("--sam3_implementation", default="official", choices=("official", "transformers"))
    parser.add_argument("--sam3_model_id", default="facebook/sam3", type=str)
    parser.add_argument("--sam3_checkpoint_path", default="checkpoints/sam3/sam3.pt", type=str)
    parser.add_argument("--sam3_offload_video_to_cpu", action="store_true", default=False)
    parser.add_argument("--sam3_offload_output_to_cpu", action="store_true", default=False)
    parser.add_argument("--sam3_image_size", default=1008, type=int)
    parser.add_argument("--low_vram", action="store_true", default=False)

    parser.add_argument("--height", default=720, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--num_frames", default=121, type=int)
    parser.add_argument("--num_dse_frames", default=None, type=int)  # None = use all DSE input frames
    parser.add_argument("--pi3_pixel_limit", default=255000, type=int)
    parser.add_argument("--pi3_vo_chunk_size", default=0, type=int)
    parser.add_argument("--pi3_vo_overlap", default=6, type=int)
    parser.add_argument("--da3_process_res", default=-1, type=int)
    parser.add_argument("--scene_scale", default=1.0, type=float)  # Scale the depths and cam_c2w of 4D reconstruction

    args = parser.parse_args()

    if args.pi3_pixel_limit <= 0:  # Process Pi3(X) at maximum (video) resolution
        args.pi3_pixel_limit = args.height * args.width
    if args.da3_process_res <= 0:  # Process DA3 at maximum (video) resolution
        args.da3_process_res = args.width

    if args.low_vram:
        # Keep inference below the service's low-memory target without changing
        # checkpoint weights. Keep SAM3's native 1008px geometry so checkpoint
        # RoPE buffers remain shape-compatible; use CPU offload for memory.
        args.sam3_image_size = 1008
        if args.pi3_pixel_limit == 255000:
            args.pi3_pixel_limit = 160000
        if args.pi3_vo_chunk_size == 48:
            args.pi3_vo_chunk_size = 24

    main(args)
