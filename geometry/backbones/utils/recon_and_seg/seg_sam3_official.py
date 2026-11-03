from typing import List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from utils.misc import cleanup


def init_sam3_video(
    checkpoint_path: str = "checkpoints/sam3/sam3.pt",
    offload_video_to_cpu: bool = False,
    offload_output_to_cpu_for_eval: bool = False,
    image_size: int = 1008,
    dtype: torch.dtype = torch.float32,
):
    print(f"Loading Segment Anything 3 model (from official implementation/repo).")
    video_predictor = build_sam3_video_predictor(
        checkpoint_path,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_output_to_cpu_for_eval=offload_output_to_cpu_for_eval,
        image_size=image_size,
        dtype=dtype,
    )
    return video_predictor


def run_sam3_video(
    video: npt.NDArray[np.uint8],
    video_predictor: Sam3VideoPredictorMultiGPU,
    keywords: List[str],
    collect_instances: bool = True,
):
    cleanup()

    num_frames, height, width, _ = video.shape
    keywords = [keyword.strip() for keyword in keywords]

    print(f"Running SAM3 inference with keywords={keywords}")

    # Convert numpy video frames to PIL images (the official SAM3 API accepts this as resource_path)
    video_pil = [Image.fromarray(video[i]) for i in range(num_frames)]

    response = video_predictor.handle_request({"type": "start_session", "resource_path": video_pil})  # Start session
    session_id = response["session_id"]
    del video_pil

    dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    seg_frames = [[] for _ in range(num_frames)]
    obj_id_offset = 0

    peak_tracked_objects = 0
    processed_frames = 0
    try:
        for keyword in keywords:
            video_predictor.handle_request(
                {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": keyword},
            )
            max_obj_id_this_keyword = -1
            for result in video_predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "max_frame_num_to_track": num_frames,
            }):
                frame_idx = result["frame_index"]
                outputs = result["outputs"]
                if outputs is None:
                    continue

                frame_stats = outputs.get("frame_stats") or {}
                tracked_objects = int(frame_stats.get("num_obj_tracked", 0))
                peak_tracked_objects = max(peak_tracked_objects, tracked_objects)
                processed_frames = max(processed_frames, frame_idx + 1)

                out_obj_ids = outputs["out_obj_ids"]
                out_probs = outputs["out_probs"]
                out_boxes_xywh = outputs["out_boxes_xywh"]
                out_binary_masks = outputs["out_binary_masks"]

                if len(out_obj_ids) == 0:
                    continue

                max_obj_id_this_keyword = max(max_obj_id_this_keyword, int(out_obj_ids.max()))

                if collect_instances:
                    boxes_xyxy = np.empty((len(out_obj_ids), 4), dtype=np.float32)
                    boxes_xyxy[:, 0] = out_boxes_xywh[:, 0] * width
                    boxes_xyxy[:, 1] = out_boxes_xywh[:, 1] * height
                    boxes_xyxy[:, 2] = (out_boxes_xywh[:, 0] + out_boxes_xywh[:, 2]) * width
                    boxes_xyxy[:, 3] = (out_boxes_xywh[:, 1] + out_boxes_xywh[:, 3]) * height

                    for i in range(len(out_obj_ids)):
                        seg_frames[frame_idx].append({
                            "id": int(out_obj_ids[i]) + obj_id_offset,
                            "keyword": keyword,
                            "score": float(out_probs[i]),
                            "box_xyxy": boxes_xyxy[i],
                            "mask": out_binary_masks[i],
                        })
                dynamic_mask[frame_idx] |= out_binary_masks.any(axis=0)

            if max_obj_id_this_keyword >= 0:
                obj_id_offset += max_obj_id_this_keyword + 1
    finally:
        print(
            f"SAM3 tracking summary: processed_frames={processed_frames}/{num_frames}, "
            f"peak_tracked_objects={peak_tracked_objects}"
        )
        video_predictor.handle_request(dict(type="close_session", session_id=session_id))

    print(f"Dynamic mask coverage: {dynamic_mask.astype(np.int64).sum() / np.prod(dynamic_mask.shape) * 100:.3f}%")
    cleanup()

    return dynamic_mask, seg_frames
