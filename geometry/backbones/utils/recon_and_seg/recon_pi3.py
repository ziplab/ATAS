from math import sqrt
from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as vT

from pi3.models.pi3 import Pi3
from pi3.models.pi3x import Pi3X
from utils.media import focal_to_intrinsics, resize_2d
from utils.misc import cleanup
from utils.recon_and_seg.recover_focal_shift import recover_focal_shift


def init_pi3(
    model_id: str = "yyfz233/Pi3X",
    device: Union[torch.device, str] = "cuda",
    dtype: torch.dtype | None = None,
):
    print(f"Loading Pi3(X) model with model_id=`{model_id}`")
    model_class = Pi3X if model_id.rstrip("/").lower().endswith("pi3x") else Pi3
    model = model_class.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
    return model


def preprocess_frames_pi3(video: npt.NDArray[np.uint8], pixel_limit: int = 255000):  # Pixel limit is approximately 384p
    num_frames, height, width, _ = video.shape

    scale = sqrt(pixel_limit / (height * width)) if height * width > pixel_limit else 1.0
    new_height, new_width = height * scale, width * scale
    new_height_14, new_width_14 = round(new_height / 14), round(new_width / 14)
    while (new_height_14 * 14) * (new_width_14 * 14) > pixel_limit:
        if new_height_14 / new_width_14 > height / width:
            new_height_14 -= 1
        else:
            new_width_14 -= 1
    new_height, new_width = max(1, new_height_14) * 14, max(1, new_width_14) * 14

    frames_tensor = []
    to_tensor = vT.ToTensor()
    for i in range(num_frames):
        frame = Image.fromarray(video[i])
        frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
        frame = to_tensor(frame)
        frames_tensor.append(frame)
    frames = torch.stack(frames_tensor, dim=0)
    return frames, (new_height, new_width)


def run_pi3(
    video: npt.NDArray[np.uint8],
    model: Pi3 | Pi3X,
    pixel_limit: int = 255000,
    vo_chunk_size: int = 0,
    vo_overlap: int = 6,
    resize_output: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[torch.device, str] = "cuda",
    verbose: bool = True,
):
    cleanup()
    num_frames, height, width, _ = video.shape

    def print_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Run Pi3(X) inference
    print_verbose(f"Running Pi3(X) inference ...")
    frames, (height_new, width_new) = preprocess_frames_pi3(video, pixel_limit=pixel_limit)
    print_verbose(f"Pi3 processed resolution: {width_new}x{height_new}")

    if vo_chunk_size > 0:
        # Pi3XVO transfers one chunk at a time. Keeping the complete input on
        # the CPU prevents long sequences from consuming GPU memory linearly.
        frames = frames.to(dtype=dtype, device="cpu")
        if vo_overlap < 1 or vo_overlap >= vo_chunk_size:
            raise ValueError(
                f"Pi3XVO overlap must satisfy 1 <= overlap < chunk_size, got "
                f"overlap={vo_overlap}, chunk_size={vo_chunk_size}"
            )
        from pi3.pipe.pi3x_vo_v2 import Pi3XVO

        print_verbose(
            f"Running Pi3XVO in chunks of {vo_chunk_size} frames "
            f"with {vo_overlap}-frame overlap"
        )
        results = Pi3XVO(model)(
            frames[None],
            chunk_size=vo_chunk_size,
            overlap=vo_overlap,
            dtype=dtype,
            device=device,
        )
        local_points, camera_poses = _camera_local_points_from_vo(results)
    else:
        frames = frames.to(dtype=dtype, device=device)
        with torch.amp.autocast(device, dtype=dtype):
            results = model(frames[None])
        local_points = results["local_points"]
        camera_poses = results["camera_poses"]

    local_points = local_points[0].detach().cpu().to(torch.float32)
    local_points = local_points.numpy().astype(np.float32)
    from utils.recon_and_seg.pi3_geometry import native_rgbd
    depths,intrinsics,ray_errors=native_rgbd(local_points)
    print_verbose(f'Pi3 native RGB-D conversion: max pinhole residual {max(ray_errors):.4f}px; no z-shift')
    cam_c2w = camera_poses[0].detach().cpu().numpy().astype(np.float32)  # f 4 4

    print_verbose(
        f"Depth map distribution: min={depths.min():.5f}, median={np.median(depths):.5f}, "
        f"mean={depths.mean():.5f}, max={depths.max():.5f}"
    )

    # Align camera poses so first frame is identity transformation
    cam_c2w = np.linalg.inv(cam_c2w[0])[None] @ cam_c2w

    if resize_output:
        # Resize outputs to original video resolution
        depths = resize_2d(depths, height, width, mode="bilinear", crop=False, inverse=True)

        intrinsics[:, 0::2] *= width / width_new
        intrinsics[:, 1::2] *= height / height_new
        print_verbose(
            f"Scaled intrinsics (first frame): fx={intrinsics[0, 0]:.3f}, fy={intrinsics[0, 1]:.3f}, "
            f"cx={intrinsics[0, 2]:.3f}, cy={intrinsics[0, 3]:.3f}"
        )

    del results
    cleanup()

    return depths, cam_c2w, intrinsics


def _camera_local_points_from_vo(
    results: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Pi3XVO's Sim3-aligned outputs to rigid cameras and local points."""
    points = results["points"]
    affine_poses = results["camera_poses"]
    linear = affine_poses[..., :3, :3]
    scales = torch.linalg.vector_norm(linear, dim=-2).mean(dim=-1).clamp_min(1e-8)
    rotations = linear / scales[..., None, None]
    translations = affine_poses[..., :3, 3]
    centered = points - translations[..., None, None, :]
    local_points = torch.einsum(
        "bnij,bnhwj->bnhwi",
        rotations.transpose(-2, -1),
        centered,
    )
    camera_poses = affine_poses.clone()
    camera_poses[..., :3, :3] = rotations
    return local_points, camera_poses
