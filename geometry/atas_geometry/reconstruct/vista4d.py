from __future__ import annotations

import os
import subprocess
from pathlib import Path


def single_pass_frame_limit() -> int:
    """Longest clip Pi3X can take in one pass: beyond it the conv head exceeds
    32-bit index math (observed at 300 frames, 1280x720 input)."""
    return int(os.environ.get("GEOMETRY_PI3_MAX_SINGLE_PASS_FRAMES", "160"))


def effective_pi3_settings(pixel_limit: int, chunk_size: int, low_vram: bool,
                           num_frames: int | None = None) -> tuple[int, int]:
    """Mirror Vista's low-VRAM default substitutions without changing explicit overrides.

    ``chunk_size == 0`` (whole clip in one pass) falls back to windowed VO for
    clips longer than the single-pass limit instead of crashing.
    """
    if chunk_size == 0 and num_frames is not None and num_frames > single_pass_frame_limit():
        chunk_size = 48
    if low_vram:
        pixel_limit = 160000 if pixel_limit == 255000 else pixel_limit
        chunk_size = 24 if chunk_size == 48 else chunk_size
    return pixel_limit, chunk_size


def probe_frame_count(python_bin: Path, video: Path) -> int:
    script = (
        "import cv2,sys; "
        "c=cv2.VideoCapture(sys.argv[1]); "
        "n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); "
        "c.release(); print(n)"
    )
    completed = subprocess.run(
        [str(python_bin), "-c", script, str(video)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    count = int(completed.stdout.strip())
    if count <= 0:
        raise RuntimeError(f"video has no frames: {video}")
    return count


def run_recon_and_seg(
    *,
    python_bin: Path,
    repo_dir: Path,
    video: Path,
    output_dir: Path,
    keywords: list[str],
    gpu_index: int,
    log_path: Path,
    pi3_model_id: str,
    sam3_checkpoint: Path,
    width: int,
    height: int,
    num_frames: int,
    pi3_pixel_limit: int,
    pi3_vo_chunk_size: int,
    pi3_vo_overlap: int,
    low_vram: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pi3_pixel_limit, pi3_vo_chunk_size = effective_pi3_settings(
        pi3_pixel_limit, pi3_vo_chunk_size, low_vram, num_frames=num_frames)
    command = [
        str(python_bin),
        "-m",
        "scripts.preprocess.recon_and_seg_single",
        "--video_path",
        str(video),
        "--output_folder",
        str(output_dir),
        "--seg_keywords",
        *keywords,
        "--recon_method",
        "pi3",
        "--pi3_model_id",
        pi3_model_id,
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_frames",
        str(num_frames),
        "--pi3_pixel_limit",
        str(pi3_pixel_limit),
        "--pi3_vo_chunk_size",
        str(pi3_vo_chunk_size),
        "--pi3_vo_overlap",
        str(pi3_vo_overlap),
        "--sam3_checkpoint_path",
        str(sam3_checkpoint),
        "--sam3_offload_video_to_cpu",
        "--sam3_offload_output_to_cpu",
        "--save_entity_masks",
    ]
    if low_vram:
        command.append("--low_vram")
    from atas_geometry.service.resident_models import resident_command
    from atas_geometry import GEOMETRY_ROOT
    command = resident_command(command, repo=GEOMETRY_ROOT,
                               vista_repo=repo_dir, log_path=log_path)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from atas_geometry.service.device_mapping import gpu_selector
    env["CUDA_VISIBLE_DEVICES"] = gpu_selector(gpu_index)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# physical_gpu_index={gpu_index}\n")
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(repo_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        from atas_geometry.service.resource_errors import log_resource_error
        resource_error = log_resource_error(log_path, physical_gpu_index=gpu_index, stage="vista4d_recon")
        if resource_error:
            raise resource_error
        raise RuntimeError(f"Vista4D recon exited {code}; see {log_path}")
    cameras = output_dir / "cameras.npz"
    if not cameras.is_file():
        raise FileNotFoundError(f"recon did not write {cameras}")
