from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from PIL import Image


def existing_artifact(shot_dir: Path) -> Path | None:
    direct = shot_dir / "result.mp4"
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    found = [
        path
        for path in shot_dir.glob("wan27/*/result.mp4")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


def observation_dir(story_dir: Path, shot_id: str) -> Path:
    """Folder of the take of ``shot_id`` that entered memory (a recapture source may
    be retaken); the original folder when no take was admitted."""
    story_dir = Path(story_dir)
    for folder in [story_dir / shot_id, *sorted(story_dir.glob(f"{shot_id}__take*"))]:
        if (folder / "world_publication.json").is_file():
            return folder
    return story_dir / shot_id


UPLOAD_LIMIT_BYTES = 95 * 1024**2   # the generator rejects inputs over 100 MB


def fit_upload_limit(video: Path) -> Path:
    """Return ``video``, or a bitrate-capped copy when it is too large to upload."""
    import subprocess
    video = Path(video)
    if video.stat().st_size <= UPLOAD_LIMIT_BYTES:
        return video
    capped = video.with_name(video.stem + ".capped.mp4")
    if not capped.is_file() or capped.stat().st_mtime < video.stat().st_mtime:
        partial = capped.with_name(capped.stem + ".partial.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-c:v", "libx264",
                        "-crf", "18", "-maxrate", "25M", "-bufsize", "25M", "-pix_fmt", "yuv420p",
                        "-an", str(partial)], check=True)
        partial.replace(capped)
    if capped.stat().st_size > UPLOAD_LIMIT_BYTES:
        raise ValueError(f"{video.name} still exceeds the generator's upload limit after re-encoding")
    return capped


def write_last_frame(video: Path, output: Path) -> Path:
    frames = iio.imread(video)
    if getattr(frames, "ndim", 0) != 4 or frames.shape[0] < 1:
        raise RuntimeError(f"cannot read last frame from {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frames[-1]).save(output)
    return output


def write_event_frame(video: Path, output: Path, seconds: float) -> Path:
    import imageio.v2 as imageio
    with imageio.get_reader(str(video)) as reader:
        fps = float(reader.get_meta_data()["fps"])
        frame = reader.get_data(int(round(float(seconds) * fps)))
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output)
    return output
