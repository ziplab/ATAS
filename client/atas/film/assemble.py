"""Final film assembly: one normalized MP4 plus a per-shot manifest.

Film delivery is independent of memory admission. Every delivered shot is
included in story order whether or not its observation entered the world state;
the manifest reports the two decisions separately.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from atas.runner.recon_ledger import write_json_atomic
WIDTH, HEIGHT, FPS, SAMPLE_RATE = 1920, 1080, 24, 48000


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _has_audio(video: Path) -> bool:
    probe = shutil.which("ffprobe")
    if probe:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=index", "-of", "csv=p=0", str(video)],
                             capture_output=True, text=True, timeout=60)
        return bool(out.stdout.strip())
    out = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(video)], capture_output=True, text=True, timeout=60)
    return "Audio:" in out.stderr


def normalize(video: Path, output: Path) -> Path:
    """Re-encode one clip to the common stream layout so concatenation is exact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
             f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p")
    command = [_ffmpeg(), "-v", "error", "-y", "-i", str(video)]
    if not _has_audio(video):
        command += ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}"]
    command += ["-map", "0:v:0", "-map", "0:a:0" if _has_audio(video) else "1:a:0", "-vf", scale,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-ar", str(SAMPLE_RATE), "-ac", "2", "-shortest", str(output)]
    temporary = output.with_suffix(".partial.mp4")
    command[-1] = str(temporary)
    proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if proc.returncode or not temporary.is_file():
        raise RuntimeError("clip normalization failed: " + proc.stderr[-1500:])
    temporary.replace(output)
    return output


def assemble(story_dir: Path, rows: list[dict[str, Any]], *, title: str = "") -> dict[str, Any]:
    """Write ``final.mp4`` and ``manifest.json`` for the shots delivered so far."""
    story_dir = Path(story_dir)
    delivered = [r for r in rows if r.get("video") and Path(r["video"]).is_file()]
    missing = [r["shot_id"] for r in rows if r not in delivered]
    manifest = {
        "schema_version": 2, "title": title, "final_video": None,
        "shots_total": len(rows), "shots_delivered": len(delivered), "missing_shots": missing,
        "film_review_passed": sum(bool(r.get("f")) for r in rows),
        "registered": sum(bool(r.get("r")) for r in rows),
        "admitted_to_world": sum(bool(r.get("admitted")) for r in rows),
        "shots": rows,
    }
    if delivered:
        cache = story_dir / "film" / "normalized"
        parts = []
        for row in delivered:
            source = Path(row["video"])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            target = cache / f"{row['shot_id']}_{digest}.mp4"
            if not target.is_file():
                normalize(source, target)
            parts.append(target)
        listing = story_dir / "film" / "concat.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
        final = story_dir / "final.mp4"
        temporary = story_dir / "final.partial.mp4"
        proc = subprocess.run([_ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0",
                               "-i", str(listing), "-c", "copy", str(temporary)],
                              capture_output=True, text=True, timeout=900)
        if proc.returncode or not temporary.is_file():
            raise RuntimeError("final concatenation failed: " + proc.stderr[-1500:])
        temporary.replace(final)
        manifest["final_video"] = str(final)
        manifest["complete"] = not missing
    else:
        manifest["complete"] = False
    write_json_atomic(story_dir / "manifest.json", manifest)
    return manifest
