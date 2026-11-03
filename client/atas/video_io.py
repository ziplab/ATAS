from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from atas.providers.wan.types import FrameArray
def read_video_frames(path: Path, *, empty_error: str | None = None) -> list[FrameArray]:
    frames = iio.imread(path)
    if frames.ndim != 4:
        if empty_error is not None:
            raise ValueError(empty_error)
        return []
    return [frame.astype(np.uint8) for frame in frames]
