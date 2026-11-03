from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atas_geometry.memory.types import AnyArray
@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    world_to_camera: AnyArray
    intrinsic: AnyArray
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.world_to_camera.shape != (4, 4):
            raise ValueError("world_to_camera must have shape (4, 4)")
        if self.intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape (3, 3)")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be positive")

    @property
    def camera_to_world(self) -> AnyArray:
        return np.linalg.inv(self.world_to_camera)

    @property
    def center_world(self) -> AnyArray:
        return self.camera_to_world[:3, 3]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "world_to_camera": self.world_to_camera.tolist(),
            "intrinsic": self.intrinsic.tolist(),
            "image_size": list(self.image_size),
        }


