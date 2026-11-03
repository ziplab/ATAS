from __future__ import annotations

from pathlib import Path

import numpy as np

from atas_geometry.memory.camera_query import CameraQuery
def write_camera_query_npz(query: CameraQuery, path: Path) -> None:
    cam_c2w = np.stack([camera.camera_to_world for camera in query.cameras]).astype(np.float32)
    intrinsics = np.stack(
        [
            np.asarray(
                [
                    camera.intrinsic[0, 0],
                    camera.intrinsic[1, 1],
                    camera.intrinsic[0, 2],
                    camera.intrinsic[1, 2],
                ],
                dtype=np.float32,
            )
            for camera in query.cameras
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cam_c2w=cam_c2w,
        intrinsics=intrinsics,
        cam_c2w_fsff=cam_c2w,
        intrinsics_fsff=intrinsics,
        world_times=np.asarray(query.world_times, dtype=np.float32),
    )


