"""Event-memory sources: registered observations replayed in scene coordinates."""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from atas_geometry.memory.vista4d_layers import ReconStore
from atas_geometry.service.paths import l0_dir, l1_path
def identifier(value, field):
    value = str(value or "")
    if not re.fullmatch(r"[\w-]+", value) or value in {".", ".."}:
        raise ValueError(f"invalid {field}")
    return value


def event_window(payload, source_duration):
    start = float(payload.get("source_time_start", 0.0))
    end = float(payload.get("source_time_end", source_duration))
    duration = float(payload.get("duration", end - start))
    if not all(math.isfinite(x) for x in (start, end, duration, source_duration)):
        raise ValueError("event times must be finite")
    if not 0 <= start < end <= source_duration or duration <= 0:
        raise ValueError("source time interval must lie within the source observation")
    if not math.isclose(duration, end - start, abs_tol=1e-6):
        raise ValueError("recapture preserves event speed: duration must equal source interval")
    return start, end, duration


def record_observation(l0, payload, fuse):
    """Persist identity, duration and the exact raw-L0 -> L1 similarity transform."""
    transform = None
    if fuse.get("home"):
        transform = {"scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0., 0., 0.]}
    elif fuse.get("merged") is True:
        transform = fuse.get("transform")
        if transform is None and (l0 / "cameras_pi3_raw.npz").is_file():
            with np.load(l0 / "cameras_pi3_raw.npz") as raw, np.load(l0 / "cameras.npz") as aligned:
                a, b = raw["cam_c2w"][0], aligned["cam_c2w"][0]
            scale = float(fuse["align_scale"])
            rotation = b[:3, :3] @ a[:3, :3].T
            translation = b[:3, 3] - scale * (rotation @ a[:3, 3])
            transform = {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()}
    metadata = {key: payload.get(key) for key in ("run_id", "story_id", "shot_id", "scene_id", "event_id", "duration")}
    if payload.get("video_path"):
        import cv2
        video = cv2.VideoCapture(str(payload["video_path"]))
        try:
            metadata["original_frame_count"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata["source_fps"] = float(video.get(cv2.CAP_PROP_FPS))
        finally:
            video.release()
        if metadata["original_frame_count"] < 1 or metadata["source_fps"] <= 0:
            raise ValueError("could not record source-frame timing")
        import hashlib
        digest = hashlib.sha256()
        with Path(payload["video_path"]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        metadata["original_video_sha256"] = digest.hexdigest()
    metadata.update(schema_version=1, registered=transform is not None, raw_to_scene=transform)
    path = l0 / "observation.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def load_source(root, payload):
    ids = {key: identifier(payload.get(key), key) for key in
           ("run_id", "story_id", "scene_id", "shot_id", "source_shot_id", "source_event_id")}
    if ids["source_shot_id"] == ids["shot_id"]:
        raise ValueError("recapture must not overwrite its source shot")
    if payload.get("event_id") != ids["source_event_id"]:
        raise ValueError("recapture event_id must equal source_event_id")
    source = l0_dir(root, ids["run_id"], ids["story_id"], ids["source_shot_id"])
    meta_file = source / "observation.json"
    if not meta_file.is_file():
        raise ValueError("source observation lacks registration metadata; register it before recapture")
    meta = json.loads(meta_file.read_text())
    for key in ("run_id", "story_id", "scene_id"):
        if meta.get(key) != ids[key]:
            raise ValueError(f"source observation {key} mismatch")
    if meta.get("shot_id") != ids["source_shot_id"] or meta.get("event_id") != ids["source_event_id"]:
        raise ValueError("source shot/event mismatch")
    if meta.get("registered") is not True or not meta.get("raw_to_scene"):
        raise ValueError("source L0 was not registered into scene L1")
    event_window(payload, float(meta["duration"]))
    return source, meta, l1_path(root, ids["run_id"], ids["story_id"], ids["scene_id"])


class SceneEventRecon(ReconStore):
    """Unproject with RAW cameras, then transform positions AND depth scale once."""
    def __init__(self, path, meta):
        super().__init__(path, duration=float(meta["duration"]), exclude_sky=True, mask_dilate=0)
        raw_path = path / "cameras_pi3_raw.npz"
        if raw_path.is_file():
            with np.load(raw_path) as raw:
                self.cam_c2w = raw["cam_c2w"].astype(np.float64)
        self.raw_cameras = self.cam_c2w.copy()
        self._source_meta=meta
        transform = meta["raw_to_scene"]
        self.scale = float(transform["scale"])
        self.rotation = np.asarray(transform["rotation"], dtype=np.float64)
        self.translation = np.asarray(transform["translation"], dtype=np.float64)
        if (not math.isfinite(self.scale) or self.scale <= 0
                or self.rotation.shape != (3, 3) or self.translation.shape != (3,)
                or not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all()
                or not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-4)):
            raise ValueError("invalid source similarity transform")
        self.cam_c2w[:, :3, :3] = self.rotation @ self.raw_cameras[:, :3, :3]
        self.cam_c2w[:, :3, 3] = self.scale * self.raw_cameras[:, :3, 3] @ self.rotation.T + self.translation
        if self.count <= 0 or len(self.dynamic_files) != self.count:
            raise ValueError("event requires a dynamic mask for every reconstructed frame")

    def unproject(self, index):
        scene_cameras = self.cam_c2w
        try:
            self.cam_c2w = self.raw_cameras
            observation = super().unproject(index)
        finally:
            self.cam_c2w = scene_cameras
        xyz = self.scale * np.asarray(observation.xyz) @ self.rotation.T + self.translation
        return replace(observation, xyz=xyz.astype(np.float32))

    def depth_at(self,index):
        return super().depth_at(index)*self.scale

    def source_index(self,world_time):
        if self._source_meta.get("source_fps") and self._source_meta.get("original_frame_count"):
            fps=float(self._source_meta["source_fps"])
            offset=(int(self._source_meta["original_frame_count"])-self.count)//2
            index=round(float(world_time)*fps)-offset
            if not 0<=index<self.count: raise ValueError("event time outside reconstructed source frames")
            return index
        return super().source_index(world_time)
