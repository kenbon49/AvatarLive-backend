"""Avatar video loading and face-box caching for accelerated inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray


log = logging.getLogger(__name__)
RGBFrame = NDArray[np.uint8]


class AvatarPreparationError(RuntimeError):
    """Raised when an avatar video cannot be turned into inference frames."""


@dataclass(frozen=True)
class AvatarFrame:
    """The small interface consumed by :class:`MuseTalkEngine`."""

    profile_id: str
    action: str
    frame_index: int
    frame: RGBFrame
    face_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class AvatarSpec:
    profile_id: str
    video_path: Path
    clip_start_seconds: float = 0.0
    clip_end_seconds: float | None = None
    ping_pong: bool = True

    def __post_init__(self) -> None:
        if self.clip_start_seconds < 0:
            raise ValueError("clip_start_seconds must not be negative")
        if self.clip_end_seconds is not None and self.clip_end_seconds <= self.clip_start_seconds:
            raise ValueError("clip_end_seconds must be greater than clip_start_seconds")


class AvatarProfile:
    def __init__(self, spec: AvatarSpec, frames: list[AvatarFrame], fps: float) -> None:
        if not frames:
            raise AvatarPreparationError(f"avatar {spec.profile_id!r} contains no frames")
        self.spec = spec
        self.frames = frames
        self.fps = float(fps)

    def sequence(self, start: int, count: int) -> list[AvatarFrame]:
        """Return a ping-pong sequence without duplicating the end frames."""

        if len(self.frames) == 1:
            return [self.frames[0]] * count
        if not self.spec.ping_pong:
            return [self.frames[(start + offset) % len(self.frames)] for offset in range(count)]
        cycle = self.frames + self.frames[-2:0:-1]
        return [cycle[(start + offset) % len(cycle)] for offset in range(count)]

    @property
    def cycle_length(self) -> int:
        if len(self.frames) == 1 or not self.spec.ping_pong:
            return len(self.frames)
        return 2 * len(self.frames) - 2


class AvatarLoader:
    """Decode avatar MP4s and cache the comparatively expensive pose pass."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        target_fps: float = 25.0,
        max_height: int = 720,
        bbox_shift: int = 5,
        detection_stride: int = 5,
    ) -> None:
        if target_fps <= 0 or max_height <= 0 or detection_stride <= 0:
            raise ValueError("fps, max_height, and detection_stride must be positive")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.target_fps = float(target_fps)
        self.max_height = int(max_height)
        self.bbox_shift = int(bbox_shift)
        self.detection_stride = int(detection_stride)

    def _signature(self, spec: AvatarSpec) -> str:
        path = spec.video_path.resolve(strict=True)
        stat = path.stat()
        value = {
            "format": 1,
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "target_fps": self.target_fps,
            "max_height": self.max_height,
            "bbox_shift": self.bbox_shift,
            "detection_stride": self.detection_stride,
            "clip_start_seconds": spec.clip_start_seconds,
            "clip_end_seconds": spec.clip_end_seconds,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()

    def _decode(self, spec: AvatarSpec) -> tuple[list[RGBFrame], float]:
        path = spec.video_path
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise AvatarPreparationError(f"cannot open avatar video: {path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or self.target_fps)
        frames: list[RGBFrame] = []
        source_index = 0
        next_time = spec.clip_start_seconds
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                timestamp = source_index / source_fps
                source_index += 1
                if spec.clip_end_seconds is not None and timestamp >= spec.clip_end_seconds:
                    break
                if timestamp + 1e-8 < spec.clip_start_seconds:
                    continue
                if timestamp + 1e-8 < next_time:
                    continue
                height, width = bgr.shape[:2]
                if height > self.max_height:
                    scale = self.max_height / height
                    bgr = cv2.resize(
                        bgr,
                        (max(2, round(width * scale)), self.max_height),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
                next_time += 1.0 / self.target_fps
        finally:
            capture.release()
        if not frames:
            raise AvatarPreparationError(f"avatar video has no decodable frames: {path}")
        return frames, self.target_fps

    @staticmethod
    def _interpolate_boxes(
        sample_indices: list[int], boxes: Iterable[Iterable[float]], count: int
    ) -> NDArray[np.int32]:
        samples = np.asarray(list(boxes), dtype=np.float32)
        if samples.shape != (len(sample_indices), 4):
            raise AvatarPreparationError("face detector returned malformed boxes")
        if np.any(samples[:, 2] <= samples[:, 0]) or np.any(samples[:, 3] <= samples[:, 1]):
            raise AvatarPreparationError("face detector failed on an avatar key frame")
        positions = np.arange(count, dtype=np.float32)
        result = np.empty((count, 4), dtype=np.int32)
        for coordinate in range(4):
            result[:, coordinate] = np.rint(
                np.interp(positions, sample_indices, samples[:, coordinate])
            ).astype(np.int32)
        return result

    def _detect_boxes(self, frames: list[RGBFrame]) -> NDArray[np.int32]:
        # mmpose initialization is intentionally delayed until preparation.
        from musetalk.utils.preprocessing import get_landmark_and_bbox

        indices = list(range(0, len(frames), self.detection_stride))
        if indices[-1] != len(frames) - 1:
            indices.append(len(frames) - 1)
        with tempfile.TemporaryDirectory(prefix="musetalk-avatar-") as directory:
            paths: list[str] = []
            for position, index in enumerate(indices):
                path = Path(directory) / f"{position:06d}.jpg"
                if not cv2.imwrite(str(path), frames[index][:, :, ::-1]):
                    raise AvatarPreparationError(f"failed to write temporary frame {path}")
                paths.append(str(path))
            boxes, _ = get_landmark_and_bbox(paths, self.bbox_shift)
        return self._interpolate_boxes(indices, boxes, len(frames))

    def load(self, spec: AvatarSpec) -> AvatarProfile:
        path = spec.video_path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise AvatarPreparationError(f"avatar is not a file: {path}")
        normalized_spec = AvatarSpec(
            spec.profile_id,
            path,
            spec.clip_start_seconds,
            spec.clip_end_seconds,
            spec.ping_pong,
        )
        signature = self._signature(normalized_spec)
        cache_path = self.cache_dir / f"{spec.profile_id}-boxes.npz"
        frames, fps = self._decode(normalized_spec)
        boxes: NDArray[np.int32] | None = None
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["signature"].item()) == signature:
                        candidate = cached["boxes"].astype(np.int32, copy=False)
                        if candidate.shape == (len(frames), 4):
                            boxes = candidate.copy()
                            log.info("loaded avatar face boxes from %s", cache_path)
            except Exception:
                log.exception("ignoring invalid avatar box cache %s", cache_path)
        if boxes is None:
            boxes = self._detect_boxes(frames)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                np.savez_compressed(handle, signature=np.asarray(signature), boxes=boxes)
            log.info("saved avatar face boxes to %s", cache_path)
        avatar_frames = [
            AvatarFrame(spec.profile_id, "loop", index, frame, tuple(int(v) for v in box))
            for index, (frame, box) in enumerate(zip(frames, boxes))
        ]
        return AvatarProfile(normalized_spec, avatar_frames, fps)
