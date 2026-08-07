"""Streaming session primitives built on the accelerated MuseTalk engine."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import threading
from typing import Iterator, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .avatar import AvatarProfile


PACKET_MAGIC = b"MSTK"
PACKET_VERSION = 1
PACKET_JPEG = 1
PACKET_PCM16 = 2
_PACKET_HEADER = struct.Struct("<4sBBHIIQ")


class RenderEngine(Protocol):
    batch_size: int

    def extract_audio_features(self, pcm16k: NDArray[np.float32], fps: float): ...

    def render_batch(self, audio_features: object, frames: Sequence[object]) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class RenderedBatch:
    start_frame: int
    frames: NDArray[np.uint8]
    pcm16: bytes


def encode_packet(packet_type: int, sequence: int, pts_us: int, payload: bytes) -> bytes:
    """Encode a self-delimiting WebSocket media packet."""

    header = _PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        int(packet_type),
        0,
        int(sequence),
        len(payload),
        int(pts_us),
    )
    return header + payload


def decode_packet(packet: bytes) -> tuple[int, int, int, bytes]:
    if len(packet) < _PACKET_HEADER.size:
        raise ValueError("media packet is shorter than its header")
    magic, version, packet_type, _flags, sequence, size, pts_us = _PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise ValueError("unsupported media packet")
    payload = packet[_PACKET_HEADER.size :]
    if len(payload) != size:
        raise ValueError("media packet payload length differs from header")
    return packet_type, sequence, pts_us, payload


def pcm16le_to_float32(data: bytes) -> NDArray[np.float32]:
    if len(data) % 2:
        raise ValueError("PCM s16le payload must contain complete 16-bit samples")
    samples = np.frombuffer(data, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


class StreamingRenderer:
    """Extract audio once, then yield mouth-rendered batches as soon as ready."""

    def __init__(self, engine: RenderEngine, *, fps: float = 25.0) -> None:
        self.engine = engine
        self.fps = float(fps)
        self._positions: dict[str, int] = {}
        self._position_lock = threading.Lock()

    def render(self, pcm16le: bytes, profile: AvatarProfile) -> Iterator[RenderedBatch]:
        pcm = pcm16le_to_float32(pcm16le)
        features = self.engine.extract_audio_features(pcm, self.fps)
        frame_count = len(features)
        if frame_count == 0:
            return
        with self._position_lock:
            start_position = self._positions.get(profile.spec.profile_id, 0)
            self._positions[profile.spec.profile_id] = (
                start_position + frame_count
            ) % profile.cycle_length
        samples_per_frame = 16000.0 / self.fps
        for start in range(0, frame_count, self.engine.batch_size):
            count = min(self.engine.batch_size, frame_count - start)
            avatar_frames = profile.sequence(start_position + start, count)
            rendered = self.engine.render_batch(features[start : start + count], avatar_frames)
            sample_start = round(start * samples_per_frame)
            sample_end = min(len(pcm), round((start + count) * samples_per_frame))
            audio = np.clip(np.rint(pcm[sample_start:sample_end] * 32768.0), -32768, 32767)
            yield RenderedBatch(start, rendered, audio.astype("<i2").tobytes())


def jpeg_bytes(frame_rgb: NDArray[np.uint8], quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode an output frame")
    return encoded.tobytes()

