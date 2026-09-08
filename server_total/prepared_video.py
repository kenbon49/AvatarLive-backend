"""Persistent transparent WebM artifacts for prepared avatar speech."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence
from uuid import uuid4
import wave

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image
from scipy import ndimage


COLOR_DISTANCE_SCALE = 100 / np.sqrt(3 * 255 * 255)


def ffmpeg_executable() -> str:
    return os.getenv("FFMPEG_BINARY", "").strip() or get_ffmpeg_exe()


def cache_key(identity: dict[str, object]) -> str:
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    directory = cache_dir / key[:2]
    return directory / f"{key}.webm", directory / f"{key}.json"


def read_manifest(cache_dir: Path, key: str) -> dict[str, Any] | None:
    video_path, manifest_path = artifact_paths(cache_dir, key)
    try:
        if video_path.stat().st_size < 1024:
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("key") != key:
        return None
    return manifest


def _background_reference(rgb: np.ndarray) -> np.ndarray:
    height, width, _channels = rgb.shape
    inset_y = max(1, min(12, height // 20))
    inset_x = max(1, min(12, width // 20))
    samples = np.concatenate(
        [
            rgb[:inset_y, :inset_x].reshape(-1, 3),
            rgb[:inset_y, -inset_x:].reshape(-1, 3),
            rgb[-inset_y:, :inset_x].reshape(-1, 3),
            rgb[-inset_y:, -inset_x:].reshape(-1, 3),
        ]
    )
    return np.median(samples.astype(np.float32), axis=0)


def jpeg_to_background_removed_rgba(
    jpeg: bytes,
    *,
    tolerance: float = 4.0,
    softness: float = 6.0,
) -> tuple[bytes, int, int]:
    """Remove only key-colored regions connected to an image edge.

    Edge connectivity keeps enclosed light clothing opaque while removing the
    white studio backdrop used by the built-in avatars.
    """
    with Image.open(io.BytesIO(jpeg)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    reference = _background_reference(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - reference, axis=2)
    distance_percent = distance * COLOR_DISTANCE_SCALE
    candidate = distance_percent <= tolerance + softness
    labels, _count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    edge_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    edge_labels = edge_labels[edge_labels != 0]
    connected_background = np.isin(labels, edge_labels)

    alpha = np.full(distance_percent.shape, 255.0, dtype=np.float32)
    if softness <= 0:
        alpha[connected_background] = 0
    else:
        position = np.clip((distance_percent - tolerance) / softness, 0.0, 1.0)
        smooth = position * position * (3.0 - 2.0 * position)
        alpha[connected_background] = smooth[connected_background] * 255.0
    rgba = np.dstack((rgb, np.rint(alpha).astype(np.uint8)))
    height, width, _channels = rgba.shape
    return rgba.tobytes(), width, height


def _write_wav(path: Path, pcm16le: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm16le)


def encode_transparent_webm(
    frames: Sequence[bytes],
    pcm16le: bytes,
    fps: float,
    output_path: Path,
    *,
    tolerance: float = 4.0,
    softness: float = 6.0,
) -> dict[str, object]:
    if not frames:
        raise ValueError("prepared video contains no frames")
    if not pcm16le or len(pcm16le) % 2:
        raise ValueError("prepared video contains invalid PCM audio")
    if fps <= 0:
        raise ValueError("prepared video FPS must be positive")

    first_rgba, width, height = jpeg_to_background_removed_rgba(
        frames[0], tolerance=tolerance, softness=softness
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp.webm"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="prepared-video-") as directory:
            audio_path = Path(directory) / "audio.wav"
            _write_wav(audio_path, pcm16le)
            command = [
                ffmpeg_executable(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                "-auto-alt-ref",
                "0",
                "-deadline",
                "good",
                "-cpu-used",
                "4",
                "-crf",
                "32",
                "-b:v",
                "0",
                "-row-mt",
                "1",
                "-metadata:s:v:0",
                "alpha_mode=1",
                "-c:a",
                "libopus",
                "-b:a",
                "64k",
                "-shortest",
                str(temporary_output),
            ]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
                try:
                    assert process.stdin is not None
                    process.stdin.write(first_rgba)
                    for jpeg in frames[1:]:
                        rgba, frame_width, frame_height = jpeg_to_background_removed_rgba(
                            jpeg, tolerance=tolerance, softness=softness
                        )
                        if (frame_width, frame_height) != (width, height):
                            raise ValueError("prepared video frame dimensions changed")
                        process.stdin.write(rgba)
                    process.stdin.close()
                    return_code = process.wait()
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
                if return_code != 0:
                    errors.seek(0)
                    detail = errors.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg failed to encode prepared video: {detail}")
        if not temporary_output.is_file() or temporary_output.stat().st_size < 1024:
            raise RuntimeError("ffmpeg produced an empty prepared video")
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(frames),
        "duration_seconds": len(pcm16le) / 32000.0,
        "bytes": output_path.stat().st_size,
        "background_removed": True,
        "container": "webm",
        "video_codec": "vp9-alpha",
        "audio_codec": "opus",
    }


def write_manifest(cache_dir: Path, key: str, values: dict[str, object]) -> None:
    _video_path, manifest_path = artifact_paths(cache_dir, key)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"key": key, **values}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
