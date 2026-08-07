"""Command-line test client for the MuseTalk streaming WebSocket service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import wave

import websockets

from .runtime import PUBLIC_AVATAR_FILES
from .streaming import PACKET_JPEG, PACKET_PCM16, decode_packet


log = logging.getLogger("musetalk-stream-client")


def load_pcm16(audio_path: Path) -> tuple[bytes, float]:
    """Use FFmpeg to decode common audio formats into the server wire format."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode the input audio")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"audio could not be decoded: {detail or audio_path}")
    pcm = completed.stdout
    if not pcm or len(pcm) % 2:
        raise ValueError(f"audio contains no complete PCM samples: {audio_path}")
    return pcm, len(pcm) / (16000.0 * 2.0)


def write_wav(path: Path, pcm16le: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm16le)


def mux_result(frames_dir: Path, pcm16le: bytes, output_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the test MP4")
    audio_path = frames_dir.parent / "received.wav"
    write_wav(audio_path, pcm16le)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%08d.jpg"),
        "-i",
        str(audio_path),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")


async def run_test(
    url: str,
    audio_path: Path,
    output_path: Path,
    profile: str,
    chunk_bytes: int,
    timeout: float,
) -> None:
    pcm, audio_duration = load_pcm16(audio_path)
    log.info("input=%s duration=%.3fs bytes=%d", audio_path, audio_duration, len(pcm))

    started_at = time.perf_counter()
    first_packet_at: float | None = None
    frame_count = 0
    audio_packets = 0
    expected_sequence = 0
    last_video_pts = -1
    last_audio_pts = -1
    received_audio = bytearray()

    with tempfile.TemporaryDirectory(prefix="musetalk-stream-client-") as temporary:
        frames_dir = Path(temporary) / "frames"
        frames_dir.mkdir()
        async with websockets.connect(
            url,
            open_timeout=timeout,
            close_timeout=5,
            ping_timeout=None,
            max_size=None,
        ) as websocket:
            initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=timeout))
            if initial.get("type") != "ready":
                raise RuntimeError(initial.get("message", f"server is not ready: {initial}"))
            fps = float(initial.get("fps", 25.0))
            log.info("connected profile=%s server_fps=%.2f", profile, fps)

            await websocket.send(json.dumps({"type": "start", "profile": profile}))
            response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=timeout))
            if response.get("type") != "started":
                raise RuntimeError(f"start was rejected: {response}")
            upload_started = time.perf_counter()
            for offset in range(0, len(pcm), chunk_bytes):
                await websocket.send(pcm[offset : offset + chunk_bytes])
            await websocket.send(json.dumps({"type": "commit"}))
            upload_elapsed = time.perf_counter() - upload_started
            log.info("audio uploaded in %.3fs; waiting for generated packets", upload_elapsed)

            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                if isinstance(message, str):
                    control = json.loads(message)
                    message_type = control.get("type")
                    if message_type == "error":
                        raise RuntimeError(control.get("message", "unknown server error"))
                    log.info("server event: %s", message_type)
                    if message_type == "stream_end":
                        break
                    continue

                packet_type, sequence, pts_us, payload = decode_packet(message)
                if sequence != expected_sequence:
                    raise RuntimeError(
                        f"packet sequence jumped from {expected_sequence} to {sequence}"
                    )
                expected_sequence += 1
                if first_packet_at is None:
                    first_packet_at = time.perf_counter()
                if packet_type == PACKET_JPEG:
                    if pts_us <= last_video_pts:
                        raise RuntimeError("video timestamps are not strictly increasing")
                    last_video_pts = pts_us
                    (frames_dir / f"{frame_count:08d}.jpg").write_bytes(payload)
                    frame_count += 1
                    if frame_count == 1 or frame_count % round(fps) == 0:
                        log.info("received %d video frames (pts %.3fs)", frame_count, pts_us / 1e6)
                elif packet_type == PACKET_PCM16:
                    if pts_us <= last_audio_pts:
                        raise RuntimeError("audio timestamps are not strictly increasing")
                    last_audio_pts = pts_us
                    if len(payload) % 2:
                        raise RuntimeError("received an incomplete PCM sample")
                    received_audio.extend(payload)
                    audio_packets += 1
                else:
                    raise RuntimeError(f"unknown binary packet type: {packet_type}")

        if frame_count == 0 or not received_audio:
            raise RuntimeError("stream ended without complete audio/video output")
        mux_result(frames_dir, bytes(received_audio), output_path, fps)

    finished_at = time.perf_counter()
    first_latency = (first_packet_at or finished_at) - started_at
    elapsed = finished_at - started_at
    output_duration = frame_count / fps
    log.info("test passed")
    log.info("first media latency: %.3fs", first_latency)
    log.info("frames=%d audio_packets=%d output_duration=%.3fs", frame_count, audio_packets, output_duration)
    log.info("total=%.3fs realtime_factor=%.3f", elapsed, elapsed / max(output_duration, 1e-6))
    log.info("result saved to %s", output_path.resolve())


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8083/v1/stream")
    parser.add_argument(
        "--audio",
        type=Path,
        default=project_root / "data" / "input" / "audio" / "demo2_audio.wav",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "output" / "accelerated_stream_test.mp4",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PUBLIC_AVATAR_FILES),
        default="chinese",
    )
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="maximum seconds to wait for each server message",
    )
    args = parser.parse_args()
    if args.chunk_bytes < 2 or args.chunk_bytes % 2:
        parser.error("--chunk-bytes must be a positive even number")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    try:
        asyncio.run(
            run_test(
                args.url,
                args.audio.resolve(),
                args.output.resolve(),
                args.profile,
                args.chunk_bytes,
                args.timeout,
            )
        )
    except KeyboardInterrupt:
        log.warning("test interrupted")
        raise SystemExit(130)
    except Exception as exc:
        log.error("test failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
