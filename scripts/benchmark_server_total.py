"""Benchmark the live server_total speech pipeline and report chunk RTF."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import time

import websockets


DEFAULT_TEXT = (
    "Welcome to the real time digital human demonstration. "
    "This longer message verifies that audio and video continue without running "
    "out of buffered media while speech synthesis and lip rendering overlap. "
    "The benchmark records every rendering chunk and reports its real time factor."
)


async def benchmark(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    first_media_at: float | None = None
    audio_packets = 0
    video_packets = 0
    metrics: list[dict[str, object]] = []
    latest_audio_end = 0.0
    playback_started_at: float | None = None
    minimum_buffer_seconds = float("inf")
    simulated_underflows = 0

    async with websockets.connect(
        args.url,
        open_timeout=args.timeout,
        max_size=None,
    ) as websocket:
        ready = json.loads(await asyncio.wait_for(websocket.recv(), args.timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"server is not ready: {ready}")
        await websocket.send(
            json.dumps(
                {
                    "type": "speak",
                    "text": args.text,
                    "profile": args.profile,
                    "language": args.language,
                    "speed": 1.0,
                }
            )
        )

        while True:
            message = await asyncio.wait_for(websocket.recv(), args.timeout)
            if isinstance(message, bytes):
                arrived_at = time.perf_counter()
                first_media_at = first_media_at or arrived_at
                if len(message) >= 6 and message[5] == 1:
                    video_packets += 1
                elif len(message) >= 6 and message[5] == 2:
                    audio_packets += 1
                    _magic, _version, _type, _flags, _sequence, size, pts_us = struct.unpack_from(
                        "<4sBBHIIQ", message
                    )
                    packet_end = pts_us / 1_000_000 + size / 32_000
                    if playback_started_at is None and packet_end >= float(
                        ready.get("playback_buffer_seconds", 1.5)
                    ):
                        playback_started_at = arrived_at
                    elif playback_started_at is not None:
                        playback_position = arrived_at - playback_started_at
                        depth_before_arrival = latest_audio_end - playback_position
                        minimum_buffer_seconds = min(
                            minimum_buffer_seconds, depth_before_arrival
                        )
                        if depth_before_arrival < 0:
                            simulated_underflows += 1
                    latest_audio_end = max(latest_audio_end, packet_end)
                continue

            event = json.loads(message)
            if event.get("type") == "chunk_metrics":
                metrics.append(event)
            elif event.get("type") == "error":
                raise RuntimeError(str(event.get("message", "pipeline failed")))
            elif event.get("type") == "conversation_end":
                if event.get("failed"):
                    raise RuntimeError("pipeline reported a failed conversation")
                break

    elapsed = time.perf_counter() - started
    rtfs = [float(item["realtime_factor"]) for item in metrics]
    durations = [float(item["audio_duration_seconds"]) for item in metrics]
    print(f"first_media_seconds={None if first_media_at is None else round(first_media_at - started, 3)}")
    print(f"total_seconds={elapsed:.3f}")
    print(f"audio_seconds={sum(durations):.3f}")
    print(f"chunks={len(metrics)} audio_packets={audio_packets} video_packets={video_packets}")
    print(
        "simulated_underflows={} minimum_buffer_seconds={}".format(
            simulated_underflows,
            None
            if minimum_buffer_seconds == float("inf")
            else round(minimum_buffer_seconds, 3),
        )
    )
    if rtfs:
        ordered = sorted(rtfs)
        p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered) - 1)))
        print(
            "rtf_mean={:.3f} rtf_p95={:.3f} rtf_max={:.3f}".format(
                statistics.fmean(rtfs), ordered[p95_index], max(rtfs)
            )
        )
        for item in metrics:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8080/v1/conversation")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--profile", default="chinese")
    parser.add_argument("--language", choices=("ZH", "EN"), default="EN")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
