# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Test client for the CosyVoice FastAPI voice-cloning service."""

import argparse
import json
import time
import wave
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple

import requests


def wait_until_ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "service is not ready"
    announced = False
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=3)
            if response.ok and response.json().get("status") == "ok":
                if announced:
                    print("service is ready")
                return
            last_error = f"health endpoint returned HTTP {response.status_code}"
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        if not announced:
            print(f"waiting for CosyVoice service at {base_url} ...")
            announced = True
        time.sleep(1)
    raise RuntimeError(
        f"CosyVoice service was not ready after {timeout:.0f} seconds: {last_error}. "
        "Start it with `python server.py` in WSL and wait for the Uvicorn message."
    )


def raise_for_service_error(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(f"server returned HTTP {response.status_code}: {detail}")


def list_speakers(base_url: str) -> None:
    response = requests.get(f"{base_url}/v1/speakers", timeout=30)
    raise_for_service_error(response)
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


def build_request(
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Optional[Dict[str, Tuple[str, BinaryIO, str]]], Optional[BinaryIO]]:
    data = {
        "tts_text": args.tts_text,
        "stream": str(not args.no_stream).lower(),
        "speed": str(args.speed),
    }
    if args.speaker_id:
        data["speaker_id"] = args.speaker_id
        return data, None, None

    if not args.prompt_wav or not args.prompt_text:
        raise ValueError(
            "provide --speaker-id, or provide both --prompt-wav and --prompt-text"
        )
    prompt_path = Path(args.prompt_wav)
    prompt_file = prompt_path.open("rb")
    files = {
        "prompt_wav": (
            prompt_path.name,
            prompt_file,
            "application/octet-stream",
        )
    }
    data["prompt_text"] = args.prompt_text
    return data, files, prompt_file


def clone_voice(args: argparse.Namespace) -> None:
    data, files, prompt_file = build_request(args)
    url = f"{args.base_url}/v1/voice-clone"
    started_at = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    remainder = b""

    try:
        try:
            response_context = requests.post(
                url,
                data=data,
                files=files,
                stream=True,
                timeout=(30, args.read_timeout),
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"cannot connect to {url}; the CosyVoice service may have stopped"
            ) from exc
        with response_context as response:
            raise_for_service_error(response)
            sample_rate = int(response.headers["X-Audio-Sample-Rate"])
            channels = int(response.headers.get("X-Audio-Channels", "1"))
            if response.headers.get("X-Audio-Sample-Format") != "s16le":
                raise RuntimeError("server returned an unsupported audio format")

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as output_wav:
                output_wav.setnchannels(channels)
                output_wav.setsampwidth(2)
                output_wav.setframerate(sample_rate)
                # A small read size preserves first-chunk latency on the client.
                try:
                    for chunk in response.iter_content(chunk_size=4096):
                        if not chunk:
                            continue
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                        chunk = remainder + chunk
                        aligned_size = len(chunk) - (len(chunk) % 2)
                        output_wav.writeframesraw(chunk[:aligned_size])
                        total_bytes += aligned_size
                        remainder = chunk[aligned_size:]
                except requests.exceptions.ChunkedEncodingError as exc:
                    raise RuntimeError(
                        "server terminated the audio stream unexpectedly; "
                        "check the server traceback for the synthesis error"
                    ) from exc
                if remainder:
                    raise RuntimeError("server returned an incomplete PCM sample")
    finally:
        if prompt_file is not None:
            prompt_file.close()

    finished_at = time.perf_counter()
    if total_bytes == 0:
        raise RuntimeError("server returned no audio")
    audio_seconds = total_bytes / 2 / channels / sample_rate
    request_seconds = finished_at - started_at
    rtf = request_seconds / audio_seconds
    realtime_speed = audio_seconds / request_seconds
    first_chunk_seconds = (
        first_chunk_at - started_at if first_chunk_at is not None else 0.0
    )
    print(f"output          : {Path(args.output).resolve()}")
    print(f"sample rate     : {sample_rate} Hz")
    print(f"audio duration  : {audio_seconds:.3f} s")
    print(f"first chunk     : {first_chunk_seconds:.3f} s")
    print(f"request duration: {request_seconds:.3f} s")
    print(f"RTF             : {rtf:.4f}")
    print(f"real-time speed : {realtime_speed:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:50000")
    parser.add_argument("--list-speakers", action="store_true", default=False)
    parser.add_argument("--tts-text", default="你好，这是一段声音克隆测试。")
    parser.add_argument(
        "--speaker-id",
        default="",
        help="preset speaker returned by /v1/speakers",
    )
    parser.add_argument(
        "--prompt-wav",
        default="asset/zero_shot_prompt.wav",
        help="uploaded reference audio",
    )
    parser.add_argument(
        "--prompt-text",
        default="希望你以后能够做的比我还好呦。",
        help="exact transcript of --prompt-wav",
    )
    parser.add_argument("--output", default="voice_clone.wav")
    parser.add_argument("--no-stream", action="store_true", default=False)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    wait_until_ready(args.base_url, args.startup_timeout)
    if args.list_speakers:
        list_speakers(args.base_url)
        return
    clone_voice(args)


if __name__ == "__main__":
    main()
