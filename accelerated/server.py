"""FastAPI/WebSocket entry point for MuseTalk 1.5 streaming inference."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .runtime import MuseTalkRuntime, RuntimeConfig
from .streaming import PACKET_JPEG, PACKET_PCM16, encode_packet, jpeg_bytes


log = logging.getLogger(__name__)
_END = object()


def _next_or_end(iterator: Iterator[object]) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return _END


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    runtime = MuseTalkRuntime(config or RuntimeConfig.defaults())
    gpu_queue = asyncio.Lock()
    startup_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal startup_task
        startup_task = asyncio.create_task(asyncio.to_thread(runtime.initialize))
        yield
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()

    app = FastAPI(
        title="MuseTalk v1.5 accelerated streaming",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/readiness")
    async def readiness():
        status = runtime.status()
        return {"status": "ready" if runtime.ready else "loading", **status}

    @app.get("/v1/avatars")
    async def avatars():
        return runtime.status()["avatars"]

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        if not runtime.ready:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "not_ready",
                    "message": runtime.initialization_error or "models/avatar are still loading",
                }
            )
            await websocket.close(code=1013)
            return

        profile_id = runtime.config.default_avatar
        audio = bytearray()
        # Playback position belongs to this WebSocket session, never the shared
        # renderer. A fresh start therefore always begins at the source first frame.
        profile_positions: dict[str, int] = {}
        max_audio_bytes = 16000 * 2 * 120
        try:
            await websocket.send_json(
                {
                    "type": "ready",
                    "profile": profile_id,
                    "sample_rate": 16000,
                    "audio_format": "pcm_s16le_mono",
                    "fps": runtime.config.fps,
                    "backend": "torch",
                    "inference_dtype": runtime.engine.inference_dtype,
                    "packet_header": "<4sBBHIIQ",
                }
            )
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    if len(audio) + len(chunk) > max_audio_bytes:
                        raise ValueError("one utterance may not exceed 120 seconds")
                    audio.extend(chunk)
                    continue
                raw_text = message.get("text")
                if raw_text is None:
                    continue
                control = json.loads(raw_text)
                control_type = control.get("type")
                if control_type == "start":
                    requested = str(control.get("profile", runtime.config.default_avatar))
                    if requested not in runtime.avatar_specs:
                        raise ValueError(f"unknown avatar: {requested}")
                    profile_id = requested
                    if not bool(control.get("continue_from_previous", False)):
                        profile_positions[profile_id] = 0
                    audio.clear()
                    await websocket.send_json({"type": "started", "profile": profile_id})
                elif control_type == "cancel":
                    audio.clear()
                    await websocket.send_json({"type": "cancelled"})
                elif control_type == "commit":
                    if len(audio) < 2:
                        raise ValueError("commit requires PCM audio")
                    pcm = bytes(audio)
                    audio.clear()
                    await websocket.send_json({"type": "queued", "profile": profile_id})
                    async with gpu_queue:
                        profile = await asyncio.to_thread(runtime.get_profile, profile_id)
                        if runtime.renderer is None:
                            raise RuntimeError("streaming renderer is unavailable")
                        start_position = profile_positions.get(profile_id, 0)
                        iterator = runtime.renderer.render(
                            pcm,
                            profile,
                            start_position=start_position,
                        )
                        sequence = 0
                        rendered_frames = 0
                        await websocket.send_json({"type": "stream_start", "profile": profile_id})
                        pending: asyncio.Task[object] | None = asyncio.create_task(
                            asyncio.to_thread(_next_or_end, iterator)
                        )
                        try:
                            while True:
                                batch = await pending
                                pending = None
                                if batch is _END:
                                    break
                                rendered_frames += len(batch.frames)
                                # Render the next GPU batch while this batch is encoded and sent.
                                pending = asyncio.create_task(
                                    asyncio.to_thread(_next_or_end, iterator)
                                )
                                start_us = round(
                                    batch.start_frame / runtime.config.fps * 1_000_000
                                )
                                await websocket.send_bytes(
                                    encode_packet(PACKET_PCM16, sequence, start_us, batch.pcm16)
                                )
                                sequence += 1
                                for offset, frame in enumerate(batch.frames):
                                    frame_number = batch.start_frame + offset
                                    pts_us = round(
                                        frame_number / runtime.config.fps * 1_000_000
                                    )
                                    payload = jpeg_bytes(frame)
                                    await websocket.send_bytes(
                                        encode_packet(PACKET_JPEG, sequence, pts_us, payload)
                                    )
                                    sequence += 1
                        finally:
                            # A disconnect must not release the GPU lock while the prefetched
                            # batch is still using the shared inference session.
                            if pending is not None:
                                try:
                                    await asyncio.shield(pending)
                                except Exception:
                                    log.exception("prefetched render batch failed during cleanup")
                            iterator.close()
                            profile_positions[profile_id] = (
                                start_position + rendered_frames
                            ) % profile.cycle_length
                        await websocket.send_json(
                            {"type": "stream_end", "packets": sequence, "profile": profile_id}
                        )
        except WebSocketDisconnect:
            return
        except Exception as exc:
            log.exception("streaming session failed")
            try:
                await websocket.send_json(
                    {"type": "error", "code": "stream_failed", "message": str(exc)}
                )
            except Exception:
                pass

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("MUSETALK_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MUSETALK_PORT", "8083")))
    parser.add_argument("--fps", type=float, default=float(os.getenv("MUSETALK_FPS", "25")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("MUSETALK_BATCH_SIZE", "1")))
    parser.add_argument(
        "--max-frame-height",
        type=int,
        default=int(os.getenv("MUSETALK_MAX_FRAME_HEIGHT", "0")),
    )
    parser.add_argument("--device", default=os.getenv("MUSETALK_DEVICE", "cuda:0"))
    args = parser.parse_args()

    import uvicorn

    config = RuntimeConfig.defaults()
    config = RuntimeConfig(
        **{
            **config.__dict__,
            "fps": args.fps,
            "batch_size": args.batch_size,
            "max_frame_height": args.max_frame_height,
            "device": args.device,
        }
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
