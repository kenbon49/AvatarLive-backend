"""FastAPI/WebSocket entry point for MuseTalk 1.5 streaming inference."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import secrets
from typing import Iterator

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .runtime import MuseTalkRuntime, RuntimeConfig
from .streaming import PACKET_JPEG, PACKET_PCM16, encode_packet, jpeg_bytes


log = logging.getLogger(__name__)
_END = object()


class PrepareAvatarRequest(BaseModel):
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")


def _next_or_end(iterator: Iterator[object]) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return _END


def _advance_profile_phase(
    start_phase: float,
    pcm_byte_count: int,
    fps: float,
    cycle_length: int,
) -> float:
    audio_frames = pcm_byte_count / 2 / 16000 * fps
    return (start_phase + audio_frames) % cycle_length


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    runtime = MuseTalkRuntime(config or RuntimeConfig.defaults())
    gpu_queue = asyncio.Lock()
    startup_task: asyncio.Task[None] | None = None
    registry_key = os.getenv("AVATAR_REGISTRY_KEY", "")

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

    @app.post("/v1/avatars/prepare")
    async def prepare_avatar(
        request: PrepareAvatarRequest,
        x_avatar_registry_key: str | None = Header(default=None),
    ):
        if not registry_key or not x_avatar_registry_key or not secrets.compare_digest(
            registry_key, x_avatar_registry_key
        ):
            raise HTTPException(status_code=401, detail="invalid avatar registry key")
        if not runtime.ready:
            raise HTTPException(status_code=503, detail="MuseTalk runtime is still loading")
        try:
            async with gpu_queue:
                profile = await asyncio.to_thread(
                    runtime.prepare_custom_profile, request.profile_id
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("custom avatar preparation failed for %s", request.profile_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "status": "ready",
            "profile_id": request.profile_id,
            "frames": len(profile.frames),
            "fps": profile.fps,
        }

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
        profile_phases: dict[str, float] = {}
        max_audio_bytes = 16000 * 2 * 120
        try:
            await websocket.send_json(
                {
                    "type": "ready",
                    "profile": profile_id,
                    "sample_rate": 16000,
                    "audio_format": "pcm_s16le_mono",
                    "fps": runtime.config.fps,
                    "jpeg_quality": runtime.config.jpeg_quality,
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
                    if not runtime.is_profile_streamable(requested):
                        raise ValueError(f"unknown or unpublished avatar: {requested}")
                    profile_id = requested
                    if not bool(control.get("continue_from_previous", False)):
                        start_position = int(control.get("start_position", 0))
                        if not 0 <= start_position <= 10_000_000:
                            raise ValueError("start_position is outside the supported range")
                        profile_phases[profile_id] = float(start_position)
                    audio.clear()
                    await websocket.send_json(
                        {
                            "type": "started",
                            "profile": profile_id,
                            "start_position": math.floor(
                                profile_phases.get(profile_id, 0.0)
                            ),
                        }
                    )
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
                        start_phase = profile_phases.get(profile_id, 0.0)
                        start_position = math.floor(start_phase)
                        iterator = runtime.renderer.render(
                            pcm,
                            profile,
                            start_position=start_position,
                        )
                        sequence = 0
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
                                    payload = jpeg_bytes(frame, runtime.config.jpeg_quality)
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
                            profile_phases[profile_id] = _advance_profile_phase(
                                start_phase,
                                len(pcm),
                                runtime.config.fps,
                                profile.cycle_length,
                            )
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("MUSETALK_BATCH_SIZE", "12")),
    )
    parser.add_argument(
        "--max-frame-height",
        type=int,
        default=int(os.getenv("MUSETALK_MAX_FRAME_HEIGHT", "0")),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=int(os.getenv("MUSETALK_JPEG_QUALITY", "92")),
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
            "jpeg_quality": args.jpeg_quality,
            "device": args.device,
        }
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
