"""End-to-end streaming question-to-avatar API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
import math
import os
import time
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from scipy.signal import resample_poly
import websockets

from llm_inference import LLMInferenceError, LiteLLMClient

from .protocol import remap_media_packet
from .segmenter import PunctuationSegmenter, TextUnit


log = logging.getLogger("server-total")
OPENVOICE_URL = os.getenv("OPENVOICE_URL", "http://localhost:8084").rstrip("/")
DEFAULT_VOICE_ID = os.getenv("OPENVOICE_DEFAULT_VOICE", "default")
MUSETALK_WS_URL = os.getenv("MUSETALK_WS_URL", "ws://localhost:8083/v1/stream")
MUSETALK_BACKEND = "torch"
MUSETALK_INFERENCE_DTYPE = "float32"
REQUEST_TIMEOUT = float(os.getenv("PIPELINE_REQUEST_TIMEOUT", "300"))
PCM_CHUNK_BYTES = 64 * 1024
SENTENCE_QUEUE_SIZE = int(os.getenv("PIPELINE_TEXT_QUEUE_SIZE", "4"))
AUDIO_QUEUE_SIZE = int(os.getenv("PIPELINE_AUDIO_QUEUE_SIZE", "12"))
TTS_STREAM_CHUNK_SECONDS = float(os.getenv("PIPELINE_TTS_CHUNK_SECONDS", "0.5"))
TTS_STREAM_STEADY_CHUNK_SECONDS = float(
    os.getenv("PIPELINE_TTS_STEADY_CHUNK_SECONDS", "1.0")
)
TTS_STREAM_MIN_TAIL_SECONDS = float(
    os.getenv("PIPELINE_TTS_MIN_TAIL_SECONDS", "0.25")
)
PLAYBACK_BUFFER_SECONDS = float(os.getenv("PIPELINE_PLAYBACK_BUFFER_SECONDS", "1.5"))
FIRST_UNIT_MIN_CHARS = int(os.getenv("PIPELINE_FIRST_UNIT_MIN_CHARS", "1"))
TARGET_UNIT_CHARS = int(os.getenv("PIPELINE_TARGET_UNIT_CHARS", "1"))
COALESCE_HARD_DELIMITERS = os.getenv(
    "PIPELINE_COALESCE_HARD_DELIMITERS", "0"
).lower() not in {"0", "false", "no"}
_END = object()
AvatarProfile = Literal[
    "chinese",
    "business_male_1",
    "chen_yu",
]
DEFAULT_AVATAR_PROFILE: AvatarProfile = "chinese"
AVATAR_CATALOG = (
    {"id": "chinese", "name": "Chinese", "default": True},
    {"id": "business_male_1", "name": "商务男", "default": False},
    {"id": "chen_yu", "name": "陈屿", "default": False},
)


class AskRequest(BaseModel):
    type: Literal["ask"]
    question: str = Field(min_length=1, max_length=2000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile: AvatarProfile = DEFAULT_AVATAR_PROFILE
    language: Literal["ZH", "EN"] = "ZH"
    voice_id: str | None = Field(default=None, min_length=1, max_length=64)
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)


class CancelRequest(BaseModel):
    type: Literal["cancel"]
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class SpeakRequest(BaseModel):
    type: Literal["speak"]
    text: str = Field(min_length=1, max_length=2000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile: AvatarProfile = DEFAULT_AVATAR_PROFILE
    language: Literal["ZH", "EN"] = "ZH"
    voice_id: str | None = Field(default=None, min_length=1, max_length=64)
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)


@dataclass(frozen=True)
class AudioUnit:
    unit: TextUnit
    pcm: bytes
    duration: float
    chunk_index: int
    produced_at: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class AudioUnitEnd:
    unit: TextUnit
    duration: float
    chunks: int


@dataclass
class MediaTimeline:
    """PTS and packet sequence shared by all media units in one response."""

    pts_offset_us: int = 0
    sequence: int = 0


class FrontendSender:
    """Serialize JSON and binary writes from concurrent pipeline stages."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._lock = asyncio.Lock()

    async def send_json(self, value: dict[str, Any]) -> None:
        async with self._lock:
            await self.websocket.send_json(value)

    async def send_bytes(self, value: bytes) -> None:
        async with self._lock:
            await self.websocket.send_bytes(value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT))
    app.state.llm = LiteLLMClient()
    yield
    await app.state.http.aclose()


app = FastAPI(
    title="MuseTalk total streaming API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, object]:
    llm = app.state.llm.config
    return {
        "status": "ok",
        "service": "server_total",
        "llm": {
            "base_url": llm.base_url,
            "model": llm.model,
            "api_key_configured": bool(llm.api_key),
            "streaming": True,
        },
        "tts_service": "openvoice",
        "openvoice_url": OPENVOICE_URL,
        "default_voice_id": DEFAULT_VOICE_ID,
        "musetalk_ws_url": MUSETALK_WS_URL,
        "musetalk_backend": MUSETALK_BACKEND,
        "musetalk_inference_dtype": MUSETALK_INFERENCE_DTYPE,
        "default_avatar": DEFAULT_AVATAR_PROFILE,
        "avatars": AVATAR_CATALOG,
        "playback_buffer_seconds": PLAYBACK_BUFFER_SECONDS,
    }


@app.get("/v1/avatars")
async def avatars() -> dict[str, object]:
    return {
        "default": DEFAULT_AVATAR_PROFILE,
        "avatars": AVATAR_CATALOG,
    }


@app.get("/v1/voices")
async def voices() -> dict[str, object]:
    try:
        response = await app.state.http.get(f"{OPENVOICE_URL}/v1/speakers")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenVoice is unavailable: {exc}"
        ) from exc


@app.post("/v1/voices/clone", status_code=201)
async def create_cloned_voice(
    name: str = Form(..., min_length=1, max_length=80),
    audio: UploadFile = File(...),
) -> dict[str, object]:
    content = await audio.read(50 * 1024 * 1024 + 1)
    await audio.close()
    if not content:
        raise HTTPException(status_code=400, detail="audio is empty")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio exceeds the 50 MB limit")
    response = await app.state.http.post(
        f"{OPENVOICE_URL}/v1/voices/clone",
        data={"name": name},
        files={
            "audio": (
                audio.filename or "reference.wav",
                content,
                audio.content_type or "application/octet-stream",
            )
        },
    )
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


async def stream_speech_chunks(
    app_: FastAPI,
    request: AskRequest | SpeakRequest,
    text: str,
    *,
    first_chunk_seconds: float | None = None,
) -> AsyncIterator[tuple[bytes, float]]:
    voice_id = request.voice_id or request.speaker or DEFAULT_VOICE_ID
    async with app_.state.http.stream(
        "POST",
        f"{OPENVOICE_URL}/v1/voice-clone",
        data={
            "tts_text": text,
            "speaker_id": voice_id,
            "language": request.language.lower(),
            "stream": str(request.speed == 1.0).lower(),
            "speed": request.speed,
            "output_sample_rate": 16000,
        },
    ) as response:
        if response.is_error:
            await response.aread()
        response.raise_for_status()
        if response.headers.get("X-Audio-Sample-Format") != "s16le":
            raise RuntimeError("OpenVoice returned an unsupported audio format")
        source_rate = int(response.headers.get("X-Audio-Sample-Rate", "0"))
        if source_rate <= 0:
            raise RuntimeError("OpenVoice did not report its sample rate")
        first_chunk_bytes = max(
            2,
            round(
                source_rate
                * 2
                * (
                    TTS_STREAM_CHUNK_SECONDS
                    if first_chunk_seconds is None
                    else first_chunk_seconds
                )
            ),
        )
        steady_chunk_bytes = max(
            2, round(source_rate * 2 * TTS_STREAM_STEADY_CHUNK_SECONDS)
        )
        first_chunk_bytes -= first_chunk_bytes % 2
        steady_chunk_bytes -= steady_chunk_bytes % 2
        min_tail_bytes = max(2, round(source_rate * 2 * TTS_STREAM_MIN_TAIL_SECONDS))
        min_tail_bytes -= min_tail_bytes % 2
        pending = bytearray()
        target_bytes = first_chunk_bytes
        emitted_first = False

        def emit_chunk(source_pcm: bytes) -> tuple[bytes, float]:
            pcm = source_pcm
            if source_rate != 16000:
                divisor = math.gcd(source_rate, 16000)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                samples = resample_poly(
                    samples,
                    16000 // divisor,
                    source_rate // divisor,
                )
                pcm = (
                    np.clip(np.rint(samples), -32768, 32767)
                    .astype("<i2")
                    .tobytes()
                )
            return pcm, len(pcm) / 32000.0

        async for incoming in response.aiter_bytes():
            if incoming:
                pending.extend(incoming)
            reserve_bytes = 0 if not emitted_first else min_tail_bytes
            while len(pending) >= target_bytes + reserve_bytes:
                source_pcm = bytes(pending[:target_bytes])
                del pending[:target_bytes]
                yield emit_chunk(source_pcm)
                emitted_first = True
                target_bytes = steady_chunk_bytes
                reserve_bytes = min_tail_bytes

        if len(pending) % 2:
            raise RuntimeError("OpenVoice returned invalid PCM audio")
        if pending:
            yield emit_chunk(bytes(pending))


async def synthesize_speech(
    app_: FastAPI, request: AskRequest | SpeakRequest, text: str
) -> tuple[bytes, float]:
    chunks: list[bytes] = []
    duration = 0.0
    async for pcm, chunk_duration in stream_speech_chunks(app_, request, text):
        chunks.append(pcm)
        duration += chunk_duration
    if not chunks:
        raise RuntimeError("OpenVoice returned empty PCM audio")
    return b"".join(chunks), duration


def _next_or_end(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _END


async def _produce_text_units(
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
    queue: asyncio.Queue[TextUnit | object],
    *,
    kind: Literal["ask", "speak"] = "ask",
) -> str:
    segmenter = PunctuationSegmenter(
        first_unit_min_chars=FIRST_UNIT_MIN_CHARS,
        target_unit_chars=TARGET_UNIT_CHARS,
        coalesce_hard_delimiters=COALESCE_HARD_DELIMITERS,
    )
    answer_parts: list[str] = []

    if kind == "speak":
        await sender.send_json({"type": "speak_start", "request_id": request_id})
        answer = request.text.strip()
        for unit in segmenter.feed(answer):
            await queue.put(unit)
        for unit in segmenter.finish():
            await queue.put(unit)
        await sender.send_json(
            {"type": "speak_result", "request_id": request_id, "text": answer}
        )
        await queue.put(_END)
        return answer

    iterator = app.state.llm.stream_answer_text(request.question)
    while True:
        delta = await asyncio.to_thread(_next_or_end, iterator)
        if delta is _END:
            break
        answer_parts.append(delta)
        await sender.send_json(
            {"type": "llm_delta", "request_id": request_id, "delta": delta}
        )
        for unit in segmenter.feed(delta):
            await queue.put(unit)
    for unit in segmenter.finish():
        await queue.put(unit)
    answer = "".join(answer_parts).strip()
    await sender.send_json(
        {"type": "llm_result", "request_id": request_id, "answer": answer}
    )
    await queue.put(_END)
    return answer


async def _synthesize_units(
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
    text_queue: asyncio.Queue[TextUnit | object],
    audio_queue: asyncio.Queue[AudioUnit | AudioUnitEnd | object],
) -> int:
    count = 0
    while True:
        item = await text_queue.get()
        if item is _END:
            break
        if not isinstance(item, TextUnit):
            raise TypeError("text queue received an invalid item")
        started = time.perf_counter()
        await sender.send_json(
            {
                "type": "tts_start",
                "request_id": request_id,
                "seq": item.seq,
                "text": item.text,
            }
        )
        duration = 0.0
        chunks = 0
        async for pcm, chunk_duration in stream_speech_chunks(
            app,
            request,
            item.text,
            first_chunk_seconds=(
                TTS_STREAM_CHUNK_SECONDS
                if count == 0
                else TTS_STREAM_STEADY_CHUNK_SECONDS
            ),
        ):
            await audio_queue.put(
                AudioUnit(item, pcm, chunk_duration, chunk_index=chunks)
            )
            duration += chunk_duration
            chunks += 1
        if chunks == 0:
            raise RuntimeError("OpenVoice returned empty PCM audio")
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        await sender.send_json(
            {
                "type": "tts_result",
                "request_id": request_id,
                "seq": item.seq,
                "text": item.text,
                "duration_seconds": round(duration, 3),
                "elapsed_ms": elapsed_ms,
                "chunks": chunks,
            }
        )
        await audio_queue.put(AudioUnitEnd(item, duration, chunks))
        count += 1
    await audio_queue.put(_END)
    return count


async def _handshake_upstream(
    upstream: Any,
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
) -> float:
    """Wait for MuseTalk readiness, validate the backend, and report to the frontend."""
    initial_raw = await asyncio.wait_for(upstream.recv(), timeout=REQUEST_TIMEOUT)
    if not isinstance(initial_raw, str):
        raise RuntimeError("MuseTalk did not send its readiness event")
    initial = json.loads(initial_raw)
    if initial.get("type") != "ready":
        raise RuntimeError(initial.get("message", "MuseTalk is not ready"))
    upstream_backend = str(initial.get("backend", "")).lower()
    if upstream_backend != MUSETALK_BACKEND:
        raise RuntimeError(
            f"MuseTalk backend mismatch: expected {MUSETALK_BACKEND}, "
            f"got {upstream_backend or 'unknown'}"
        )
    upstream_dtype = str(initial.get("inference_dtype", "")).lower()
    if upstream_dtype != MUSETALK_INFERENCE_DTYPE:
        raise RuntimeError(
            f"MuseTalk dtype mismatch: expected {MUSETALK_INFERENCE_DTYPE}, "
            f"got {upstream_dtype or 'unknown'}"
        )
    fps = float(initial.get("fps", 25))
    await sender.send_json(
        {
            "type": "musetalk_ready",
            "request_id": request_id,
            "profile": request.profile,
            "fps": fps,
            "sample_rate": 16000,
            "backend": upstream_backend,
            "inference_dtype": upstream_dtype,
        }
    )
    return fps


async def _render_units(
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
    audio_queue: asyncio.Queue[AudioUnit | AudioUnitEnd | object],
    *,
    timeline: MediaTimeline | None = None,
) -> int:
    timeline = timeline if timeline is not None else MediaTimeline()
    rendered = 0
    async with websockets.connect(
        MUSETALK_WS_URL,
        open_timeout=REQUEST_TIMEOUT,
        close_timeout=5,
        ping_timeout=None,
        max_size=None,
    ) as upstream:
        fps = await _handshake_upstream(
            upstream, sender, request, request_id
        )
        active_unit: TextUnit | None = None
        active_packet_count = 0

        while True:
            item = await audio_queue.get()
            if item is _END:
                break
            if isinstance(item, AudioUnitEnd):
                if active_unit is None or active_unit.seq != item.unit.seq:
                    raise RuntimeError("audio unit ended without an active segment")
                await sender.send_json(
                    {
                        "type": "segment_end",
                        "request_id": request_id,
                        "seq": item.unit.seq,
                        "text": item.unit.text,
                        "duration_seconds": round(item.duration, 3),
                        "chunks": item.chunks,
                        "packets": active_packet_count,
                    }
                )
                active_unit = None
                active_packet_count = 0
                rendered += 1
                continue
            if not isinstance(item, AudioUnit):
                raise TypeError("audio queue received an invalid item")
            unit = item.unit
            render_started_at = time.perf_counter()
            queue_wait_ms = round((render_started_at - item.produced_at) * 1000)
            if item.chunk_index == 0:
                if active_unit is not None:
                    raise RuntimeError("new audio segment started before the previous end")
                active_unit = unit
                await sender.send_json(
                    {
                        "type": "segment_start",
                        "request_id": request_id,
                        "seq": unit.seq,
                        "text": unit.text,
                        "delimiter": unit.delimiter,
                        "pts_us": timeline.pts_offset_us,
                    }
                )
            elif active_unit is None or active_unit.seq != unit.seq:
                raise RuntimeError("audio chunk does not match the active segment")
            await upstream.send(
                json.dumps(
                    {
                        "type": "start",
                        "profile": request.profile,
                        "continue_from_previous": timeline.sequence > 0,
                    }
                )
            )
            for offset in range(0, len(item.pcm), PCM_CHUNK_BYTES):
                await upstream.send(item.pcm[offset : offset + PCM_CHUNK_BYTES])
            await upstream.send(json.dumps({"type": "commit"}))
            committed_at = time.perf_counter()

            packet_count = 0
            text_unit_sent = item.chunk_index > 0
            stream_started_at: float | None = None
            first_media_at: float | None = None
            while True:
                message = await asyncio.wait_for(
                    upstream.recv(), timeout=REQUEST_TIMEOUT
                )
                if isinstance(message, bytes):
                    if first_media_at is None:
                        first_media_at = time.perf_counter()
                    if not text_unit_sent:
                        await sender.send_json(
                            {
                                "type": "text_unit",
                                "request_id": request_id,
                                "seq": unit.seq,
                                "text": unit.text,
                                "delimiter": unit.delimiter,
                                "pts_us": timeline.pts_offset_us,
                            }
                        )
                        text_unit_sent = True
                    outgoing, _packet_type, _pts_us = remap_media_packet(
                        message,
                        sequence=timeline.sequence,
                        pts_offset_us=timeline.pts_offset_us,
                    )
                    await sender.send_bytes(outgoing)
                    timeline.sequence += 1
                    packet_count += 1
                    continue
                control = json.loads(message)
                if control.get("type") == "error":
                    raise RuntimeError(control.get("message", "MuseTalk stream failed"))
                if control.get("type") == "stream_start" and not text_unit_sent:
                    await sender.send_json(
                        {
                            "type": "text_unit",
                            "request_id": request_id,
                            "seq": unit.seq,
                            "text": unit.text,
                            "delimiter": unit.delimiter,
                            "pts_us": timeline.pts_offset_us,
                        }
                    )
                    text_unit_sent = True
                if control.get("type") == "stream_start":
                    stream_started_at = time.perf_counter()
                control.update(
                    {
                        "request_id": request_id,
                        "segment_seq": unit.seq,
                        "chunk_index": item.chunk_index,
                    }
                )
                await sender.send_json(control)
                if control.get("type") == "stream_end":
                    break

            render_finished_at = time.perf_counter()
            processing_seconds = render_finished_at - render_started_at
            realtime_factor = processing_seconds / max(item.duration, 1e-6)
            metrics = {
                "type": "chunk_metrics",
                "request_id": request_id,
                "segment_seq": unit.seq,
                "chunk_index": item.chunk_index,
                "audio_duration_seconds": round(item.duration, 3),
                "queue_wait_ms": queue_wait_ms,
                "commit_to_stream_start_ms": (
                    round((stream_started_at - committed_at) * 1000)
                    if stream_started_at is not None
                    else None
                ),
                "first_media_ms": (
                    round((first_media_at - render_started_at) * 1000)
                    if first_media_at is not None
                    else None
                ),
                "processing_ms": round(processing_seconds * 1000),
                "realtime_factor": round(realtime_factor, 3),
                "packets": packet_count,
            }
            await sender.send_json(metrics)
            log.info(
                "MuseTalk chunk request=%s segment=%d chunk=%d duration=%.3fs "
                "processing=%.3fs rtf=%.3f queue_wait=%dms",
                request_id,
                unit.seq,
                item.chunk_index,
                item.duration,
                processing_seconds,
                realtime_factor,
                queue_wait_ms,
            )
            active_packet_count += packet_count
            rendered_frames = math.floor(len(item.pcm) / 2 / 16000 * fps)
            timeline.pts_offset_us += round(rendered_frames / fps * 1_000_000)
    return rendered


async def run_pipeline(
    websocket: WebSocket | FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str | None = None,
) -> None:
    sender = (
        websocket
        if isinstance(websocket, FrontendSender)
        else FrontendSender(websocket)
    )
    resolved_request_id = request_id or request.request_id or uuid4().hex
    started_at = time.perf_counter()
    kind = request.type
    text_queue: asyncio.Queue[TextUnit | object] = asyncio.Queue(
        maxsize=SENTENCE_QUEUE_SIZE
    )
    audio_queue: asyncio.Queue[AudioUnit | AudioUnitEnd | object] = asyncio.Queue(
        maxsize=AUDIO_QUEUE_SIZE
    )
    await sender.send_json(
        {
            "type": "conversation_start",
            "request_id": resolved_request_id,
            "profile": request.profile,
            "kind": kind,
        }
    )
    if kind == "ask":
        await sender.send_json({"type": "llm_start", "request_id": resolved_request_id})

    tasks = [
        asyncio.create_task(
            _produce_text_units(
                sender, request, resolved_request_id, text_queue, kind=kind
            )
        ),
        asyncio.create_task(
            _synthesize_units(
                sender,
                request,
                resolved_request_id,
                text_queue,
                audio_queue,
            )
        ),
        asyncio.create_task(
            _render_units(
                sender,
                request,
                resolved_request_id,
                audio_queue,
            )
        ),
    ]
    try:
        answer, synthesized, rendered = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    await sender.send_json(
        {
            "type": "conversation_end",
            "request_id": resolved_request_id,
            "kind": kind,
            "answer": answer,
            "units": synthesized,
            "rendered_units": rendered,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
            "cancelled": False,
        }
    )


async def _run_and_report(
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
) -> None:
    try:
        await run_pipeline(sender, request, request_id)
    except asyncio.CancelledError:
        raise
    except LLMInferenceError as exc:
        log.error("LLM request failed: %s", exc)
        await sender.send_json(
            {
                "type": "error",
                "request_id": request_id,
                "stage": "llm",
                "message": str(exc),
            }
        )
        await sender.send_json(
            {
                "type": "conversation_end",
                "request_id": request_id,
                "failed": True,
                "cancelled": False,
            }
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        log.exception("OpenVoice request failed")
        await sender.send_json(
            {
                "type": "error",
                "request_id": request_id,
                "stage": "tts",
                "message": detail or str(exc),
            }
        )
        await sender.send_json(
            {
                "type": "conversation_end",
                "request_id": request_id,
                "failed": True,
                "cancelled": False,
            }
        )
    except Exception as exc:
        log.exception("conversation pipeline failed")
        await sender.send_json(
            {
                "type": "error",
                "request_id": request_id,
                "stage": "pipeline",
                "message": str(exc),
            }
        )
        await sender.send_json(
            {
                "type": "conversation_end",
                "request_id": request_id,
                "failed": True,
                "cancelled": False,
            }
        )


@app.websocket("/v1/conversation")
async def conversation(websocket: WebSocket) -> None:
    await websocket.accept()
    sender = FrontendSender(websocket)
    await sender.send_json(
        {
            "type": "ready",
            "protocol": "MSTK/2",
            "packet_header": "<4sBBHIIQ",
            "sample_rate": 16000,
            "playback_buffer_seconds": PLAYBACK_BUFFER_SECONDS,
            "default_profile": DEFAULT_AVATAR_PROFILE,
            "avatars": AVATAR_CATALOG,
        }
    )
    active_task: asyncio.Task[None] | None = None
    active_request_id: str | None = None
    try:
        while True:
            payload = await websocket.receive_json()
            if active_task is not None and active_task.done():
                await asyncio.gather(active_task, return_exceptions=True)
                active_task = None
                active_request_id = None

            message_type = payload.get("type")

            if message_type == "cancel":
                try:
                    cancel = CancelRequest.model_validate(payload)
                except ValidationError as exc:
                    await sender.send_json(
                        {"type": "error", "stage": "request", "message": str(exc)}
                    )
                    continue
                if active_task is None or (
                    cancel.request_id is not None
                    and cancel.request_id != active_request_id
                ):
                    await sender.send_json(
                        {
                            "type": "error",
                            "stage": "request",
                            "message": "no matching active request",
                        }
                    )
                    continue
                active_task.cancel()
                await asyncio.gather(active_task, return_exceptions=True)
                await sender.send_json(
                    {
                        "type": "conversation_end",
                        "request_id": active_request_id,
                        "cancelled": True,
                    }
                )
                active_task = None
                active_request_id = None
                continue

            if message_type == "ask":
                try:
                    request: AskRequest | SpeakRequest = AskRequest.model_validate(
                        payload
                    )
                except ValidationError as exc:
                    await sender.send_json(
                        {"type": "error", "stage": "request", "message": str(exc)}
                    )
                    continue
            elif message_type == "speak":
                try:
                    request = SpeakRequest.model_validate(payload)
                except ValidationError as exc:
                    await sender.send_json(
                        {"type": "error", "stage": "request", "message": str(exc)}
                    )
                    continue
            else:
                await sender.send_json(
                    {
                        "type": "error",
                        "stage": "request",
                        "message": f"unknown message type: {message_type!r}",
                    }
                )
                continue

            if active_task is not None and not active_task.done():
                await sender.send_json(
                    {
                        "type": "error",
                        "stage": "request",
                        "message": "a conversation request is already running",
                    }
                )
                continue
            active_request_id = request.request_id or uuid4().hex
            request = request.model_copy(update={"request_id": active_request_id})
            active_task = asyncio.create_task(
                _run_and_report(
                    sender,
                    request,
                    active_request_id,
                )
            )
    except WebSocketDisconnect:
        if active_task is not None:
            active_task.cancel()
            await asyncio.gather(active_task, return_exceptions=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
