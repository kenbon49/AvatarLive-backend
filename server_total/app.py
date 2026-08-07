"""End-to-end streaming question-to-avatar API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import httpx
from pydantic import BaseModel, Field, ValidationError
import websockets

from llm_inference import LLMInferenceError, LiteLLMClient

from .protocol import remap_media_packet
from .segmenter import PunctuationSegmenter, TextUnit


log = logging.getLogger("server-total")
MELOTTS_URL = os.getenv("MELOTTS_URL", "http://localhost:8084").rstrip("/")
MELOTTS_BERT_BACKEND = "onnx"
MUSETALK_WS_URL = os.getenv("MUSETALK_WS_URL", "ws://localhost:8083/v1/stream")
MUSETALK_BACKEND = os.getenv("MUSETALK_BACKEND", "onnx").strip().lower()
REQUEST_TIMEOUT = float(os.getenv("PIPELINE_REQUEST_TIMEOUT", "300"))
PCM_CHUNK_BYTES = 64 * 1024
SENTENCE_QUEUE_SIZE = int(os.getenv("PIPELINE_TEXT_QUEUE_SIZE", "4"))
AUDIO_QUEUE_SIZE = int(os.getenv("PIPELINE_AUDIO_QUEUE_SIZE", "2"))
FIRST_UNIT_MIN_CHARS = int(os.getenv("PIPELINE_FIRST_UNIT_MIN_CHARS", "12"))
TARGET_UNIT_CHARS = int(os.getenv("PIPELINE_TARGET_UNIT_CHARS", "20"))
IDLE_CHUNK_SECONDS = float(os.getenv("PIPELINE_IDLE_CHUNK_SECONDS", "1.0"))
IDLE_MAX_AHEAD_SECONDS = float(os.getenv("PIPELINE_IDLE_MAX_AHEAD_SECONDS", "1.5"))
IDLE_RETRY_SECONDS = float(os.getenv("PIPELINE_IDLE_RETRY_SECONDS", "1.0"))
_END = object()
IDLE_REQUEST_ID = "idle"
AvatarProfile = Literal[
    "chinese",
    "business_male_1",
    "casual_male",
    "middle_aged_male",
    "casual_conversation",
    "casual_female",
]
DEFAULT_AVATAR_PROFILE: AvatarProfile = "chinese"
AVATAR_CATALOG = (
    {"id": "chinese", "name": "Chinese", "default": True},
    {"id": "business_male_1", "name": "商务男1", "default": False},
    {"id": "casual_male", "name": "休闲风", "default": False},
    {"id": "middle_aged_male", "name": "中年", "default": False},
    {"id": "casual_conversation", "name": "休闲交流", "default": False},
    {"id": "casual_female", "name": "休闲女", "default": False},
)


class AskRequest(BaseModel):
    type: Literal["ask"]
    question: str = Field(min_length=1, max_length=2000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile: AvatarProfile = DEFAULT_AVATAR_PROFILE
    language: Literal["ZH", "EN"] = "ZH"
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
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)


class IdleStartRequest(BaseModel):
    type: Literal["idle_start"]
    profile: AvatarProfile = DEFAULT_AVATAR_PROFILE
    language: Literal["ZH", "EN"] = "ZH"
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)


class IdleStopRequest(BaseModel):
    type: Literal["idle_stop"]


@dataclass(frozen=True)
class AudioUnit:
    unit: TextUnit
    pcm: bytes
    duration: float
    elapsed_ms: int


@dataclass
class MediaTimeline:
    """Shared conversation-wide PTS/sequence so idle and answer chunks are seamless."""

    pts_offset_us: int = 0
    sequence: int = 0
    playback_started_at: float | None = None

    async def wait_for_idle_budget(self, duration_us: int) -> None:
        """Bound generated idle media so the browser queue cannot grow forever."""

        now = time.perf_counter()
        if self.playback_started_at is None:
            self.playback_started_at = now
        buffered_end = (self.pts_offset_us + duration_us) / 1_000_000
        elapsed = now - self.playback_started_at
        delay = buffered_end - elapsed - max(0.0, IDLE_MAX_AHEAD_SECONDS)
        if delay > 0:
            await asyncio.sleep(delay)


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
        "melotts_url": MELOTTS_URL,
        "melotts_bert_backend": MELOTTS_BERT_BACKEND,
        "musetalk_ws_url": MUSETALK_WS_URL,
        "musetalk_backend": MUSETALK_BACKEND,
        "default_avatar": DEFAULT_AVATAR_PROFILE,
        "avatars": AVATAR_CATALOG,
    }


@app.get("/v1/avatars")
async def avatars() -> dict[str, object]:
    return {
        "default": DEFAULT_AVATAR_PROFILE,
        "avatars": AVATAR_CATALOG,
    }


async def synthesize_speech(app_: FastAPI, request: AskRequest, text: str) -> tuple[bytes, float]:
    response = await app_.state.http.post(
        f"{MELOTTS_URL}/v1/synthesize",
        json={
            "text": text,
            "language": request.language,
            "speaker": request.speaker,
            "speed": request.speed,
            "sample_rate": 16000,
            "bert_backend": MELOTTS_BERT_BACKEND,
        },
    )
    response.raise_for_status()
    pcm = response.content
    if not pcm or len(pcm) % 2:
        raise RuntimeError("MeloTTS returned invalid PCM audio")
    duration = float(response.headers.get("X-Duration-Seconds", len(pcm) / 32000.0))
    return pcm, duration


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
    )
    answer_parts: list[str] = []

    if kind == "speak":
        await sender.send_json({"type": "speak_start", "request_id": request_id})
        for unit in segmenter.feed(request.text):
            await queue.put(unit)
        for unit in segmenter.finish():
            await queue.put(unit)
        answer = request.text.strip()
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
    audio_queue: asyncio.Queue[AudioUnit | object],
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
        pcm, duration = await synthesize_speech(app, request, item.text)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        await sender.send_json(
            {
                "type": "tts_result",
                "request_id": request_id,
                "seq": item.seq,
                "text": item.text,
                "duration_seconds": round(duration, 3),
                "elapsed_ms": elapsed_ms,
            }
        )
        await audio_queue.put(AudioUnit(item, pcm, duration, elapsed_ms))
        count += 1
    await audio_queue.put(_END)
    return count


async def _handshake_upstream(
    upstream: Any,
    sender: FrontendSender,
    request: AskRequest | SpeakRequest | IdleStartRequest,
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
    await sender.send_json(
        {
            "type": "musetalk_ready",
            "request_id": request_id,
            "profile": request.profile,
            "fps": initial.get("fps", 25),
            "sample_rate": 16000,
            "backend": upstream_backend,
        }
    )
    return float(initial.get("fps", 25))


async def _render_units(
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
    audio_queue: asyncio.Queue[AudioUnit | object],
    *,
    timeline: MediaTimeline | None = None,
    on_first_audio: Callable[[], Awaitable[None]] | None = None,
) -> int:
    timeline = timeline if timeline is not None else MediaTimeline()
    rendered = 0
    first_unit = True
    async with websockets.connect(
        MUSETALK_WS_URL,
        open_timeout=REQUEST_TIMEOUT,
        close_timeout=5,
        ping_timeout=None,
        max_size=None,
    ) as upstream:
        await _handshake_upstream(upstream, sender, request, request_id)

        while True:
            item = await audio_queue.get()
            if item is _END:
                break
            if not isinstance(item, AudioUnit):
                raise TypeError("audio queue received an invalid item")
            if first_unit and on_first_audio is not None:
                await on_first_audio()
                first_unit = False
            unit = item.unit
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
            await upstream.send(json.dumps({"type": "start", "profile": request.profile}))
            for offset in range(0, len(item.pcm), PCM_CHUNK_BYTES):
                await upstream.send(item.pcm[offset : offset + PCM_CHUNK_BYTES])
            await upstream.send(json.dumps({"type": "commit"}))

            packet_count = 0
            text_unit_sent = False
            while True:
                message = await asyncio.wait_for(upstream.recv(), timeout=REQUEST_TIMEOUT)
                if isinstance(message, bytes):
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
                control.update({"request_id": request_id, "segment_seq": unit.seq})
                await sender.send_json(control)
                if control.get("type") == "stream_end":
                    break

            await sender.send_json(
                {
                    "type": "segment_end",
                    "request_id": request_id,
                    "seq": unit.seq,
                    "text": unit.text,
                    "duration_seconds": round(item.duration, 3),
                    "packets": packet_count,
                }
            )
            timeline.pts_offset_us += round(len(item.pcm) / 2 / 16000 * 1_000_000)
            rendered += 1
    return rendered


async def _run_idle(
    sender: FrontendSender,
    request: IdleStartRequest,
    request_id: str,
    timeline: MediaTimeline,
) -> None:
    """Keep the avatar playing by continuously rendering silent audio through MuseTalk."""
    async with websockets.connect(
        MUSETALK_WS_URL,
        open_timeout=REQUEST_TIMEOUT,
        close_timeout=5,
        ping_timeout=None,
        max_size=None,
    ) as upstream:
        fps = await _handshake_upstream(upstream, sender, request, request_id)
        await sender.send_json(
            {
                "type": "idle_started",
                "request_id": request_id,
                "profile": request.profile,
                "fps": fps,
                "sample_rate": 16000,
            }
        )
        frame_duration_us = round(1_000_000 / fps)
        silence = b"\x00\x00" * int(IDLE_CHUNK_SECONDS * 16000)
        silence_duration_us = round(len(silence) / 2 / 16000 * 1_000_000)
        while True:
            chunk_max_pts = 0
            packet_count = 0
            await timeline.wait_for_idle_budget(silence_duration_us)
            await sender.send_json(
                {
                    "type": "idle_chunk",
                    "request_id": request_id,
                    "seconds": round(len(silence) / 2 / 16000, 3),
                }
            )
            await upstream.send(json.dumps({"type": "start", "profile": request.profile}))
            for offset in range(0, len(silence), PCM_CHUNK_BYTES):
                await upstream.send(silence[offset : offset + PCM_CHUNK_BYTES])
            await upstream.send(json.dumps({"type": "commit"}))
            try:
                while True:
                    message = await asyncio.wait_for(upstream.recv(), timeout=REQUEST_TIMEOUT)
                    if isinstance(message, bytes):
                        outgoing, _packet_type, pts_us = remap_media_packet(
                            message,
                            sequence=timeline.sequence,
                            pts_offset_us=timeline.pts_offset_us,
                        )
                        chunk_max_pts = max(chunk_max_pts, pts_us - timeline.pts_offset_us)
                        timeline.sequence += 1
                        packet_count += 1
                        await sender.send_bytes(outgoing)
                        continue
                    control = json.loads(message)
                    if control.get("type") == "error":
                        raise RuntimeError(control.get("message", "MuseTalk idle stream failed"))
                    control.update({"request_id": request_id})
                    await sender.send_json(control)
                    if control.get("type") == "stream_end":
                        break
            except asyncio.CancelledError:
                if packet_count:
                    timeline.pts_offset_us += chunk_max_pts + frame_duration_us
                raise
            timeline.pts_offset_us += max(
                silence_duration_us, chunk_max_pts + frame_duration_us
            )


async def run_pipeline(
    websocket: WebSocket | FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str | None = None,
    *,
    session: "ConversationSession | None" = None,
) -> None:
    sender = websocket if isinstance(websocket, FrontendSender) else FrontendSender(websocket)
    resolved_request_id = request_id or request.request_id or uuid4().hex
    started_at = time.perf_counter()
    kind = request.type
    text_queue: asyncio.Queue[TextUnit | object] = asyncio.Queue(
        maxsize=SENTENCE_QUEUE_SIZE
    )
    audio_queue: asyncio.Queue[AudioUnit | object] = asyncio.Queue(maxsize=AUDIO_QUEUE_SIZE)
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

    timeline = session.timeline if session is not None else None
    on_first_audio = session.stop_idle if session is not None else None

    tasks = [
        asyncio.create_task(
            _produce_text_units(sender, request, resolved_request_id, text_queue, kind=kind)
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
                timeline=timeline,
                on_first_audio=on_first_audio,
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


class ConversationSession:
    """Per-connection live state shared by idle and question tasks."""

    def __init__(self, sender: FrontendSender) -> None:
        self.sender = sender
        self.timeline = MediaTimeline()
        self.idle_enabled = False
        self.idle_task: asyncio.Task[None] | None = None

    async def start_idle(self, request: IdleStartRequest) -> None:
        await self.stop_idle()
        self.idle_task = asyncio.create_task(
            self._supervise_idle(request)
        )
        self.idle_task.add_done_callback(self._on_idle_done)

    async def _supervise_idle(self, request: IdleStartRequest) -> None:
        while self.idle_enabled:
            try:
                await _run_idle(self.sender, request, IDLE_REQUEST_ID, self.timeline)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("idle stream failed; retrying")
                await self.sender.send_json(
                    {
                        "type": "idle_error",
                        "request_id": IDLE_REQUEST_ID,
                        "message": str(exc),
                        "retry_seconds": IDLE_RETRY_SECONDS,
                    }
                )
                await asyncio.sleep(max(0.0, IDLE_RETRY_SECONDS))

    async def stop_idle(self) -> None:
        task, self.idle_task = self.idle_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def resume_idle(self, request: IdleStartRequest | None) -> None:
        if self.idle_enabled and request is not None:
            task = self.idle_task
            if task is None or task.done():
                await self.start_idle(request)

    def _on_idle_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            log.error("idle stream task failed: %s", error)


async def _run_and_report(
    sender: FrontendSender,
    request: AskRequest | SpeakRequest,
    request_id: str,
    *,
    session: ConversationSession | None = None,
    resume_request: IdleStartRequest | None = None,
) -> None:
    cancelled = False
    try:
        await run_pipeline(sender, request, request_id, session=session)
    except asyncio.CancelledError:
        cancelled = True
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
        log.exception("MeloTTS request failed")
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
    finally:
        if not cancelled and session is not None and resume_request is not None:
            await session.resume_idle(resume_request)


@app.websocket("/v1/conversation")
async def conversation(websocket: WebSocket) -> None:
    await websocket.accept()
    sender = FrontendSender(websocket)
    session = ConversationSession(sender)
    await sender.send_json(
        {
            "type": "ready",
            "protocol": "MSTK/2",
            "packet_header": "<4sBBHIIQ",
            "sample_rate": 16000,
            "default_profile": DEFAULT_AVATAR_PROFILE,
            "avatars": AVATAR_CATALOG,
        }
    )
    active_task: asyncio.Task[None] | None = None
    active_request_id: str | None = None
    idle_request: IdleStartRequest | None = None
    try:
        while True:
            payload = await websocket.receive_json()
            if active_task is not None and active_task.done():
                await asyncio.gather(active_task, return_exceptions=True)
                active_task = None
                active_request_id = None

            message_type = payload.get("type")

            if message_type == "idle_start":
                try:
                    idle_start = IdleStartRequest.model_validate(payload)
                except ValidationError as exc:
                    await sender.send_json(
                        {"type": "error", "stage": "request", "message": str(exc)}
                    )
                    continue
                session.idle_enabled = True
                idle_request = idle_start
                if active_task is None or active_task.done():
                    await session.start_idle(idle_start)
                continue

            if message_type == "idle_stop":
                session.idle_enabled = False
                idle_request = None
                await session.stop_idle()
                await sender.send_json({"type": "idle_stopped"})
                continue

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
                await session.resume_idle(idle_request)
                continue

            if message_type == "ask":
                try:
                    request: AskRequest | SpeakRequest = AskRequest.model_validate(payload)
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
                    session=session,
                    resume_request=idle_request,
                )
            )
    except WebSocketDisconnect:
        if active_task is not None:
            active_task.cancel()
            await asyncio.gather(active_task, return_exceptions=True)
        await session.stop_idle()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
