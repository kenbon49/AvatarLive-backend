"""End-to-end streaming question-to-avatar API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Annotated, Any, AsyncIterator, Literal
from urllib.parse import quote
from uuid import uuid4
import wave

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
from fastapi.responses import Response
import httpx
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from scipy.signal import resample_poly
import websockets

from llm_inference import LLMInferenceError, LiteLLMClient

from .protocol import remap_media_packet
from .segmenter import PunctuationSegmenter, TextUnit


log = logging.getLogger("server-total")
TTS_SERVICE = os.getenv("TTS_SERVICE", "melotts").strip().lower()
if TTS_SERVICE not in {"melotts", "openvoice"}:
    raise RuntimeError(f"unsupported TTS_SERVICE: {TTS_SERVICE}")
TTS_URL = os.getenv(
    "TTS_URL",
    os.getenv("OPENVOICE_URL", "http://localhost:8084"),
).rstrip("/")
DEFAULT_VOICE_ID = os.getenv(
    "TTS_DEFAULT_VOICE",
    os.getenv("OPENVOICE_DEFAULT_VOICE", "default"),
)
LEGACY_VOICE_IDS = frozenset(
    {
        "default_female",
        "customer_service_female",
        "gentle_female",
        "corporate_narrator_male",
    }
)
MUSETALK_WS_URL = os.getenv("MUSETALK_WS_URL", "ws://localhost:8083/v1/stream")
MUSETALK_HTTP_URL = os.getenv("MUSETALK_HTTP_URL", "http://localhost:8083").rstrip("/")
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
TTS_CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", "/app/data/tts-cache"))
TTS_CACHE_VERSION = os.getenv("TTS_CACHE_VERSION", "v1").strip() or "v1"
TTS_PREPARE_CONCURRENCY = max(1, int(os.getenv("TTS_PREPARE_CONCURRENCY", "2")))
PLAYBACK_BUFFER_SECONDS = float(os.getenv("PIPELINE_PLAYBACK_BUFFER_SECONDS", "1.5"))
MEDIA_SEND_AHEAD_SECONDS = max(
    PLAYBACK_BUFFER_SECONDS,
    float(os.getenv("PIPELINE_MEDIA_SEND_AHEAD_SECONDS", "2.5")),
)
FIRST_UNIT_MIN_CHARS = int(os.getenv("PIPELINE_FIRST_UNIT_MIN_CHARS", "1"))
TARGET_UNIT_CHARS = int(os.getenv("PIPELINE_TARGET_UNIT_CHARS", "1"))
COALESCE_HARD_DELIMITERS = os.getenv(
    "PIPELINE_COALESCE_HARD_DELIMITERS", "0"
).lower() not in {"0", "false", "no"}
_END = object()
_tts_cache_locks: dict[str, asyncio.Lock] = {}
AvatarProfile = str
AVATAR_PROFILE_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,95}$"
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
    profile: AvatarProfile = Field(
        default=DEFAULT_AVATAR_PROFILE, pattern=AVATAR_PROFILE_PATTERN
    )
    language: Literal["ZH", "EN"] = "ZH"
    voice_id: str | None = Field(default=None, min_length=1, max_length=64)
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)
    source_time_seconds: float = Field(default=0.0, ge=0.0, le=3600.0)


class CancelRequest(BaseModel):
    type: Literal["cancel"]
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class SpeakRequest(BaseModel):
    type: Literal["speak"]
    text: str = Field(min_length=1, max_length=2000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    profile: AvatarProfile = Field(
        default=DEFAULT_AVATAR_PROFILE, pattern=AVATAR_PROFILE_PATTERN
    )
    language: Literal["ZH", "EN"] = "ZH"
    voice_id: str | None = Field(default=None, min_length=1, max_length=64)
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)
    source_time_seconds: float = Field(default=0.0, ge=0.0, le=3600.0)


SpeechText = Annotated[str, Field(min_length=1, max_length=2000)]


class SpeechPrepareRequest(BaseModel):
    texts: list[SpeechText] = Field(min_length=1, max_length=50)
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


@dataclass
class MediaSendPacer:
    """Keep generated media close enough to its real-time presentation clock."""

    max_ahead_seconds: float = MEDIA_SEND_AHEAD_SECONDS
    started_at: float = field(default_factory=time.perf_counter)

    async def wait_until_sendable(self, pts_us: int) -> None:
        target = self.started_at + max(
            0.0, pts_us / 1_000_000 - self.max_ahead_seconds
        )
        delay = target - time.perf_counter()
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
    try:
        await asyncio.to_thread(TTS_CACHE_DIR.mkdir, parents=True, exist_ok=True)
    except OSError:
        log.warning("TTS cache directory is unavailable", exc_info=True)
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


async def _musetalk_avatar_catalog() -> tuple[dict[str, object], ...]:
    static_names = {str(item["id"]): str(item["name"]) for item in AVATAR_CATALOG}
    try:
        response = await app.state.http.get(f"{MUSETALK_HTTP_URL}/v1/avatars")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("MuseTalk avatar catalog is not a list")
        catalog: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            profile_id = str(item.get("id") or "")
            if not re.fullmatch(AVATAR_PROFILE_PATTERN, profile_id):
                continue
            if item.get("custom") and (
                item.get("status") != "ready" or not item.get("prepared")
            ):
                continue
            catalog.append(
                {
                    "id": profile_id,
                    "name": static_names.get(profile_id, str(item.get("name") or profile_id)),
                    "default": profile_id == DEFAULT_AVATAR_PROFILE,
                    "custom": bool(item.get("custom")),
                    "avatar_id": str(item.get("avatar_id") or profile_id),
                    "version": str(item.get("version") or ""),
                }
            )
        if catalog:
            return tuple(catalog)
    except (AttributeError, httpx.HTTPError, ValueError, TypeError):
        log.warning("failed to refresh avatar catalog from MuseTalk")
    return AVATAR_CATALOG


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
        "tts_service": TTS_SERVICE,
        "tts_url": TTS_URL,
        "supports_voice_clone": TTS_SERVICE == "openvoice",
        "default_voice_id": DEFAULT_VOICE_ID,
        "tts_cache_version": TTS_CACHE_VERSION,
        "musetalk_ws_url": MUSETALK_WS_URL,
        "musetalk_http_url": MUSETALK_HTTP_URL,
        "musetalk_backend": MUSETALK_BACKEND,
        "musetalk_inference_dtype": MUSETALK_INFERENCE_DTYPE,
        "default_avatar": DEFAULT_AVATAR_PROFILE,
        "avatars": AVATAR_CATALOG,
        "playback_buffer_seconds": PLAYBACK_BUFFER_SECONDS,
    }


@app.get("/v1/avatars")
async def avatars() -> dict[str, object]:
    catalog = await _musetalk_avatar_catalog()
    return {
        "default": DEFAULT_AVATAR_PROFILE,
        "avatars": catalog,
    }


@app.get("/v1/voices")
async def voices() -> dict[str, object]:
    try:
        response = await app.state.http.get(f"{TTS_URL}/v1/speakers")
        response.raise_for_status()
        payload = response.json()
        speakers = payload.get("speakers", [])
        if not isinstance(speakers, list):
            raise ValueError(f"{TTS_SERVICE} returned an invalid speaker catalog")
        catalog = []
        for speaker in speakers:
            if not isinstance(speaker, dict) or not isinstance(speaker.get("id"), str):
                continue
            catalog.append(
                {
                    "voice_id": speaker["id"],
                    "name": speaker.get("name") or speaker["id"],
                    "kind": "preset" if speaker.get("default") is True else "clone",
                    "source": {
                        "provider": "MeloTTS" if TTS_SERVICE == "melotts" else "OpenVoice",
                        "sample_url": (
                            f"/v1/voices/{quote(speaker['id'], safe='')}/preview"
                        ),
                    },
                }
            )
        default_voice = payload.get("default", DEFAULT_VOICE_ID)
        return {
            "default": default_voice
            if isinstance(default_voice, str)
            else DEFAULT_VOICE_ID,
            "voices": catalog,
        }
    except (AttributeError, TypeError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(
            status_code=502, detail=f"{TTS_SERVICE} is unavailable: {exc}"
        ) from exc


@app.post("/v1/voices/clone", status_code=201)
async def create_cloned_voice(
    name: str = Form(..., min_length=1, max_length=80),
    audio: UploadFile = File(...),
) -> dict[str, object]:
    if TTS_SERVICE != "openvoice":
        raise HTTPException(
            status_code=409,
            detail="Voice cloning is unavailable while TTS_SERVICE=melotts",
        )
    content = await audio.read(50 * 1024 * 1024 + 1)
    await audio.close()
    if not content:
        raise HTTPException(status_code=400, detail="audio is empty")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio exceeds the 50 MB limit")
    response = await app.state.http.post(
        f"{TTS_URL}/v1/voices/clone",
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


def _wav_from_pcm(pcm: bytes, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


@app.get("/v1/voices/{voice_id}/preview")
async def voice_preview(voice_id: str) -> Response:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", voice_id):
        raise HTTPException(status_code=400, detail="invalid voice id")
    request = SpeakRequest(
        type="speak",
        text="你好，很高兴认识你。这是我的声音试听。",
        voice_id=voice_id,
    )
    try:
        pcm, _duration = await synthesize_speech(app, request, request.text)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise HTTPException(
            status_code=status if 400 <= status < 500 else 502,
            detail=f"{TTS_SERVICE} preview is unavailable",
        ) from exc
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502, detail=f"{TTS_SERVICE} preview failed: {exc}"
        ) from exc
    return Response(
        content=_wav_from_pcm(pcm),
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=86400"},
    )


async def stream_speech_chunks(
    app_: FastAPI,
    request: AskRequest | SpeakRequest,
    text: str,
    *,
    first_chunk_seconds: float | None = None,
) -> AsyncIterator[tuple[bytes, float]]:
    voice_id = request.voice_id or request.speaker or DEFAULT_VOICE_ID
    if voice_id in LEGACY_VOICE_IDS:
        voice_id = DEFAULT_VOICE_ID
    data: dict[str, str | float | int] = {
        "tts_text": text,
        "language": request.language.lower(),
        "speed": request.speed,
        "output_sample_rate": 16000,
    }
    if TTS_SERVICE == "openvoice":
        endpoint = "/v1/voice-clone"
        data.update({
            "speaker_id": voice_id,
            "stream": str(request.speed == 1.0).lower(),
        })
    else:
        endpoint = "/v1/tts"
    async with app_.state.http.stream(
        "POST",
        f"{TTS_URL}{endpoint}",
        data=data,
    ) as response:
        if response.is_error:
            await response.aread()
        response.raise_for_status()
        if response.headers.get("X-Audio-Sample-Format") != "s16le":
            raise RuntimeError(f"{TTS_SERVICE} returned an unsupported audio format")
        source_rate = int(response.headers.get("X-Audio-Sample-Rate", "0"))
        if source_rate <= 0:
            raise RuntimeError(f"{TTS_SERVICE} did not report its sample rate")
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
            raise RuntimeError(f"{TTS_SERVICE} returned invalid PCM audio")
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
        raise RuntimeError(f"{TTS_SERVICE} returned empty PCM audio")
    return b"".join(chunks), duration


def _effective_voice_id(request: AskRequest | SpeakRequest) -> str:
    voice_id = request.voice_id or request.speaker or DEFAULT_VOICE_ID
    return DEFAULT_VOICE_ID if voice_id in LEGACY_VOICE_IDS else voice_id


def _speech_cache_key(request: AskRequest | SpeakRequest, text: str) -> str:
    identity = json.dumps(
        {
            "cache_version": TTS_CACHE_VERSION,
            "language": request.language.lower(),
            "sample_rate": 16000,
            "service": TTS_SERVICE,
            "speed": request.speed,
            "text": text,
            "voice_id": _effective_voice_id(request),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _speech_cache_path(request: AskRequest | SpeakRequest, text: str) -> Path:
    key = _speech_cache_key(request, text)
    return TTS_CACHE_DIR / key[:2] / f"{key}.pcm"


def _read_cached_pcm(path: Path) -> bytes | None:
    try:
        pcm = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("failed to read TTS cache entry %s", path, exc_info=True)
        return None
    if pcm and len(pcm) % 2 == 0:
        return pcm
    log.warning("ignoring invalid TTS cache entry %s", path)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("failed to remove invalid TTS cache entry %s", path, exc_info=True)
    return None


def _write_cached_pcm(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(pcm)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def _load_cached_speech(
    request: AskRequest | SpeakRequest, text: str
) -> tuple[bytes, float] | None:
    pcm = await asyncio.to_thread(_read_cached_pcm, _speech_cache_path(request, text))
    if pcm is None:
        return None
    return pcm, len(pcm) / 32000.0


async def _store_cached_speech(
    request: AskRequest | SpeakRequest, text: str, pcm: bytes
) -> None:
    try:
        await asyncio.to_thread(
            _write_cached_pcm, _speech_cache_path(request, text), pcm
        )
    except OSError:
        # Cache persistence is an optimization; it must never be required for
        # live speech to continue.
        log.warning("failed to persist synthesized speech", exc_info=True)


async def get_or_synthesize_speech(
    app_: FastAPI, request: AskRequest | SpeakRequest, text: str
) -> tuple[bytes, float, bool]:
    cached = await _load_cached_speech(request, text)
    if cached is not None:
        return cached[0], cached[1], True

    cache_key = _speech_cache_key(request, text)
    lock = _tts_cache_locks.setdefault(cache_key, asyncio.Lock())
    try:
        async with lock:
            cached = await _load_cached_speech(request, text)
            if cached is not None:
                return cached[0], cached[1], True
            pcm, duration = await synthesize_speech(app_, request, text)
            await _store_cached_speech(request, text, pcm)
            return pcm, duration, False
    finally:
        if _tts_cache_locks.get(cache_key) is lock:
            _tts_cache_locks.pop(cache_key, None)


def _segment_speech_text(text: str) -> list[TextUnit]:
    segmenter = PunctuationSegmenter(
        first_unit_min_chars=FIRST_UNIT_MIN_CHARS,
        target_unit_chars=TARGET_UNIT_CHARS,
        coalesce_hard_delimiters=COALESCE_HARD_DELIMITERS,
    )
    return [*segmenter.feed(text), *segmenter.finish()]


@app.post("/v1/speech/prepare")
async def prepare_speech(request: SpeechPrepareRequest) -> dict[str, object]:
    started = time.perf_counter()
    speech_request = SpeakRequest(
        type="speak",
        text="prepare",
        language=request.language,
        voice_id=request.voice_id,
        speaker=request.speaker,
        speed=request.speed,
    )
    units: list[str] = []
    seen: set[str] = set()
    for text in request.texts:
        for unit in _segment_speech_text(text.strip()):
            if unit.text not in seen:
                seen.add(unit.text)
                units.append(unit.text)

    semaphore = asyncio.Semaphore(TTS_PREPARE_CONCURRENCY)

    async def prepare_unit(text: str) -> tuple[bool | None, str | None]:
        try:
            async with semaphore:
                _pcm, _duration, cached = await get_or_synthesize_speech(
                    app, speech_request, text
                )
            return cached, None
        except Exception as exc:
            log.exception("failed to prepare TTS cache for %r", text)
            return None, str(exc)

    results = await asyncio.gather(*(prepare_unit(text) for text in units))
    hits = sum(cached is True for cached, _error in results)
    generated = sum(cached is False for cached, _error in results)
    errors = [error for _cached, error in results if error]
    return {
        "requested_texts": len(request.texts),
        "units": len(units),
        "cache_hits": hits,
        "generated": generated,
        "failed": len(errors),
        "errors": errors[:5],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


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
    answer_parts: list[str] = []

    if kind == "speak":
        await sender.send_json({"type": "speak_start", "request_id": request_id})
        answer = request.text.strip()
        for unit in _segment_speech_text(answer):
            await queue.put(unit)
        await sender.send_json(
            {"type": "speak_result", "request_id": request_id, "text": answer}
        )
        await queue.put(_END)
        return answer

    segmenter = PunctuationSegmenter(
        first_unit_min_chars=FIRST_UNIT_MIN_CHARS,
        target_unit_chars=TARGET_UNIT_CHARS,
        coalesce_hard_delimiters=COALESCE_HARD_DELIMITERS,
    )
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
        cached = False
        if TTS_SERVICE == "melotts":
            pcm, chunk_duration, cached = await get_or_synthesize_speech(
                app, request, item.text
            )
            await audio_queue.put(
                AudioUnit(item, pcm, chunk_duration, chunk_index=0)
            )
            duration += chunk_duration
            chunks = 1
        else:
            cached_audio = await _load_cached_speech(request, item.text)
            cached = cached_audio is not None
            streamed_pcm: list[bytes] = []
            if cached_audio is not None:
                pcm, chunk_duration = cached_audio
                await audio_queue.put(
                    AudioUnit(item, pcm, chunk_duration, chunk_index=0)
                )
                duration = chunk_duration
                chunks = 1
            else:
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
                    streamed_pcm.append(pcm)
                    await audio_queue.put(AudioUnit(item, pcm, chunk_duration, chunks))
                    duration += chunk_duration
                    chunks += 1
            if streamed_pcm:
                await _store_cached_speech(request, item.text, b"".join(streamed_pcm))
        if chunks == 0:
            raise RuntimeError(f"{TTS_SERVICE} returned empty PCM audio")
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
                "cached": cached,
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
        media_pacer = MediaSendPacer()
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
                        "start_position": math.floor(request.source_time_seconds * fps),
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
                    outgoing, _packet_type, outgoing_pts_us = remap_media_packet(
                        message,
                        sequence=timeline.sequence,
                        pts_offset_us=timeline.pts_offset_us,
                    )
                    await media_pacer.wait_until_sendable(outgoing_pts_us)
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
            timeline.pts_offset_us += round(len(item.pcm) / 2 / 16000 * 1_000_000)
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
    avatar_catalog = await _musetalk_avatar_catalog()
    available_profiles = {str(item["id"]) for item in avatar_catalog}
    await sender.send_json(
        {
            "type": "ready",
            "protocol": "MSTK/2",
            "packet_header": "<4sBBHIIQ",
            "sample_rate": 16000,
            "playback_buffer_seconds": PLAYBACK_BUFFER_SECONDS,
            "default_profile": DEFAULT_AVATAR_PROFILE,
            "avatars": avatar_catalog,
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

            if request.profile not in available_profiles:
                await sender.send_json(
                    {
                        "type": "error",
                        "stage": "request",
                        "message": f"unknown or unpublished avatar: {request.profile}",
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
