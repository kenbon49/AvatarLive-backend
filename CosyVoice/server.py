# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FastAPI service for streaming CosyVoice zero-shot voice cloning."""

import argparse
import json
import logging
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402
from voice_registry import VoiceLibrary, VoiceRecord, WhisperTranscriber  # noqa: E402


LOGGER = logging.getLogger("cosyvoice.fastapi")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_PROMPT_SECONDS = 30


@dataclass(frozen=True)
class ReferenceVoice:
    speaker_id: str
    name: str
    prompt_text: str
    prompt_wav: Path
    kind: str = "preset"
    language: str = "zh-CN"
    whisper_text: str = ""
    source: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None

    @classmethod
    def from_record(cls, record: VoiceRecord) -> "ReferenceVoice":
        return cls(
            speaker_id=record.voice_id,
            name=record.name,
            prompt_text=record.prompt_text,
            prompt_wav=record.prompt_wav,
            kind=record.kind,
            language=record.language,
            whisper_text=record.whisper_text,
            source=record.source,
            created_at=record.created_at,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.speaker_id,
            "speaker_id": self.speaker_id,
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "prompt_text": self.prompt_text,
            "whisper_text": self.whisper_text,
            "source": self.source,
            "created_at": self.created_at,
        }


class ServiceState:
    def __init__(self) -> None:
        self.model = None
        self.references: Dict[str, ReferenceVoice] = {}
        self.library: Optional[VoiceLibrary] = None
        self.transcriber = WhisperTranscriber()
        self.inference_lock = threading.Lock()
        self.initial_token_hop_len: Optional[int] = None


state = ServiceState()
app = FastAPI(title="CosyVoice Voice Clone API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=[
        "X-Audio-Sample-Rate",
        "X-Audio-Channels",
        "X-Audio-Sample-Format",
    ],
)


def load_references(config_path: Path) -> Dict[str, ReferenceVoice]:
    if not config_path.is_file():
        raise FileNotFoundError(f"speaker config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict) or not config:
        raise ValueError("speaker config must be a non-empty JSON object")

    references: Dict[str, ReferenceVoice] = {}
    for speaker_id, item in config.items():
        if not isinstance(item, dict):
            raise ValueError(f"speaker {speaker_id!r} must be a JSON object")
        prompt_text = str(item.get("prompt_text", "")).strip()
        wav_value = str(item.get("prompt_wav", "")).strip()
        if not speaker_id.strip() or not prompt_text or not wav_value:
            raise ValueError(
                f"speaker {speaker_id!r} requires prompt_wav and prompt_text"
            )

        wav_path = Path(wav_value)
        if not wav_path.is_absolute():
            wav_path = (config_path.parent / wav_path).resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(
                f"reference audio for speaker {speaker_id!r} does not exist: {wav_path}"
            )
        validate_prompt_audio(load_wav(str(wav_path), 16000))
        references[speaker_id] = ReferenceVoice(
            speaker_id=speaker_id,
            name=str(item.get("name", speaker_id)),
            prompt_text=prompt_text,
            prompt_wav=wav_path,
        )
    return references


def add_control_tokens(prompt_text: str) -> str:
    if (
        type(state.model).__name__ == "CosyVoice3"
        and "<|endofprompt|>" not in prompt_text
    ):
        return f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
    return prompt_text


def cache_reference(reference: ReferenceVoice) -> None:
    model = state.model
    if model is None:
        raise RuntimeError("model is not loaded")
    with state.inference_lock:
        model.add_zero_shot_spk(
            add_control_tokens(reference.prompt_text),
            str(reference.prompt_wav),
            reference.speaker_id,
        )


def validate_prompt_audio(speech: torch.Tensor) -> None:
    duration = speech.shape[1] / 16000
    if duration <= 0:
        raise ValueError("reference audio is empty")
    if duration > MAX_PROMPT_SECONDS:
        raise ValueError(
            f"reference audio must not exceed {MAX_PROMPT_SECONDS} seconds"
        )


def remove_temp_prompt(path: Optional[Path]) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


async def store_uploaded_prompt(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg"}:
        suffix = ".wav"
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    await upload.close()
    if not data:
        raise HTTPException(status_code=400, detail="prompt_wav is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"prompt_wav exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit",
        )
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="cosyvoice_prompt_", suffix=suffix, delete=False
        ) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)
        validate_prompt_audio(load_wav(str(temp_path), 16000))
        return temp_path
    except Exception as exc:
        remove_temp_prompt(temp_path)
        raise HTTPException(
            status_code=400, detail=f"cannot decode prompt_wav: {exc}"
        ) from exc


def pcm_stream(
    tts_text: str,
    prompt_text: str,
    prompt_wav: Path,
    stream: bool,
    speed: float,
    output_sample_rate: Optional[int] = None,
    zero_shot_spk_id: str = "",
    temp_prompt: Optional[Path] = None,
) -> Iterator[bytes]:
    try:
        # Model internals contain shared caches, so one process serializes inference.
        with state.inference_lock:
            model = state.model
            if model is None:
                raise RuntimeError("model is not loaded")

            # CosyVoice2/3 grows this shared value during a request; restore the
            # configured first-chunk size so every request has the same latency.
            if state.initial_token_hop_len is not None:
                model.model.token_hop_len = state.initial_token_hop_len

            prompt_text = add_control_tokens(prompt_text)

            outputs = model.inference_zero_shot(
                tts_text,
                prompt_text,
                str(prompt_wav),
                zero_shot_spk_id=zero_shot_spk_id,
                stream=stream,
                speed=speed,
            )
            for output in outputs:
                speech = output["tts_speech"].detach().cpu()
                if output_sample_rate and output_sample_rate != model.sample_rate:
                    speech = torchaudio.functional.resample(
                        speech,
                        model.sample_rate,
                        output_sample_rate,
                    )
                speech = speech.numpy()
                pcm = (np.clip(speech, -1.0, 1.0) * 32767.0).astype("<i2")
                yield pcm.tobytes()
    finally:
        remove_temp_prompt(temp_prompt)


def next_chunk(iterator: Iterator[bytes]) -> Optional[bytes]:
    try:
        return next(iterator)
    except StopIteration:
        return None


def prepend_chunk(first_chunk: bytes, iterator: Iterator[bytes]) -> Iterator[bytes]:
    yield first_chunk
    yield from iterator


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state.model is not None else "loading",
        "model": type(state.model).__name__ if state.model is not None else None,
        "sample_rate": state.model.sample_rate if state.model is not None else None,
        "speaker_count": len(state.references),
    }


@app.get("/v1/speakers")
def list_speakers() -> dict:
    return {
        "default_voice_id": state.library.default_voice_id if state.library else None,
        "voices": [reference.public_dict() for reference in state.references.values()],
        "speakers": [
            reference.public_dict() for reference in state.references.values()
        ],
    }


@app.post("/v1/voices/clone", status_code=201)
async def create_cloned_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    if state.model is None or state.library is None:
        raise HTTPException(status_code=503, detail="service is not ready")
    temp_prompt = await store_uploaded_prompt(audio)
    try:
        prompt_text = await run_in_threadpool(state.transcriber.transcribe, temp_prompt)
        record = await run_in_threadpool(
            state.library.add_clone, name, prompt_text, temp_prompt
        )
        reference = ReferenceVoice.from_record(record)
        await run_in_threadpool(cache_reference, reference)
        state.references[reference.speaker_id] = reference
        return reference.public_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("voice clone registration failed")
        raise HTTPException(
            status_code=500, detail=f"voice clone registration failed: {exc}"
        ) from exc
    finally:
        remove_temp_prompt(temp_prompt)


@app.post("/v1/voice-clone")
async def voice_clone(
    tts_text: str = Form(...),
    speaker_id: Optional[str] = Form(None),
    prompt_text: Optional[str] = Form(None),
    prompt_wav: Optional[UploadFile] = File(None),
    stream: bool = Form(True),
    speed: float = Form(1.0),
    output_sample_rate: Optional[int] = Form(None),
) -> StreamingResponse:
    tts_text = tts_text.strip()
    speaker_id = speaker_id.strip() if speaker_id else None
    prompt_text = prompt_text.strip() if prompt_text else None
    if not tts_text:
        raise HTTPException(status_code=400, detail="tts_text must not be empty")
    if speed <= 0:
        raise HTTPException(status_code=400, detail="speed must be greater than zero")
    if output_sample_rate is not None and not 8000 <= output_sample_rate <= 48000:
        raise HTTPException(
            status_code=400,
            detail="output_sample_rate must be between 8000 and 48000",
        )
    if stream and speed != 1.0:
        raise HTTPException(
            status_code=400,
            detail="CosyVoice only supports speed=1.0 during streaming inference",
        )
    if speaker_id and prompt_wav is not None:
        raise HTTPException(
            status_code=400,
            detail="use either speaker_id or prompt_wav, not both",
        )

    if speaker_id:
        reference = state.references.get(speaker_id)
        if reference is None:
            raise HTTPException(
                status_code=404, detail=f"unknown speaker_id: {speaker_id}"
            )
        selected_text = reference.prompt_text
        selected_wav = reference.prompt_wav
        selected_speaker_id = reference.speaker_id
        temp_prompt = None
    else:
        if prompt_wav is None or not prompt_text:
            raise HTTPException(
                status_code=400,
                detail="provide speaker_id, or provide both prompt_wav and prompt_text",
            )
        selected_text = prompt_text
        selected_wav = await store_uploaded_prompt(prompt_wav)
        selected_speaker_id = ""
        temp_prompt = selected_wav

    if state.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    sample_rate = output_sample_rate or state.model.sample_rate
    headers = {
        "X-Audio-Sample-Rate": str(sample_rate),
        "X-Audio-Channels": "1",
        "X-Audio-Sample-Format": "s16le",
        "Content-Disposition": 'inline; filename="voice_clone.pcm"',
    }
    audio_iterator = pcm_stream(
        tts_text,
        selected_text,
        selected_wav,
        stream,
        speed,
        output_sample_rate=sample_rate,
        zero_shot_spk_id=selected_speaker_id,
        temp_prompt=temp_prompt,
    )
    try:
        # Generate the first audio block before sending HTTP 200. This converts
        # startup failures into a normal JSON error instead of a broken stream.
        first_chunk = await run_in_threadpool(next_chunk, audio_iterator)
    except Exception as exc:
        audio_iterator.close()
        LOGGER.exception("voice synthesis failed before the first audio chunk")
        raise HTTPException(
            status_code=500, detail=f"voice synthesis failed: {exc}"
        ) from exc
    if first_chunk is None:
        audio_iterator.close()
        raise HTTPException(status_code=500, detail="model returned no audio")

    return StreamingResponse(
        prepend_chunk(first_chunk, audio_iterator),
        media_type="application/octet-stream",
        headers=headers,
        background=BackgroundTask(remove_temp_prompt, temp_prompt),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model-dir",
        default=str(ROOT_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
        help="local CosyVoice2/3 model directory",
    )
    parser.add_argument(
        "--voice-manifest",
        type=Path,
        default=ROOT_DIR / "voice_library" / "manifest.json",
        help="JSON file containing preset reference voices and clone storage",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use FP16 autocast on CUDA (enabled by default)",
    )
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--load-vllm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    state.library = VoiceLibrary(args.voice_manifest.resolve())
    state.references = {
        record.voice_id: ReferenceVoice.from_record(record)
        for record in state.library.list()
    }
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    LOGGER.info("loading model from %s", args.model_dir)
    state.model = AutoModel(
        model_dir=args.model_dir,
        fp16=args.fp16,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
    )
    state.initial_token_hop_len = getattr(state.model.model, "token_hop_len", None)
    for reference in state.references.values():
        cache_reference(reference)
        LOGGER.info("cached preset speaker %s", reference.speaker_id)
    LOGGER.info(
        "loaded %s at %d Hz with %d preset speaker(s), fp16=%s",
        type(state.model).__name__,
        state.model.sample_rate,
        len(state.references),
        state.model.model.fp16,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
