"""Persistent HTTP service for direct MeloTTS synthesis."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import os
import threading
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response
import numpy as np
from scipy.signal import resample_poly
import torch

from melo import TTS


DEVICE = os.getenv("MELOTTS_DEVICE", "auto")
DEFAULT_VOICE_ID = os.getenv("TTS_DEFAULT_VOICE", "default")
WARMUP_TEXT = os.getenv("MELOTTS_WARMUP_TEXT", "你好，欢迎使用实时语音。").strip()
SUPPORTED_LANGUAGES = {
    "zh": ("ZH", "ZH"),
    "en": ("EN", "EN-Default"),
}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class MeloTTSEngine:
    """Keep MeloTTS and its BERT frontend resident between requests."""

    def __init__(self, device: str = DEVICE) -> None:
        self.device = resolve_device(device)
        self.gpu_name: str | None = None
        self.tf32_enabled = False
        if self.device.startswith("cuda"):
            self.gpu_name = torch.cuda.get_device_name(torch.device(self.device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            self.tf32_enabled = True
        self._lock = threading.Lock()
        self._models: dict[str, TTS] = {}
        self._models["zh"] = self._load_model("zh")
        self.warmup_completed = False
        if WARMUP_TEXT:
            self.synthesize(WARMUP_TEXT, "zh", 1.0, 16000)
            self.warmup_completed = True

    def _load_model(self, language: str) -> TTS:
        model_language, _speaker = SUPPORTED_LANGUAGES[language]
        return TTS(model_language, device=self.device)

    def _model_for(self, language: str) -> TTS:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language: {language}")
        if language not in self._models:
            self._models[language] = self._load_model(language)
        return self._models[language]

    def synthesize(
        self,
        text: str,
        language: str,
        speed: float,
        output_sample_rate: int,
    ) -> bytes:
        if not text.strip():
            raise ValueError("tts_text must not be empty")
        if not 0.25 < speed <= 3.0:
            raise ValueError("speed must be greater than 0.25 and at most 3.0")
        if output_sample_rate < 8000 or output_sample_rate > 48000:
            raise ValueError("output_sample_rate must be between 8000 and 48000")

        language = language.lower()
        with self._lock:
            model = self._model_for(language)
            _model_language, speaker = SUPPORTED_LANGUAGES[language]
            try:
                speaker_id = model.hps.data.spk2id[speaker]
            except KeyError as exc:
                raise RuntimeError(f"MeloTTS speaker is unavailable: {speaker}") from exc
            audio = model.tts_to_file(
                text,
                speaker_id,
                output_path=None,
                speed=speed,
                quiet=True,
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))

        samples = np.asarray(audio, dtype=np.float32)
        source_rate = int(model.hps.data.sampling_rate)
        if source_rate != output_sample_rate:
            divisor = int(np.gcd(source_rate, output_sample_rate))
            samples = resample_poly(
                samples,
                output_sample_rate // divisor,
                source_rate // divisor,
            )
        return (
            np.clip(np.rint(samples * 32767.0), -32768, 32767)
            .astype("<i2")
            .tobytes()
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = await asyncio.to_thread(MeloTTSEngine)
    yield


app = FastAPI(title="MeloTTS synthesis service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    engine: MeloTTSEngine = app.state.engine
    return {
        "status": "ok",
        "service": "melotts",
        "model": "MeloTTS",
        "device": engine.device,
        "gpu_name": engine.gpu_name,
        "tf32_enabled": engine.tf32_enabled,
        "warmup_completed": engine.warmup_completed,
        "default_voice_id": DEFAULT_VOICE_ID,
        "loaded_languages": sorted(engine._models),
    }


@app.get("/v1/speakers")
async def speakers() -> dict[str, Any]:
    return {
        "default": DEFAULT_VOICE_ID,
        "speakers": [
            {
                "id": DEFAULT_VOICE_ID,
                "name": "MeloTTS 中文女声",
                "default": True,
            }
        ],
    }


@app.post("/v1/tts")
async def synthesize(
    tts_text: str = Form(..., min_length=1),
    language: str = Form("zh"),
    speed: float = Form(1.0),
    output_sample_rate: int = Form(16000),
) -> Response:
    engine: MeloTTSEngine = app.state.engine
    try:
        pcm = await asyncio.to_thread(
            engine.synthesize,
            tts_text,
            language,
            speed,
            output_sample_rate,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(output_sample_rate),
            "X-Audio-Sample-Format": "s16le",
            "X-TTS-Provider": "MeloTTS",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MeloTTS HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("melotts_server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
