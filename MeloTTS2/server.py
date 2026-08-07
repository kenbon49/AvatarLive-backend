"""HTTP service for persistent MeloTTS inference."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import math
import os
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
import torch

from melo import TTS


log = logging.getLogger("melotts-server")


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: Literal["ZH", "EN"] = "ZH"
    speaker: str | None = None
    speed: float = Field(default=1.0, gt=0.25, le=3.0)
    sample_rate: Literal[16000] = 16000
    bert_backend: Literal["onnx"] = "onnx"


class ModelRegistry:
    """Load each language once and serialize inference on the model device."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._models: dict[str, TTS] = {}
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def get(self, language: str) -> TTS:
        model = self._models.get(language)
        if model is not None:
            return model
        with self._load_lock:
            model = self._models.get(language)
            if model is None:
                log.info(
                    "loading MeloTTS language=%s device=%s bert_backend=onnx",
                    language,
                    self.device,
                )
                model = TTS(language, device=self.device, bert_backend="onnx")
                self._models[language] = model
        return model

    def synthesize(self, request: SynthesisRequest) -> bytes:
        model = self.get(request.language)
        speakers = model.hps.data.spk2id
        speaker = request.speaker or next(iter(speakers))
        if speaker not in speakers:
            raise ValueError(
                f"unknown {request.language} speaker {speaker!r}; choose from {list(speakers)}"
            )
        with self._inference_lock:
            audio = model.tts_to_file(
                request.text.strip(),
                speakers[speaker],
                output_path=None,
                speed=request.speed,
                quiet=True,
            )
        source_rate = int(model.hps.data.sampling_rate)
        if source_rate != request.sample_rate:
            divisor = math.gcd(source_rate, request.sample_rate)
            audio = resample_poly(
                audio,
                request.sample_rate // divisor,
                source_rate // divisor,
            )
        pcm = (audio.clip(-1.0, 1.0) * 32767.0).round().astype("<i2")
        return pcm.tobytes()

    def speakers(self, language: str) -> list[str]:
        return list(self.get(language).hps.data.spk2id)

    def device_status(self) -> dict[str, object]:
        model_devices = {
            language: str(next(tts.model.parameters()).device)
            for language, tts in self._models.items()
        }
        bert_devices = {
            language: str(tts.bert.device)
            for language, tts in self._models.items()
            if tts.bert is not None
        }
        bert_backends = {
            language: tts.bert_backend for language, tts in self._models.items()
        }
        loaded_languages = set(self._models)
        resident_devices = [*model_devices.values(), *bert_devices.values()]
        gpu_ready = (
            bool(loaded_languages)
            and loaded_languages <= set(model_devices)
            and loaded_languages <= set(bert_devices)
            and all(torch.device(device).type == "cuda" for device in resident_devices)
        )
        return {
            "model_devices": model_devices,
            "bert_devices": bert_devices,
            "bert_backends": bert_backends,
            "gpu_ready": gpu_ready,
        }


DEVICE = os.getenv("MELOTTS_DEVICE", "cuda:0")
DEFAULT_LANGUAGE = os.getenv("MELOTTS_DEFAULT_LANGUAGE", "ZH").upper()
WARMUP_TEXT = os.getenv("MELOTTS_WARMUP_TEXT", "你好。")
registry = ModelRegistry(DEVICE)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if DEFAULT_LANGUAGE not in {"ZH", "EN"}:
        raise ValueError("MELOTTS_DEFAULT_LANGUAGE must be ZH or EN")
    # A short synthesis warms both ONNX BERT and the PyTorch acoustic model.
    warmup = SynthesisRequest(text=WARMUP_TEXT, language=DEFAULT_LANGUAGE)
    await asyncio.to_thread(registry.synthesize, warmup)
    yield


app = FastAPI(title="MeloTTS2 API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "melotts2",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(torch.device(DEVICE))
        if torch.device(DEVICE).type == "cuda" and torch.cuda.is_available()
        else None,
        "loaded_languages": sorted(registry._models),
        **registry.device_status(),
    }


@app.get("/v1/voices/{language}")
def voices(language: Literal["ZH", "EN"]) -> dict[str, object]:
    return {"language": language, "speakers": registry.speakers(language)}


@app.post("/v1/synthesize")
def synthesize(request: SynthesisRequest) -> Response:
    try:
        pcm = registry.synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("speech synthesis failed")
        raise HTTPException(status_code=500, detail="speech synthesis failed") from exc
    duration = len(pcm) / (request.sample_rate * 2.0)
    return Response(
        pcm,
        media_type="application/octet-stream",
        headers={
            "X-Audio-Format": "pcm_s16le_mono",
            "X-Sample-Rate": str(request.sample_rate),
            "X-Duration-Seconds": f"{duration:.6f}",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8084)
