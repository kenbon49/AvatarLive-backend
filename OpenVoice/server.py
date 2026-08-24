"""Persistent HTTP service for OpenVoice V2 voice cloning."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
import numpy as np
from scipy.signal import resample_poly
import torch

from inference import (
    MODEL_CONFIG,
    ROOT,
    load_converter,
    load_v2_synthesizer,
    require_file,
    resolve_device,
    se_extractor,
    suppress_model_output,
    synchronize_device,
)


VOICE_ROOT = Path(os.getenv("OPENVOICE_VOICE_DIR", ROOT / "voice_library")).resolve()
PROCESSED_ROOT = Path(
    os.getenv("OPENVOICE_PROCESSED_DIR", ROOT / "processed")
).resolve()
DEFAULT_REFERENCE = Path(
    os.getenv("OPENVOICE_DEFAULT_REFERENCE", ROOT.parent / "data/input/audio/yongen.wav")
).resolve()
DEFAULT_VOICE_ID = os.getenv("OPENVOICE_DEFAULT_VOICE", "default")
DEVICE = os.getenv("OPENVOICE_DEVICE", "auto")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SUPPORTED_LANGUAGES = frozenset({"zh", "en"})
SAFE_ID = re.compile(r"[^a-zA-Z0-9_-]+")


class OpenVoiceEngine:
    """Keep OpenVoice models and extracted target embeddings resident in memory."""

    def __init__(self, device: str = DEVICE) -> None:
        self.device = resolve_device(device)
        self._lock = threading.Lock()
        self._models: dict[str, tuple[Any, torch.Tensor]] = {}
        self._target_embeddings: dict[str, torch.Tensor] = {}
        VOICE_ROOT.mkdir(parents=True, exist_ok=True)
        PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

        with suppress_model_output():
            self.converter = load_converter("v2", self.device)
            se_extractor.get_local_vad_model()
            self._models["zh"] = self._load_synthesizer("zh")

        require_file(DEFAULT_REFERENCE, "default reference audio")
        self._write_default_metadata()
        self._target_embeddings[DEFAULT_VOICE_ID] = self._load_or_extract_embedding(
            DEFAULT_VOICE_ID, DEFAULT_REFERENCE
        )

    def _load_synthesizer(self, language: str) -> tuple[Any, torch.Tensor]:
        args = argparse.Namespace(language=language, speaker="auto")
        return load_v2_synthesizer(args, self.device)

    def _model_for(self, language: str) -> tuple[Any, torch.Tensor]:
        if language not in self._models:
            with suppress_model_output():
                self._models[language] = self._load_synthesizer(language)
        return self._models[language]

    @staticmethod
    def _voice_dir(voice_id: str) -> Path:
        return VOICE_ROOT / voice_id

    def _write_default_metadata(self) -> None:
        voice_dir = self._voice_dir(DEFAULT_VOICE_ID)
        voice_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": DEFAULT_VOICE_ID,
            "name": "Default OpenVoice clone",
            "reference": str(DEFAULT_REFERENCE),
            "default": True,
        }
        (voice_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _reference_for(self, voice_id: str) -> Path:
        if voice_id == DEFAULT_VOICE_ID:
            return DEFAULT_REFERENCE
        voice_dir = self._voice_dir(voice_id)
        metadata_path = voice_dir / "metadata.json"
        if not metadata_path.is_file():
            raise KeyError(voice_id)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reference = voice_dir / metadata["reference"]
        require_file(reference, "voice reference audio")
        return reference

    def _load_or_extract_embedding(
        self, voice_id: str, reference: Path
    ) -> torch.Tensor:
        embedding_path = self._voice_dir(voice_id) / "embedding.pth"
        if embedding_path.is_file():
            return torch.load(embedding_path, map_location=self.device).to(self.device)
        with suppress_model_output():
            embedding, _ = se_extractor.get_se(
                str(reference),
                self.converter,
                target_dir=str(PROCESSED_ROOT),
                vad=True,
            )
        torch.save(embedding.detach().cpu(), embedding_path)
        return embedding

    def target_embedding(self, voice_id: str) -> torch.Tensor:
        if voice_id not in self._target_embeddings:
            reference = self._reference_for(voice_id)
            self._target_embeddings[voice_id] = self._load_or_extract_embedding(
                voice_id, reference
            )
        return self._target_embeddings[voice_id]

    def list_voices(self) -> list[dict[str, Any]]:
        voices = []
        for metadata_path in sorted(VOICE_ROOT.glob("*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            voices.append(metadata)
        voices.sort(key=lambda item: (not item.get("default", False), item["id"]))
        return voices

    def register_voice(
        self, name: str, filename: str | None, content: bytes
    ) -> dict[str, Any]:
        with self._lock:
            return self._register_voice(name, filename, content)

    def _register_voice(
        self, name: str, filename: str | None, content: bytes
    ) -> dict[str, Any]:
        digest = hashlib.sha256(content).hexdigest()[:12]
        name_slug = SAFE_ID.sub("-", name).strip("-").lower()[:32] or "voice"
        voice_id = f"{name_slug}-{digest}"
        voice_dir = self._voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "reference.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            suffix = ".wav"
        reference_name = f"reference{suffix}"
        reference_path = voice_dir / reference_name
        reference_path.write_bytes(content)
        metadata = {
            "id": voice_id,
            "name": name.strip(),
            "reference": reference_name,
            "default": False,
        }
        (voice_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._target_embeddings[voice_id] = self._load_or_extract_embedding(
            voice_id, reference_path
        )
        return metadata

    def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str,
        speed: float,
        output_sample_rate: int,
    ) -> bytes:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language: {language}")
        if not text.strip():
            raise ValueError("tts_text must not be empty")
        if not 0.25 < speed <= 3.0:
            raise ValueError("speed must be greater than 0.25 and at most 3.0")
        if output_sample_rate < 8000 or output_sample_rate > 48000:
            raise ValueError("output_sample_rate must be between 8000 and 48000")

        with self._lock, tempfile.TemporaryDirectory(prefix="openvoice-") as temp_dir:
            model, source_embedding = self._model_for(language)
            target_embedding = self.target_embedding(voice_id)
            base_path = Path(temp_dir) / "base.wav"
            language_config = MODEL_CONFIG["v2"]["languages"][language]
            speaker_config = language_config["speakers"][
                language_config["default_speaker"]
            ]
            with suppress_model_output():
                model.tts_to_file(
                    text,
                    model.hps.data.spk2id[speaker_config["melo_speaker"]],
                    str(base_path),
                    speed=speed,
                    quiet=True,
                )
                audio = self.converter.convert(
                    str(base_path),
                    source_embedding,
                    target_embedding,
                    output_path=None,
                    message="@MyShell",
                )
            synchronize_device(self.device)

        source_rate = int(self.converter.hps.data.sampling_rate)
        samples = np.asarray(audio, dtype=np.float32)
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
    app.state.engine = await asyncio.to_thread(OpenVoiceEngine)
    yield


app = FastAPI(title="OpenVoice cloning service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    engine: OpenVoiceEngine = app.state.engine
    return {
        "status": "ok",
        "service": "openvoice",
        "version": "v2",
        "device": engine.device,
        "default_voice_id": DEFAULT_VOICE_ID,
        "voices": len(engine.list_voices()),
    }


@app.get("/v1/speakers")
async def speakers() -> dict[str, Any]:
    engine: OpenVoiceEngine = app.state.engine
    return {"default": DEFAULT_VOICE_ID, "speakers": engine.list_voices()}


@app.post("/v1/voices/clone", status_code=201)
async def clone_voice(
    name: str = Form(..., min_length=1, max_length=80),
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    content = await audio.read(MAX_UPLOAD_BYTES + 1)
    await audio.close()
    if not content:
        raise HTTPException(status_code=400, detail="audio is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio exceeds the 50 MB limit")
    engine: OpenVoiceEngine = app.state.engine
    try:
        voice = await asyncio.to_thread(
            engine.register_voice, name, audio.filename, content
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"voice_id": voice["id"], "voice": voice}


@app.post("/v1/voice-clone")
async def voice_clone(
    tts_text: str = Form(..., min_length=1),
    speaker_id: str = Form(DEFAULT_VOICE_ID),
    language: str = Form("zh"),
    speed: float = Form(1.0),
    output_sample_rate: int = Form(16000),
    stream: bool = Form(False),
) -> Response:
    del stream  # OpenVoice returns one generated utterance; the gateway rechunks PCM.
    engine: OpenVoiceEngine = app.state.engine
    try:
        pcm = await asyncio.to_thread(
            engine.synthesize,
            tts_text,
            speaker_id,
            language.lower(),
            speed,
            output_sample_rate,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown speaker: {speaker_id}") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": str(output_sample_rate),
            "X-Audio-Sample-Format": "s16le",
            "X-OpenVoice-Speaker": speaker_id,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVoice HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
