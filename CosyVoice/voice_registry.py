"""Persistent preset and user-cloned voice catalog for CosyVoice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

import torch
import torchaudio

from cosyvoice.utils.file_utils import load_wav


ROOT_DIR = Path(__file__).resolve().parent
VOICE_LIBRARY_DIR = Path(
    os.getenv("COSYVOICE_VOICE_LIBRARY", ROOT_DIR / "voice_library")
).resolve()
MANIFEST_PATH = VOICE_LIBRARY_DIR / "manifest.json"
CLONE_DIR = VOICE_LIBRARY_DIR / "clones"
VOICE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True)
class VoiceRecord:
    voice_id: str
    name: str
    kind: str
    language: str
    prompt_text: str
    prompt_wav: Path
    whisper_text: str = ""
    source: dict[str, Any] | None = None
    created_at: str | None = None

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("prompt_wav")
        return value


class VoiceLibrary:
    def __init__(self, manifest_path: Path = MANIFEST_PATH) -> None:
        self.manifest_path = manifest_path
        self.clone_dir = manifest_path.parent / "clones"
        self._lock = threading.RLock()
        self.default_voice_id = ""
        self._voices: dict[str, VoiceRecord] = {}
        self.reload()

    def _record_from_dict(self, item: dict[str, Any], base_dir: Path) -> VoiceRecord:
        voice_id = str(item.get("voice_id", "")).strip()
        if not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise ValueError(f"invalid voice_id in catalog: {voice_id!r}")
        prompt_wav = Path(str(item.get("prompt_wav", "")))
        if not prompt_wav.is_absolute():
            prompt_wav = (base_dir / prompt_wav).resolve()
        prompt_text = str(item.get("prompt_text", "")).strip()
        if not prompt_text or not prompt_wav.is_file():
            raise ValueError(f"voice {voice_id!r} has an invalid prompt")
        return VoiceRecord(
            voice_id=voice_id,
            name=str(item.get("name", voice_id)).strip() or voice_id,
            kind=str(item.get("kind", "preset")),
            language=str(item.get("language", "zh-CN")),
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            whisper_text=str(item.get("whisper_text", "")).strip(),
            source=item.get("source"),
            created_at=item.get("created_at"),
        )

    def reload(self) -> None:
        with self._lock:
            with self.manifest_path.open("r", encoding="utf-8") as source:
                manifest = json.load(source)
            voices: dict[str, VoiceRecord] = {}
            for item in manifest.get("voices", []):
                record = self._record_from_dict(item, self.manifest_path.parent)
                voices[record.voice_id] = record
            self.clone_dir.mkdir(parents=True, exist_ok=True)
            for metadata_path in sorted(self.clone_dir.glob("*/metadata.json")):
                with metadata_path.open("r", encoding="utf-8") as source:
                    record = self._record_from_dict(
                        json.load(source), metadata_path.parent
                    )
                voices[record.voice_id] = record
            default_voice_id = str(manifest.get("default_voice_id", ""))
            if default_voice_id not in voices:
                raise ValueError("default_voice_id does not exist in the voice catalog")
            self.default_voice_id = default_voice_id
            self._voices = voices

    def list(self) -> list[VoiceRecord]:
        with self._lock:
            return list(self._voices.values())

    def get(self, voice_id: str | None) -> VoiceRecord:
        resolved = voice_id or self.default_voice_id
        with self._lock:
            try:
                return self._voices[resolved]
            except KeyError as exc:
                raise KeyError(f"unknown voice_id: {resolved}") from exc

    def add_clone(self, name: str, prompt_text: str, source_audio: Path) -> VoiceRecord:
        name = name.strip()
        prompt_text = prompt_text.strip()
        if not name or len(name) > 80:
            raise ValueError("name must contain 1 to 80 characters")
        if not prompt_text:
            raise ValueError("Whisper did not recognize any speech")
        voice_id = f"clone_{uuid4().hex[:12]}"
        voice_dir = self.clone_dir / voice_id
        voice_dir.mkdir(parents=False, exist_ok=False)
        prompt_wav = voice_dir / "reference.wav"
        speech = load_wav(str(source_audio), 16000)
        torchaudio.save(str(prompt_wav), speech, 16000)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "voice_id": voice_id,
            "name": name,
            "kind": "clone",
            "language": "zh-CN",
            "prompt_wav": "reference.wav",
            "prompt_text": prompt_text,
            "whisper_text": prompt_text,
            "created_at": created_at,
            "source": {"provider": "user-upload"},
        }
        metadata_path = voice_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record = self._record_from_dict(metadata, voice_dir)
        with self._lock:
            self._voices[voice_id] = record
        return record


class WhisperTranscriber:
    """Lazy local Hugging Face Whisper inference with no network fallback."""

    def __init__(self) -> None:
        self.model_dir = Path(
            os.getenv("WHISPER_MODEL_DIR", ROOT_DIR.parent / "models" / "whisper")
        ).resolve()
        self.tokenizer_dir = Path(
            os.getenv(
                "WHISPER_TOKENIZER_DIR",
                ROOT_DIR.parent / ".tmp-whisper-tokenizer",
            )
        ).resolve()
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self._feature_extractor = None
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import (
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
            WhisperTokenizer,
        )

        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self._tokenizer = WhisperTokenizer.from_pretrained(
            self.tokenizer_dir,
            language="zh",
            task="transcribe",
            local_files_only=True,
        )
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device)
        self._model.generation_config.forced_decoder_ids = None
        self._model.config.forced_decoder_ids = None
        self._model.eval()

    def transcribe(self, audio_path: Path) -> str:
        with self._lock:
            self._load()
            speech = load_wav(str(audio_path), 16000).squeeze(0).cpu().numpy()
            inputs = self._feature_extractor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(self.device)
            prompt_ids = [self._model.config.decoder_start_token_id]
            prompt_ids.extend(
                token_id
                for _, token_id in self._tokenizer.get_decoder_prompt_ids(
                    language="zh", task="transcribe", no_timestamps=True
                )
            )
            decoder_input_ids = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                generated = self._model.generate(
                    input_features,
                    attention_mask=inputs.attention_mask.to(self.device),
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=256,
                )
            return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[
                0
            ].strip()
