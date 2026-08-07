from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch
from torch import nn
from tqdm import tqdm

from .models import SynthesizerTrn
from .split_utils import split_sentence
from .utils import get_hparams_from_file, get_text_for_tts_infer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LANGUAGES = {"ZH", "EN"}


class TTS(nn.Module):
    """Offline MeloTTS with ONNX BERT and PyTorch acoustics by default."""

    def __init__(
        self,
        language,
        device="auto",
        config_path=None,
        ckpt_path=None,
        bert_backend="onnx",
        onnx_dir=None,
    ):
        super().__init__()
        language = language.upper()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Only {sorted(SUPPORTED_LANGUAGES)} are supported, got {language!r}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        model_dir = PROJECT_ROOT / "pretrained_model" / language
        config_path = Path(config_path) if config_path else model_dir / "config.json"
        ckpt_path = Path(ckpt_path) if ckpt_path else model_dir / "checkpoint.pth"
        if not config_path.is_file() or not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing model files under {model_dir}")

        hps = get_hparams_from_file(config_path)
        model = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=hps.num_tones,
            num_languages=hps.num_languages,
            **hps.model,
        ).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        self.model = model
        self.hps = hps
        self.language_code = language
        self.language = "ZH_MIX_EN" if language == "ZH" else "EN"
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(hps.symbols)}
        self.bert_backend = bert_backend.lower()
        if self.bert_backend == "onnx":
            from .onnx_api import OnnxBert

            self.bert = OnnxBert(language, device=str(self.device), onnx_dir=onnx_dir)
        elif self.bert_backend == "pytorch":
            self.bert = None
        else:
            raise ValueError("bert_backend must be onnx or pytorch")

    @staticmethod
    def _concat_audio(segments, sampling_rate, speed):
        silence = np.zeros(int(sampling_rate * 0.05 / speed), dtype=np.float32)
        parts = []
        for segment in segments:
            parts.extend((segment.astype(np.float32), silence))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def tts_to_file(
        self,
        text,
        speaker_id,
        output_path=None,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        speed=1.0,
        quiet=False,
    ):
        if not text.strip():
            raise ValueError("Text must not be empty")
        if speed <= 0:
            raise ValueError("Speed must be greater than zero")

        texts = split_sentence(text, language_str=self.language)
        iterator = texts if quiet else tqdm(texts, desc="Synthesizing")
        audio_segments = []
        for sentence in iterator:
            sentence = re.sub(r"([a-z])([A-Z])", r"\1 \2", sentence)
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
                sentence,
                self.language,
                self.hps,
                self.device,
                self.symbol_to_id,
                bert_feature=self.bert.feature if self.bert is not None else None,
            )
            with torch.inference_mode():
                phones = phones.to(self.device).unsqueeze(0)
                tones = tones.to(self.device).unsqueeze(0)
                lang_ids = lang_ids.to(self.device).unsqueeze(0)
                bert = bert.to(self.device).unsqueeze(0)
                ja_bert = ja_bert.to(self.device).unsqueeze(0)
                lengths = torch.tensor([phones.size(1)], device=self.device)
                speakers = torch.tensor([speaker_id], device=self.device)
                audio = self.model.infer(
                    phones,
                    lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1.0 / speed,
                )[0][0, 0].cpu().float().numpy()
            audio_segments.append(audio)

        audio = self._concat_audio(audio_segments, self.hps.data.sampling_rate, speed)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.hps.data.sampling_rate)
        return audio
