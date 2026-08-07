"""ONNX Runtime inference API for the exported English/Chinese MeloTTS models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

from .split_utils import split_sentence
from .text import cleaned_text_to_sequence
from .text.cleaner import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGUAGE_CONFIG = {
    "ZH": {
        "frontend": "ZH_MIX_EN",
        "bert_dir": PROJECT_ROOT / "bert_model" / "multilingual",
        "bert_file": "multilingual_bert.onnx",
    },
    "EN": {
        "frontend": "EN",
        "bert_dir": PROJECT_ROOT / "bert_model" / "english_bert",
        "bert_file": "english_bert.onnx",
    },
}


def _cuda_provider_options(device_id):
    """Return CUDA provider options supported by the pinned ORT 1.17 runtime."""

    # ORT 1.17.1 rejects user_compute_stream during provider construction.
    # I/O Binding still keeps model inputs and outputs in CUDA memory.
    return {
        "device_id": str(device_id),
        "do_copy_in_default_stream": "1",
    }


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _intersperse(values, item):
    result = [item] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


def _generate_path(durations, x_mask):
    """Build the monotonic text-to-frame alignment outside the ONNX graphs."""

    durations = np.asarray(durations, dtype=np.int64)
    if durations.ndim != 3 or durations.shape[1] != 1:
        raise ValueError(f"Expected durations [B,1,T], got {durations.shape}")
    cumulative = np.cumsum(durations[:, 0], axis=1)
    y_lengths = cumulative[:, -1]
    max_frames = int(y_lengths.max())
    positions = np.arange(max_frames, dtype=np.int64)[None, :, None]
    previous = np.pad(cumulative[:, :-1], ((0, 0), (1, 0)))[:, None, :]
    current = cumulative[:, None, :]
    path = ((positions >= previous) & (positions < current)).astype(np.float32)
    path *= x_mask[:, 0, None, :]
    y_mask = (
        np.arange(max_frames, dtype=np.int64)[None, :] < y_lengths[:, None]
    ).astype(np.float32)[:, None, :]
    return path[:, None, :, :], y_mask


def _torch_dtype_to_numpy(dtype):
    import torch

    mapping = {
        torch.float32: np.float32,
        torch.int64: np.int64,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported I/O Binding tensor dtype: {dtype}") from error


def _run_with_torch_binding(session, inputs, outputs, device_id):
    """Run ORT directly against contiguous CUDA tensor buffers."""

    binding = session.io_binding()
    bound_inputs = []
    for name, value in inputs.items():
        value = value.contiguous()
        bound_inputs.append(value)
        binding.bind_input(
            name,
            "cuda",
            device_id,
            _torch_dtype_to_numpy(value.dtype),
            tuple(value.shape),
            value.data_ptr(),
        )
    for name, value in outputs.items():
        if not value.is_contiguous():
            raise ValueError(f"Bound output {name!r} must be contiguous")
        binding.bind_output(
            name,
            "cuda",
            device_id,
            _torch_dtype_to_numpy(value.dtype),
            tuple(value.shape),
            value.data_ptr(),
        )
    session.run_with_iobinding(binding)
    return tuple(outputs.values())


class _BertSession:
    def __init__(self, session, tokenizer, torch_device=None):
        self.session = session
        self.tokenizer = tokenizer
        self.input_names = {item.name for item in session.get_inputs()}
        self.torch_device = torch_device
        self.device_id = torch_device.index if torch_device is not None else None
        hidden_size = session.get_outputs()[0].shape[-1]
        if not isinstance(hidden_size, int):
            raise RuntimeError(f"BERT hidden size must be static, got {hidden_size!r}")
        self.hidden_size = hidden_size

    def feature(self, text, word2ph):
        encoded = self.tokenizer(text, return_tensors="np")
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(
            encoded.get("attention_mask", np.ones_like(input_ids)), dtype=np.int64
        )
        token_type_ids = np.asarray(
            encoded.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int64
        )
        candidates = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        features = self.session.run(
            ["features"],
            {name: value for name, value in candidates.items() if name in self.input_names},
        )[0][0]
        if features.shape[0] != len(word2ph):
            raise RuntimeError(
                f"BERT token length {features.shape[0]} does not match word2ph "
                f"length {len(word2ph)}"
            )
        expanded = np.repeat(features, np.asarray(word2ph, dtype=np.int64), axis=0)
        return np.ascontiguousarray(expanded.T, dtype=np.float32)

    def feature_cuda(self, text, word2ph):
        import torch

        encoded = self.tokenizer(text, return_tensors="np")
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        candidates = {
            "input_ids": input_ids,
            "attention_mask": np.asarray(
                encoded.get("attention_mask", np.ones_like(input_ids)), dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                encoded.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int64
            ),
        }
        inputs = {
            name: torch.as_tensor(value, device=self.torch_device)
            for name, value in candidates.items()
            if name in self.input_names
        }
        features = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], self.hidden_size),
            dtype=torch.float32,
            device=self.torch_device,
        )
        _run_with_torch_binding(
            self.session, inputs, {"features": features}, self.device_id
        )
        if features.shape[1] != len(word2ph):
            raise RuntimeError(
                f"BERT token length {features.shape[1]} does not match word2ph "
                f"length {len(word2ph)}"
            )
        repeats = torch.as_tensor(word2ph, dtype=torch.int64, device=self.torch_device)
        return torch.repeat_interleave(features[0], repeats, dim=0).transpose(0, 1).contiguous()


class OnnxBert:
    """Generate MeloTTS phone-level BERT features with ONNX Runtime."""

    def __init__(self, language, device="cuda", onnx_dir=None):
        language = language.upper()
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Only {sorted(LANGUAGE_CONFIG)} are supported")
        language_config = LANGUAGE_CONFIG[language]
        self.language_code = language
        self.onnx_dir = Path(onnx_dir or PROJECT_ROOT / "onnx_models").resolve()

        import onnxruntime as ort

        available = ort.get_available_providers()
        if device == "auto":
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        device_name = str(device).lower()
        self.torch_device = None
        self.device_id = None
        if device_name.startswith("cuda"):
            import torch

            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDAExecutionProvider is not available")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch is required for ONNX I/O Binding")
            self.device_id = (
                int(device_name.split(":", 1)[1]) if ":" in device_name else 0
            )
            self.torch_device = torch.device(f"cuda:{self.device_id}")
            provider_options = _cuda_provider_options(self.device_id)
            providers = [("CUDAExecutionProvider", provider_options)]
        elif device_name == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")

        model_path = self.onnx_dir / "bert" / language_config["bert_file"]
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX BERT model: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        if self.torch_device is not None:
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"ONNX BERT did not initialize on GPU: {model_path}")
            session.disable_fallback()
        tokenizer = AutoTokenizer.from_pretrained(
            language_config["bert_dir"], local_files_only=True
        )
        self.session = _BertSession(session, tokenizer, self.torch_device)

    @property
    def device(self):
        return self.torch_device or "cpu"

    def feature(self, text, word2ph):
        if self.torch_device is not None:
            return self.session.feature_cuda(text, word2ph)

        import torch

        return torch.from_numpy(self.session.feature(text, word2ph))


class OnnxTTS:
    """Synthesize speech with BERT and MeloTTS ONNX Runtime sessions."""

    def __init__(
        self,
        language,
        onnx_dir=None,
        device="cuda",
        config_path=None,
        seed=None,
    ):
        language = language.upper()
        if language not in LANGUAGE_CONFIG:
            raise ValueError(f"Only {sorted(LANGUAGE_CONFIG)} are supported")
        self.language_code = language
        language_config = LANGUAGE_CONFIG[language]
        self.language = language_config["frontend"]
        self.onnx_dir = Path(onnx_dir or PROJECT_ROOT / "onnx_models").resolve()

        config_path = Path(config_path or PROJECT_ROOT / "pretrained_model" / language / "config.json")
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        self.hps = _namespace(config)
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(config["symbols"])}
        self.rng = np.random.default_rng(seed)

        import onnxruntime as ort

        available = ort.get_available_providers()
        if device == "auto":
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        device_name = str(device).lower()
        require_cuda = device_name.startswith("cuda")
        self.use_io_binding = require_cuda
        self.torch_device = None
        self.device_id = None
        if require_cuda:
            import torch

            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDAExecutionProvider is not available")
            device_id = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch is required for ONNX I/O Binding")
            self.torch_device = torch.device(f"cuda:{device_id}")
            self.device_id = device_id
            self.torch_generator = torch.Generator(device=self.torch_device)
            if seed is None:
                self.torch_generator.seed()
            else:
                self.torch_generator.manual_seed(seed)
            cuda_options = _cuda_provider_options(device_id)
            providers = [("CUDAExecutionProvider", cuda_options)]
        elif device_name == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3

        def load_session(path):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing ONNX model: {path}")
            session = ort.InferenceSession(
                str(path), sess_options=session_options, providers=providers
            )
            if require_cuda:
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError(f"ONNX model did not initialize on GPU: {path}")
                session.disable_fallback()
            return session

        bert_path = self.onnx_dir / "bert" / language_config["bert_file"]
        tokenizer = AutoTokenizer.from_pretrained(
            language_config["bert_dir"], local_files_only=True
        )
        self.bert = _BertSession(
            load_session(bert_path), tokenizer, torch_device=self.torch_device
        )
        model_dir = self.onnx_dir / language
        self.encoder = load_session(model_dir / "encoder_duration.onnx")
        self.flow = load_session(model_dir / "flow.onnx")
        self.generator = load_session(model_dir / "generator.onnx")
        self.inter_channels = int(self.hps.model.inter_channels)
        self.gin_channels = int(self.hps.model.gin_channels)
        self.hop_length = int(np.prod(self.hps.model.upsample_rates))

    @staticmethod
    def _concat_audio(segments, sampling_rate, speed):
        silence = np.zeros(int(sampling_rate * 0.05 / speed), dtype=np.float32)
        parts = []
        for segment in segments:
            parts.extend((np.asarray(segment, dtype=np.float32), silence))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def _text_inputs(self, text):
        norm_text, phones, tones, word2ph = clean_text(text, self.language)
        phones, tones, language_ids = cleaned_text_to_sequence(
            phones, tones, self.language, self.symbol_to_id
        )
        if self.hps.data.add_blank:
            phones = _intersperse(phones, 0)
            tones = _intersperse(tones, 0)
            language_ids = _intersperse(language_ids, 0)
            word2ph = [count * 2 for count in word2ph]
            word2ph[0] += 1

        feature = self.bert.feature(norm_text, word2ph)
        if feature.shape[1] != len(phones):
            raise RuntimeError(
                f"BERT length {feature.shape[1]} does not match phone length {len(phones)}"
            )
        length = len(phones)
        return {
            "phones": np.asarray(phones, dtype=np.int64)[None, :],
            "lengths": np.asarray([length], dtype=np.int64),
            "tones": np.asarray(tones, dtype=np.int64)[None, :],
            "language_ids": np.asarray(language_ids, dtype=np.int64)[None, :],
            "bert": np.zeros((1, 1024, length), dtype=np.float32),
            "ja_bert": feature[None, :, :],
            "duration_noise": self.rng.standard_normal((1, 2, length)).astype(np.float32),
        }

    def _text_inputs_cuda(self, text):
        import torch

        norm_text, phones, tones, word2ph = clean_text(text, self.language)
        phones, tones, language_ids = cleaned_text_to_sequence(
            phones, tones, self.language, self.symbol_to_id
        )
        if self.hps.data.add_blank:
            phones = _intersperse(phones, 0)
            tones = _intersperse(tones, 0)
            language_ids = _intersperse(language_ids, 0)
            word2ph = [count * 2 for count in word2ph]
            word2ph[0] += 1

        feature = self.bert.feature_cuda(norm_text, word2ph)
        if feature.shape[1] != len(phones):
            raise RuntimeError(
                f"BERT length {feature.shape[1]} does not match phone length {len(phones)}"
            )
        length = len(phones)
        device = self.torch_device
        return {
            "phones": torch.tensor([phones], dtype=torch.int64, device=device),
            "lengths": torch.tensor([length], dtype=torch.int64, device=device),
            "tones": torch.tensor([tones], dtype=torch.int64, device=device),
            "language_ids": torch.tensor(
                [language_ids], dtype=torch.int64, device=device
            ),
            "bert": torch.zeros((1, 1024, length), device=device),
            "ja_bert": feature.unsqueeze(0),
            "duration_noise": torch.randn(
                (1, 2, length), device=device, generator=self.torch_generator
            ),
        }

    def _synthesize_sentence_cuda(
        self,
        sentence,
        speaker_id,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        speed,
        max_frames,
    ):
        import torch

        inputs = self._text_inputs_cuda(sentence)
        text_length = inputs["phones"].shape[1]
        inputs.update(
            {
                "speaker_ids": torch.tensor(
                    [speaker_id], dtype=torch.int64, device=self.torch_device
                ),
                "noise_scale_w": torch.tensor(
                    noise_scale_w, dtype=torch.float32, device=self.torch_device
                ),
            }
        )
        outputs = {
            "m_p": torch.empty(
                (1, self.inter_channels, text_length), device=self.torch_device
            ),
            "logs_p": torch.empty(
                (1, self.inter_channels, text_length), device=self.torch_device
            ),
            "x_mask": torch.empty((1, 1, text_length), device=self.torch_device),
            "logw_sdp": torch.empty((1, 1, text_length), device=self.torch_device),
            "logw_dp": torch.empty((1, 1, text_length), device=self.torch_device),
            "g": torch.empty(
                (1, self.gin_channels, 1), device=self.torch_device
            ),
        }
        m_p, logs_p, x_mask, logw_sdp, logw_dp, g = _run_with_torch_binding(
            self.encoder, inputs, outputs, self.device_id
        )
        logw = logw_sdp * float(sdp_ratio) + logw_dp * (1.0 - float(sdp_ratio))
        durations = torch.ceil(torch.exp(logw) * x_mask / float(speed)).to(torch.int64)
        cumulative = torch.cumsum(durations[:, 0], dim=1)
        frame_count = int(cumulative[:, -1].max().item())
        if frame_count < 1:
            raise RuntimeError("MeloTTS predicted an empty waveform")
        if frame_count > max_frames:
            raise RuntimeError(
                f"MeloTTS predicted {frame_count} frames, exceeding max_frames={max_frames}"
            )

        positions = torch.arange(frame_count, device=self.torch_device)[None, :, None]
        previous = torch.nn.functional.pad(cumulative[:, :-1], (1, 0))[:, None, :]
        attention = ((positions >= previous) & (positions < cumulative[:, None, :])).float()
        attention = attention * x_mask[:, 0, None, :]
        y_mask = (
            torch.arange(frame_count, device=self.torch_device)[None, :]
            < cumulative[:, -1, None]
        ).float()[:, None, :]
        expanded_m = torch.matmul(attention, m_p.transpose(1, 2)).transpose(1, 2)
        expanded_logs = torch.matmul(attention, logs_p.transpose(1, 2)).transpose(1, 2)
        latent_noise = torch.randn(
            expanded_m.shape, device=self.torch_device, generator=self.torch_generator
        )
        z_p = expanded_m + latent_noise * torch.exp(expanded_logs) * float(noise_scale)
        z = torch.empty(
            tuple(z_p.shape), dtype=z_p.dtype, device=self.torch_device
        )
        _run_with_torch_binding(
            self.flow,
            {"z_p": z_p, "y_mask": y_mask, "g": g},
            {"z": z},
            self.device_id,
        )
        audio = torch.empty(
            (1, 1, frame_count * self.hop_length), device=self.torch_device
        )
        _run_with_torch_binding(
            self.generator,
            {"z": z * y_mask, "g": g},
            {"audio": audio},
            self.device_id,
        )
        return audio[0, 0].cpu().numpy()

    def _synthesize_sentence(
        self,
        sentence,
        speaker_id,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        speed,
        max_frames,
    ):
        if self.use_io_binding:
            return self._synthesize_sentence_cuda(
                sentence,
                speaker_id,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                speed,
                max_frames,
            )
        inputs = self._text_inputs(sentence)
        inputs.update(
            {
                "speaker_ids": np.asarray([speaker_id], dtype=np.int64),
                "noise_scale_w": np.asarray(noise_scale_w, dtype=np.float32),
            }
        )
        m_p, logs_p, x_mask, logw_sdp, logw_dp, g = self.encoder.run(
            ["m_p", "logs_p", "x_mask", "logw_sdp", "logw_dp", "g"], inputs
        )
        logw = logw_sdp * float(sdp_ratio) + logw_dp * (1.0 - float(sdp_ratio))
        durations = np.ceil(np.exp(logw) * x_mask / float(speed)).astype(np.int64)
        frame_count = int(durations.sum())
        if frame_count < 1:
            raise RuntimeError("MeloTTS predicted an empty waveform")
        if frame_count > max_frames:
            raise RuntimeError(
                f"MeloTTS predicted {frame_count} frames, exceeding max_frames={max_frames}"
            )

        attention, y_mask = _generate_path(durations, x_mask)
        expanded_m = np.matmul(attention[:, 0], np.transpose(m_p, (0, 2, 1)))
        expanded_logs = np.matmul(attention[:, 0], np.transpose(logs_p, (0, 2, 1)))
        expanded_m = np.transpose(expanded_m, (0, 2, 1)).astype(np.float32)
        expanded_logs = np.transpose(expanded_logs, (0, 2, 1)).astype(np.float32)
        latent_noise = self.rng.standard_normal(expanded_m.shape).astype(np.float32)
        z_p = expanded_m + latent_noise * np.exp(expanded_logs) * float(noise_scale)
        z = self.flow.run(
            ["z"],
            {
                "z_p": np.ascontiguousarray(z_p, dtype=np.float32),
                "y_mask": np.ascontiguousarray(y_mask, dtype=np.float32),
                "g": np.ascontiguousarray(g, dtype=np.float32),
            },
        )[0]
        audio = self.generator.run(
            ["audio"],
            {
                "z": np.ascontiguousarray(z * y_mask, dtype=np.float32),
                "g": np.ascontiguousarray(g, dtype=np.float32),
            },
        )[0]
        return np.asarray(audio[0, 0], dtype=np.float32)

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
        max_frames=20000,
    ):
        if not text.strip():
            raise ValueError("Text must not be empty")
        if speed <= 0:
            raise ValueError("Speed must be greater than zero")
        if not 0 <= sdp_ratio <= 1:
            raise ValueError("sdp_ratio must be between 0 and 1")

        sentences = split_sentence(text, language_str=self.language)
        audio_segments = []
        for sentence in sentences:
            sentence = re.sub(r"([a-z])([A-Z])", r"\1 \2", sentence)
            audio_segments.append(
                self._synthesize_sentence(
                    sentence,
                    speaker_id,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    speed,
                    max_frames,
                )
            )
        audio = self._concat_audio(audio_segments, self.hps.data.sampling_rate, speed)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.hps.data.sampling_rate)
        return audio
