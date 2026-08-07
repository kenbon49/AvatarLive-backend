"""ONNX Runtime CUDA inference adapters for MuseTalk's exported models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


_NUMPY_DTYPES = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


class OnnxRuntimeError(RuntimeError):
    """Raised when the requested ONNX execution provider cannot be used."""


class OrtTorchSession:
    """Run ORT with torch tensors and zero-copy CUDA I/O binding."""

    def __init__(self, path: str | Path, device: str | torch.device = "cuda:0") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxRuntimeError(
                "onnxruntime-gpu is required for the ONNX backend"
            ) from exc

        self.path = Path(path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        available = set(ort.get_available_providers())
        if self.device.type == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise OnnxRuntimeError(
                    "CUDAExecutionProvider is unavailable; install an onnxruntime-gpu "
                    "build compatible with the installed CUDA/cuDNN runtime"
                )
            providers: list[Any] = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self.device.index or 0,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.path), options, providers=providers)
        active = self.session.get_providers()
        if self.device.type == "cuda" and active[0] != "CUDAExecutionProvider":
            raise OnnxRuntimeError(f"ONNX Runtime fell back to {active[0]} for {self.path}")

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def run(self, inputs: dict[str, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        inputs = {
            name: value.detach().to(self.device).contiguous()
            for name, value in inputs.items()
        }
        metadata = {item.name: item for item in self.session.get_inputs()}
        for name, value in inputs.items():
            expected = metadata[name].shape
            for axis, (wanted, actual) in enumerate(zip(expected, value.shape)):
                if isinstance(wanted, int) and wanted != actual:
                    raise OnnxRuntimeError(
                        f"{self.path.name} input {name!r} axis {axis} requires {wanted}, "
                        f"got {actual}; re-export ONNX with the service batch size"
                    )
        if self.device.type != "cuda":
            arrays = {name: value.cpu().numpy() for name, value in inputs.items()}
            result = self.session.run(None, arrays)[0]
            return torch.from_numpy(result).to(dtype=output.dtype)

        binding = self.session.io_binding()
        device_id = self.device.index or 0
        for name, value in inputs.items():
            binding.bind_input(
                name,
                "cuda",
                device_id,
                _NUMPY_DTYPES[value.dtype],
                tuple(value.shape),
                value.data_ptr(),
            )
        binding.bind_output(
            self.session.get_outputs()[0].name,
            "cuda",
            device_id,
            _NUMPY_DTYPES[output.dtype],
            tuple(output.shape),
            output.data_ptr(),
        )
        torch.cuda.synchronize(self.device)
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return output


class OnnxUNet:
    """Drop-in callable matching diffusers' UNet `.sample` result."""

    def __init__(self, path: str | Path, device: str | torch.device, dtype=torch.float16):
        self.runtime = OrtTorchSession(path, device)
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = self.runtime.session.get_inputs()[0].shape[0]

    def __call__(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> SimpleNamespace:
        latent = latent.to(device=self.device, dtype=self.dtype)
        prompt = encoder_hidden_states.to(device=self.device, dtype=self.dtype)
        output = torch.empty(
            (latent.shape[0], 4, latent.shape[2], latent.shape[3]),
            device=self.device,
            dtype=self.dtype,
        )
        result = self.runtime.run(
            {"latent": latent, "timestep": timestep, "audio_prompt": prompt}, output
        )
        return SimpleNamespace(sample=result)


class OnnxVAEDecoder:
    """Decode scaled MuseTalk latents and preserve the legacy BGR uint8 API."""

    def __init__(self, path: str | Path, device: str | torch.device, dtype=torch.float16):
        self.runtime = OrtTorchSession(path, device)
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = self.runtime.session.get_inputs()[0].shape[0]

    def decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(device=self.device, dtype=self.dtype)
        output = torch.empty(
            (latent.shape[0], 3, latent.shape[2] * 8, latent.shape[3] * 8),
            device=self.device,
            dtype=self.dtype,
        )
        return self.runtime.run({"latent": latent}, output)

    def decode_latents(self, latent: torch.Tensor) -> np.ndarray:
        image = (self.decode_tensor(latent) / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float().cpu().numpy()
        return (image[..., ::-1] * 255).round().astype(np.uint8)


class OnnxVAEEncoder:
    """Encode images to moments; callers choose deterministic mode or sampling."""

    def __init__(self, path: str | Path, device: str | torch.device, dtype=torch.float16):
        self.runtime = OrtTorchSession(path, device)
        self.device = torch.device(device)
        self.dtype = dtype

    def moments(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.device, dtype=self.dtype)
        output = torch.empty(
            (image.shape[0], 8, image.shape[2] // 8, image.shape[3] // 8),
            device=self.device,
            dtype=self.dtype,
        )
        return self.runtime.run({"image": image}, output)

    def encode(self, image: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        mean, logvar = self.moments(image).chunk(2, dim=1)
        if not sample:
            return mean
        logvar = logvar.clamp(-30.0, 20.0)
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


class OnnxWhisperEncoder:
    """Adapter matching `WhisperModel.encoder(...).hidden_states`."""

    def __init__(self, path: str | Path, device: str | torch.device, dtype=torch.float16):
        self.runtime = OrtTorchSession(path, device)
        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(
        self,
        input_features: torch.Tensor,
        output_hidden_states: bool = True,
        **_: Any,
    ) -> SimpleNamespace:
        if not output_hidden_states:
            raise ValueError("MuseTalk's Whisper ONNX adapter requires hidden states")
        features = input_features.to(device=self.device, dtype=self.dtype)
        output = torch.empty(
            (features.shape[0], 1500, 5, 384),
            device=self.device,
            dtype=self.dtype,
        )
        stacked = self.runtime.run({"input_features": features}, output)
        return SimpleNamespace(hidden_states=tuple(stacked.unbind(dim=2)))
