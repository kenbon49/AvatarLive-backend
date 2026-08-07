"""Benchmark MuseTalk PyTorch and ONNX Runtime CUDA with production tensor shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def timed(fn: Callable[[], torch.Tensor], warmup: int, runs: int) -> tuple[list[float], torch.Tensor]:
    result = fn()
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples, result


def summarize(samples: list[float], batch: int) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)
    mean = statistics.fmean(samples)
    return {
        "mean_ms": round(mean, 3),
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "items_per_second": round(batch * 1000.0 / mean, 2),
    }


def accuracy(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    left = expected.detach().float().reshape(-1)
    right = actual.detach().float().reshape(-1)
    difference = (left - right).abs()
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0)
    return {
        "max_abs": float(difference.max().cpu()),
        "mean_abs": float(difference.mean().cpu()),
        "cosine": float(cosine.cpu()),
    }


def comparison(
    name: str,
    torch_fn: Callable[[], torch.Tensor],
    onnx_fn: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    batch: int,
) -> dict:
    torch_times, torch_output = timed(torch_fn, args.warmup, args.runs)
    onnx_times, onnx_output = timed(onnx_fn, args.warmup, args.runs)
    torch_stats = summarize(torch_times, batch)
    onnx_stats = summarize(onnx_times, batch)
    result = {
        "model": name,
        "batch": batch,
        "pytorch": torch_stats,
        "onnxruntime": onnx_stats,
        "onnx_speedup": round(torch_stats["mean_ms"] / onnx_stats["mean_ms"], 3),
        "accuracy": accuracy(torch_output, onnx_output),
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "models/onnx")
    parser.add_argument("--models", nargs="+", default=["all"], choices=("all", "unet", "vae_encoder", "vae_decoder", "whisper", "pipeline"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--whisper-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This comparison requires CUDA")
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import WhisperModel

    from musetalk.models.unet import PositionalEncoding
    from musetalk.onnx_inference import (
        OnnxUNet,
        OnnxVAEDecoder,
        OnnxVAEEncoder,
        OnnxWhisperEncoder,
    )

    device = torch.device(args.device)
    dtype = torch.float16
    selected = {"unet", "vae_encoder", "vae_decoder", "whisper", "pipeline"}
    if "all" not in args.models:
        selected = set(args.models)
    torch.manual_seed(7)
    results = []

    vae = None
    ort_encoder = None
    ort_decoder = None
    if selected & {"vae_encoder", "vae_decoder", "pipeline"}:
        vae = AutoencoderKL.from_pretrained(str(ROOT / "models/sd-vae"))
        vae = vae.eval().to(device=device, dtype=dtype)
    if "vae_encoder" in selected:
        ort_encoder = OnnxVAEEncoder(args.onnx_dir / "musetalk_vae_encoder.onnx", device)
        image = torch.randn(args.batch_size, 3, 256, 256, device=device, dtype=dtype)
        results.append(comparison(
            "vae_encoder_moments",
            lambda: vae.quant_conv(vae.encoder(image)),
            lambda: ort_encoder.moments(image),
            args,
            args.batch_size,
        ))
    if selected & {"vae_decoder", "pipeline"}:
        ort_decoder = OnnxVAEDecoder(args.onnx_dir / "musetalk_vae_decoder.onnx", device)
    if "vae_decoder" in selected:
        latent = torch.randn(args.batch_size, 4, 32, 32, device=device, dtype=dtype)
        scale = float(vae.config.scaling_factor)
        results.append(comparison(
            "vae_decoder",
            lambda: vae.decode(latent / scale, return_dict=False)[0],
            lambda: ort_decoder.decode_tensor(latent),
            args,
            args.batch_size,
        ))

    whisper = None
    if "whisper" in selected:
        whisper = WhisperModel.from_pretrained(str(ROOT / "models/whisper"))
        whisper = whisper.encoder.eval().to(device=device, dtype=dtype)
        ort_whisper = OnnxWhisperEncoder(args.onnx_dir / "musetalk_whisper_encoder.onnx", device)
        mel = torch.randn(args.whisper_batch_size, 80, 3000, device=device, dtype=dtype)
        results.append(comparison(
            "whisper_encoder_all_hidden_states",
            lambda: torch.stack(whisper(mel, output_hidden_states=True).hidden_states, dim=2),
            lambda: torch.stack(ort_whisper(mel).hidden_states, dim=2),
            args,
            args.whisper_batch_size,
        ))

    unet = None
    ort_unet = None
    if selected & {"unet", "pipeline"}:
        config = json.loads((ROOT / "models/musetalkV15/musetalk.json").read_text())
        unet = UNet2DConditionModel(**config)
        weights = torch.load(ROOT / "models/musetalkV15/unet.pth")
        unet.load_state_dict(weights)
        del weights
        unet = unet.eval().to(device=device, dtype=dtype)
        ort_unet = OnnxUNet(args.onnx_dir / "musetalk_unet.onnx", device)
    latent_input = torch.randn(args.batch_size, 8, 32, 32, device=device, dtype=dtype)
    prompt = torch.randn(args.batch_size, 50, 384, device=device, dtype=dtype)
    timestep = torch.zeros(1, device=device, dtype=torch.int64)
    if "unet" in selected:
        results.append(comparison(
            "unet",
            lambda: unet(latent_input, timestep, encoder_hidden_states=prompt).sample,
            lambda: ort_unet(latent_input, timestep, prompt).sample,
            args,
            args.batch_size,
        ))
    if "pipeline" in selected:
        pe = PositionalEncoding(384).eval().to(device=device, dtype=dtype)
        scale = float(vae.config.scaling_factor)

        def torch_pipeline() -> torch.Tensor:
            prediction = unet(latent_input, timestep, encoder_hidden_states=pe(prompt)).sample
            return vae.decode(prediction / scale, return_dict=False)[0]

        def onnx_pipeline() -> torch.Tensor:
            prediction = ort_unet(latent_input, timestep, pe(prompt)).sample
            return ort_decoder.decode_tensor(prediction)

        results.append(comparison(
            "streaming_unet_vae_pipeline",
            torch_pipeline,
            onnx_pipeline,
            args,
            args.batch_size,
        ))

    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "results": results,
    }
    output = args.output or args.onnx_dir / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
