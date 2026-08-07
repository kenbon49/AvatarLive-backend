"""Export the neural-network parts of MuseTalk 1.5 to ONNX.

Image/audio preprocessing, face detection, positional encoding and blending stay
outside ONNX because they are inexpensive Python/OpenCV orchestration steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def register_scaled_dot_product_attention_symbolic(opset: int) -> None:
    """Backport the SDPA ONNX decomposition missing from PyTorch 2.0."""

    from torch.onnx import symbolic_helper

    def symbolic(g, query, key, value, attention_mask, dropout_p, is_causal, scale=None):
        scalar_type = query.type().scalarType()
        onnx_dtype = 10 if scalar_type == "Half" else 1  # FLOAT16 / FLOAT
        shape = g.op("Shape", query)
        head_dim = g.op(
            "Gather",
            shape,
            g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
            axis_i=0,
        )
        head_dim = g.op("Cast", head_dim, to_i=onnx_dtype)
        factor = g.op("Reciprocal", g.op("Sqrt", head_dim))
        key = g.op("Transpose", key, perm_i=[0, 1, 3, 2])
        scores = g.op("MatMul", query, key)
        scores = g.op("Mul", scores, factor)
        if not symbolic_helper._is_none(attention_mask):
            scores = g.op("Add", scores, attention_mask)
        probabilities = g.op("Softmax", scores, axis_i=-1)
        return g.op("MatMul", probabilities, value)

    torch.onnx.register_custom_op_symbolic(
        "aten::scaled_dot_product_attention", symbolic, opset
    )


class UNetExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        audio_prompt: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            latent, timestep, encoder_hidden_states=audio_prompt, return_dict=False
        )[0]


class VAEEncoderExport(nn.Module):
    """Return distribution moments; sampling deliberately remains outside ONNX."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(image))


class VAEDecoderExport(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.scaling_factor = float(scaling_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = self.post_quant_conv(latent / self.scaling_factor)
        return self.decoder(latent)


class WhisperEncoderExport(nn.Module):
    """MuseTalk consumes the embedding output plus all four encoder layers."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        states = self.encoder(
            input_features, output_hidden_states=True, return_dict=True
        ).hidden_states
        return torch.stack(states, dim=2)


def _export(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_name: str,
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {path.name} ...")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with torch.inference_mode():
            torch.onnx.export(
                module,
                inputs,
                str(temporary),
                input_names=input_names,
                output_names=[output_name],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                # PyTorch 2.0 can inflate the fixed-batch UNet beyond protobuf's
                # 2 GiB limit when folding constants. ORT performs its own folding.
                do_constant_folding=False,
            )
        import onnx

        onnx.checker.check_model(str(temporary))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Validated {path}")


def export_unet(args: argparse.Namespace, dtype: torch.dtype) -> None:
    from diffusers.models.attention_processor import AttnProcessor2_0
    from musetalk.models.unet import UNet

    loaded = UNet(
        unet_config=str(args.unet_config),
        model_path=str(args.unet_weights),
        device=args.device,
    )
    model = loaded.model.eval().to(device=args.device, dtype=dtype)
    register_scaled_dot_product_attention_symbolic(args.opset)
    model.set_attn_processor(AttnProcessor2_0())
    wrapper = UNetExport(model).eval()
    batch = args.batch_size
    _export(
        wrapper,
        (
            torch.randn(batch, 8, 32, 32, device=args.device, dtype=dtype),
            torch.zeros(1, device=args.device, dtype=torch.int64),
            torch.randn(batch, 50, 384, device=args.device, dtype=dtype),
        ),
        args.output_dir / "musetalk_unet.onnx",
        ["latent", "timestep", "audio_prompt"],
        "latent_out",
        {},
        args.opset,
    )


def _load_vae(args: argparse.Namespace, dtype: torch.dtype):
    from diffusers import AutoencoderKL
    from diffusers.models.attention_processor import AttnProcessor

    vae = AutoencoderKL.from_pretrained(str(args.vae_dir))
    vae.set_attn_processor(AttnProcessor())
    return vae.eval().to(device=args.device, dtype=dtype)


def export_vae_encoder(args: argparse.Namespace, dtype: torch.dtype) -> None:
    vae = _load_vae(args, dtype)
    _export(
        VAEEncoderExport(vae).eval(),
        (torch.randn(args.batch_size, 3, 256, 256, device=args.device, dtype=dtype),),
        args.output_dir / "musetalk_vae_encoder.onnx",
        ["image"],
        "moments",
        {},
        args.opset,
    )


def export_vae_decoder(args: argparse.Namespace, dtype: torch.dtype) -> None:
    vae = _load_vae(args, dtype)
    _export(
        VAEDecoderExport(vae, vae.config.scaling_factor).eval(),
        (torch.randn(args.batch_size, 4, 32, 32, device=args.device, dtype=dtype),),
        args.output_dir / "musetalk_vae_decoder.onnx",
        ["latent"],
        "image",
        {},
        args.opset,
    )


def export_whisper(args: argparse.Namespace, dtype: torch.dtype) -> None:
    from transformers import WhisperModel

    model = WhisperModel.from_pretrained(
        str(args.whisper_dir), attn_implementation="eager"
    ).encoder
    model = model.eval().to(device=args.device, dtype=dtype)
    _export(
        WhisperEncoderExport(model).eval(),
        (torch.randn(args.whisper_batch_size, 80, 3000, device=args.device, dtype=dtype),),
        args.output_dir / "musetalk_whisper_encoder.onnx",
        ["input_features"],
        "hidden_states",
        {"input_features": {0: "batch"}, "hidden_states": {0: "batch"}},
        args.opset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", "unet", "vae_encoder", "vae_decoder", "whisper"),
        default=["all"],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/onnx")
    parser.add_argument("--unet-config", type=Path, default=ROOT / "models/musetalkV15/musetalk.json")
    parser.add_argument("--unet-weights", type=Path, default=ROOT / "models/musetalkV15/unet.pth")
    parser.add_argument("--vae-dir", type=Path, default=ROOT / "models/sd-vae")
    parser.add_argument("--whisper-dir", type=Path, default=ROOT / "models/whisper")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--whisper-batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--float32", action="store_true", help="Export FP32 instead of FP16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    dtype = torch.float32 if args.float32 else torch.float16
    selected = {"unet", "vae_encoder", "vae_decoder", "whisper"}
    if "all" not in args.models:
        selected = set(args.models)
    exporters = {
        "unet": export_unet,
        "vae_encoder": export_vae_encoder,
        "vae_decoder": export_vae_decoder,
        "whisper": export_whisper,
    }
    args.output_dir = args.output_dir.resolve()
    for name, exporter in exporters.items():
        if name in selected:
            exporter(args, dtype)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    artifacts = {
        "unet": "musetalk_unet.onnx",
        "vae_encoder": "musetalk_vae_encoder.onnx",
        "vae_decoder": "musetalk_vae_decoder.onnx",
        "whisper": "musetalk_whisper_encoder.onnx",
    }
    metadata = {
        "format": 1,
        "opset": args.opset,
        "precision": "fp32" if args.float32 else "fp16",
        "unet_batch_size": args.batch_size,
        "vae_batch_size": args.batch_size,
        "whisper_batch_size": args.whisper_batch_size,
        "models": sorted(
            name
            for name, filename in artifacts.items()
            if (args.output_dir / filename).is_file()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
