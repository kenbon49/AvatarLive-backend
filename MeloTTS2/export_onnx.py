"""Export the local English/Chinese MeloTTS inference models to ONNX."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from torch import nn

# BERT export is text-only. Ignore an installed but ABI-incompatible torchvision
# instead of letting a Transformers optional vision import block model loading.
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils

transformers_utils.is_torchvision_available = lambda: False
transformers_import_utils.is_torchvision_available = lambda: False

from transformers import AutoModel

from melo.models import SynthesizerTrn
from melo.utils import get_hparams_from_file


ROOT = Path(__file__).resolve().parent
BERT_MODELS = {
    "EN": ("english_bert", ROOT / "bert_model" / "english_bert"),
    "ZH": ("multilingual_bert", ROOT / "bert_model" / "multilingual"),
}


class BertFeatureExport(nn.Module):
    """Return the same third-from-last hidden state used by MeloTTS."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return output.hidden_states[-3]


class AcousticEncoderExport(nn.Module):
    """Export text encoding plus both duration predictors.

    MeloTTS creates random duration noise inside the stochastic predictor. Making
    it an input keeps the ONNX graph deterministic and permits seeded inference.
    """

    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def _stochastic_duration(self, x, x_mask, g, duration_noise, noise_scale_w):
        predictor = self.model.sdp
        hidden = torch.detach(x)
        hidden = predictor.pre(hidden)
        hidden = hidden + predictor.cond(torch.detach(g))
        hidden = predictor.convs(hidden, x_mask)
        hidden = predictor.proj(hidden) * x_mask

        flows = list(reversed(predictor.flows))
        flows = flows[:-2] + [flows[-1]]
        z = duration_noise * noise_scale_w
        for flow in flows:
            z = flow(z, x_mask, g=hidden, reverse=True)
        return torch.split(z, [1, 1], dim=1)[0]

    def forward(
        self,
        phones,
        lengths,
        speaker_ids,
        tones,
        language_ids,
        bert,
        ja_bert,
        duration_noise,
        noise_scale_w,
    ):
        g = self.model.emb_g(speaker_ids).unsqueeze(-1)
        g_p = None if self.model.use_vc else g
        x, m_p, logs_p, x_mask = self.model.enc_p(
            phones, lengths, tones, language_ids, bert, ja_bert, g=g_p
        )
        logw_sdp = self._stochastic_duration(
            x, x_mask, g, duration_noise, noise_scale_w
        )
        logw_dp = self.model.dp(x, x_mask, g=g)
        return m_p, logs_p, x_mask, logw_sdp, logw_dp, g


class FlowDecoderExport(nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.flow = model.flow

    def forward(self, z_p, y_mask, g):
        return self.flow(z_p, y_mask, g=g, reverse=True)


class WaveformGeneratorExport(nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.generator = model.dec

    def forward(self, z, g):
        return self.generator(z, g=g)


def _load_acoustic_model(language: str, device: torch.device):
    model_dir = ROOT / "pretrained_model" / language
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "checkpoint.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing MeloTTS model under {model_dir}")

    hps = get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, hps


def _export(module, inputs, path, input_names, output_names, dynamic_axes, opset):
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.inference_mode():
        export_options = dict(
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
        # Torch 2.9+ defaults to the dynamo exporter, which adds an onnxscript
        # dependency. The legacy path is stable for these fixed PyTorch modules.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_options["dynamo"] = False
        torch.onnx.export(
            module,
            inputs,
            path,
            **export_options,
        )
    import onnx

    onnx.checker.check_model(onnx.load(path))
    print(f"exported: {path}")


def export_bert(language: str, output_dir: Path, device: torch.device, opset: int):
    output_name, model_dir = BERT_MODELS[language]
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    wrapper = BertFeatureExport(model)
    sequence = 16
    inputs = (
        torch.zeros((1, sequence), dtype=torch.long, device=device),
        torch.ones((1, sequence), dtype=torch.long, device=device),
        torch.zeros((1, sequence), dtype=torch.long, device=device),
    )
    axes = {
        name: {0: "batch", 1: "tokens"}
        for name in ("input_ids", "attention_mask", "token_type_ids", "features")
    }
    path = output_dir / "bert" / f"{output_name}.onnx"
    _export(
        wrapper,
        inputs,
        path,
        ["input_ids", "attention_mask", "token_type_ids"],
        ["features"],
        axes,
        opset,
    )
    return path


def export_acoustic(language: str, output_dir: Path, device: torch.device, opset: int):
    model, hps = _load_acoustic_model(language, device)
    target = output_dir / language
    text_length = 16
    latent_length = 32
    batch = 1
    dtype = next(model.parameters()).dtype

    encoder_inputs = (
        torch.zeros((batch, text_length), dtype=torch.long, device=device),
        torch.tensor([text_length], dtype=torch.long, device=device),
        torch.zeros((batch,), dtype=torch.long, device=device),
        torch.zeros((batch, text_length), dtype=torch.long, device=device),
        torch.zeros((batch, text_length), dtype=torch.long, device=device),
        torch.zeros((batch, 1024, text_length), dtype=dtype, device=device),
        torch.zeros((batch, 768, text_length), dtype=dtype, device=device),
        torch.randn((batch, 2, text_length), dtype=dtype, device=device),
        torch.tensor(0.8, dtype=dtype, device=device),
    )
    encoder_input_names = [
        "phones",
        "lengths",
        "speaker_ids",
        "tones",
        "language_ids",
        "bert",
        "ja_bert",
        "duration_noise",
        "noise_scale_w",
    ]
    encoder_output_names = ["m_p", "logs_p", "x_mask", "logw_sdp", "logw_dp", "g"]
    encoder_axes = {
        "phones": {0: "batch", 1: "text"},
        "lengths": {0: "batch"},
        "speaker_ids": {0: "batch"},
        "tones": {0: "batch", 1: "text"},
        "language_ids": {0: "batch", 1: "text"},
        "bert": {0: "batch", 2: "text"},
        "ja_bert": {0: "batch", 2: "text"},
        "duration_noise": {0: "batch", 2: "text"},
        "m_p": {0: "batch", 2: "text"},
        "logs_p": {0: "batch", 2: "text"},
        "x_mask": {0: "batch", 2: "text"},
        "logw_sdp": {0: "batch", 2: "text"},
        "logw_dp": {0: "batch", 2: "text"},
        "g": {0: "batch"},
    }
    _export(
        AcousticEncoderExport(model),
        encoder_inputs,
        target / "encoder_duration.onnx",
        encoder_input_names,
        encoder_output_names,
        encoder_axes,
        opset,
    )

    inter_channels = int(hps.model.inter_channels)
    gin_channels = int(hps.model.gin_channels)
    z_p = torch.randn((batch, inter_channels, latent_length), dtype=dtype, device=device)
    y_mask = torch.ones((batch, 1, latent_length), dtype=dtype, device=device)
    g = torch.randn((batch, gin_channels, 1), dtype=dtype, device=device)
    _export(
        FlowDecoderExport(model),
        (z_p, y_mask, g),
        target / "flow.onnx",
        ["z_p", "y_mask", "g"],
        ["z"],
        {
            "z_p": {0: "batch", 2: "frames"},
            "y_mask": {0: "batch", 2: "frames"},
            "g": {0: "batch"},
            "z": {0: "batch", 2: "frames"},
        },
        opset,
    )

    # Removing weight norm preserves eval output and produces a simpler ONNX graph.
    model.dec.remove_weight_norm()
    _export(
        WaveformGeneratorExport(model),
        (z_p, g),
        target / "generator.onnx",
        ["z", "g"],
        ["audio"],
        {
            "z": {0: "batch", 2: "frames"},
            "g": {0: "batch"},
            "audio": {0: "batch", 2: "samples"},
        },
        opset,
    )
    return {
        "encoder": target / "encoder_duration.onnx",
        "flow": target / "flow.onnx",
        "generator": target / "generator.onnx",
        "sampling_rate": int(hps.data.sampling_rate),
        "speakers": dict(hps.data.spk2id.items()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "onnx_models")
    parser.add_argument("--languages", nargs="+", choices=("ZH", "EN"), default=("ZH", "EN"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-acoustic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir = args.output_dir.resolve()
    manifest = {"opset": args.opset, "languages": {}}

    for language in dict.fromkeys(args.languages):
        language_info = {}
        if not args.skip_bert:
            language_info["bert"] = str(
                export_bert(language, output_dir, device, args.opset)
                .relative_to(output_dir)
                .as_posix()
            )
        if not args.skip_acoustic:
            acoustic = export_acoustic(language, output_dir, device, args.opset)
            language_info.update(
                {
                    "encoder": acoustic["encoder"].relative_to(output_dir).as_posix(),
                    "flow": acoustic["flow"].relative_to(output_dir).as_posix(),
                    "generator": acoustic["generator"].relative_to(output_dir).as_posix(),
                    "sampling_rate": acoustic["sampling_rate"],
                    "speakers": acoustic["speakers"],
                }
            )
        manifest["languages"][language] = language_info

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
