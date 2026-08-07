"""Command-line inference with exported MeloTTS ONNX models."""

import argparse

from melo.onnx_api import OnnxTTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("ZH", "EN"), required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sdp-ratio", type=float, default=0.2)
    parser.add_argument("--noise-scale", type=float, default=0.6)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = OnnxTTS(
        args.language,
        onnx_dir=args.onnx_dir,
        device=args.device,
        seed=args.seed,
    )
    speakers = vars(model.hps.data.spk2id)
    speaker_name = args.speaker or next(iter(speakers))
    if speaker_name not in speakers:
        raise ValueError(
            f"Unknown speaker {speaker_name!r}; available: {', '.join(speakers)}"
        )
    model.tts_to_file(
        args.text,
        speakers[speaker_name],
        args.output,
        speed=args.speed,
        sdp_ratio=args.sdp_ratio,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        quiet=True,
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
