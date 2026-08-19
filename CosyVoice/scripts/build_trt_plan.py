#!/usr/bin/env python3
"""Prebuild and validate the CosyVoice Flow TensorRT plan."""

import argparse
from pathlib import Path

import tensorrt as trt

from cosyvoice.utils.file_utils import convert_onnx_to_trt


TRT_SHAPES = {
    "min_shape": [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)],
    "opt_shape": [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)],
    "max_shape": [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)],
    "input_names": ["x", "mask", "mu", "cond"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/app/pretrained_models/CosyVoice2-0.5B"),
        help="model directory mounted inside the container",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="engine precision; must match the API server's --fp16 setting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the existing plan is valid",
    )
    return parser.parse_args()


def deserialize_plan(plan_path: Path):
    logger = trt.Logger(trt.Logger.WARNING)
    with plan_path.open("rb") as plan_file:
        return trt.Runtime(logger).deserialize_cuda_engine(plan_file.read())


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    onnx_path = model_dir / "flow.decoder.estimator.fp32.onnx"
    plan_path = model_dir / f"flow.decoder.estimator.{args.precision}.mygpu.plan"

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

    if plan_path.is_file() and plan_path.stat().st_size > 0 and not args.force:
        if deserialize_plan(plan_path) is not None:
            print(f"TensorRT plan is already valid: {plan_path}")
            return
        print(f"Existing TensorRT plan is invalid or incompatible; rebuilding: {plan_path}")

    print(f"Building {args.precision} TensorRT plan from {onnx_path}")
    convert_onnx_to_trt(
        str(plan_path),
        TRT_SHAPES,
        str(onnx_path),
        fp16=args.precision == "fp16",
    )
    if not plan_path.is_file() or plan_path.stat().st_size == 0:
        raise RuntimeError(f"TensorRT did not create a plan: {plan_path}")
    if deserialize_plan(plan_path) is None:
        raise RuntimeError(f"TensorRT created an unreadable plan: {plan_path}")

    print(f"TensorRT plan is ready: {plan_path} ({plan_path.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
