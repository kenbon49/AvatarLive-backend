"""Download Linux wheels used by the CosyVoice Docker image."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


WHEELS = {
    "onnx==1.16.0": "onnx-1.16.0-*.whl",
    "onnxruntime-gpu==1.18.0": "onnxruntime_gpu-1.18.0-*.whl",
}
PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
ORT_CUDA_INDEX = (
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/"
    "_packaging/onnxruntime-cuda-12/pypi/simple"
)


def is_complete_wheel(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Linux ONNX wheels used by the CosyVoice Docker image."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "wheels",
        help="Directory that receives the wheels (default: CosyVoice/wheels).",
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    packages = []
    for package, pattern in WHEELS.items():
        valid_files = [path for path in destination.glob(pattern) if is_complete_wheel(path)]
        if valid_files:
            print(f"Already downloaded: {valid_files[0].name}")
        else:
            packages.append(package)

    if not packages:
        print("All ONNX wheels are already complete.")
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--only-binary=:all:",
        "--platform",
        "manylinux_2_28_x86_64",
        "--platform",
        "manylinux2014_x86_64",
        "--python-version",
        "310",
        "--implementation",
        "cp",
        "--abi",
        "cp310",
        "--index-url",
        PYPI_INDEX,
        "--extra-index-url",
        ORT_CUDA_INDEX,
        "--progress-bar",
        "on",
        "--dest",
        str(destination),
        *packages,
    ]
    print("Downloading to:", destination)
    subprocess.run(command, check=True)

    incomplete = [
        package
        for package, pattern in WHEELS.items()
        if not any(is_complete_wheel(path) for path in destination.glob(pattern))
    ]
    if incomplete:
        raise RuntimeError("Incomplete downloads: " + ", ".join(incomplete))

    print("ONNX wheel downloads complete.")


if __name__ == "__main__":
    main()
