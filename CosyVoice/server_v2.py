# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Start the voice-cloning API with the local CosyVoice2 model."""

import sys
from pathlib import Path

from server import main


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_V2_MODEL = ROOT_DIR / "pretrained_models" / "CosyVoice2-0.5B"


def has_cli_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    if not has_cli_option("--model-dir"):
        sys.argv.extend(["--model-dir", str(DEFAULT_V2_MODEL)])
    main()
