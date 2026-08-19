#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cosyvoice_dir="$(cd -- "${script_dir}/.." && pwd)"

docker run --rm --gpus all \
  --volume "${cosyvoice_dir}:/app" \
  cosyvoice:dev \
  python /app/scripts/build_trt_plan.py "$@"
