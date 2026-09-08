#!/usr/bin/env bash
set -euo pipefail

endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
root="models"

download_file() {
  local repo="$1"
  local source="$2"
  local destination="$3"
  local destination_dir
  local destination_name

  destination_dir="$(dirname "$destination")"
  destination_name="$(basename "$destination")"
  mkdir -p "$destination_dir"

  if command -v aria2c >/dev/null; then
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=4M \
      --file-allocation=none \
      --auto-file-renaming=false \
      --allow-overwrite=true \
      --max-tries=0 \
      --retry-wait=5 \
      --timeout=120 \
      --connect-timeout=30 \
      --console-log-level=warn \
      --show-console-readout=false \
      --summary-interval=30 \
      --dir="$destination_dir" \
      --out="$destination_name" \
      "$endpoint/$repo/resolve/main/$source?download=true"
    return
  fi

  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 100 \
    --retry-all-errors \
    --retry-delay 5 \
    --output "$destination" \
    "$endpoint/$repo/resolve/main/$source?download=true"
}

download_group() {
  while (( "$#" )); do
    download_file "$1" "$2" "$3"
    shift 3
  done
}

downloads=(
  TMElyralab/MuseTalk musetalkV15/musetalk.json "$root/musetalkV15/musetalk.json"
  stabilityai/sd-vae-ft-mse config.json "$root/sd-vae/config.json"
  stabilityai/sd-vae-ft-mse diffusion_pytorch_model.bin "$root/sd-vae/diffusion_pytorch_model.bin"
  openai/whisper-tiny config.json "$root/whisper/config.json"
  openai/whisper-tiny pytorch_model.bin "$root/whisper/pytorch_model.bin"
  openai/whisper-tiny preprocessor_config.json "$root/whisper/preprocessor_config.json"
  yzd-v/DWPose dw-ll_ucoco_384.pth "$root/dwpose/dw-ll_ucoco_384.pth"
)
if [[ "${MUSETALK_SKIP_UNET:-0}" != "1" ]]; then
  downloads+=(
    TMElyralab/MuseTalk musetalkV15/unet.pth "$root/musetalkV15/unet.pth"
  )
fi
download_group "${downloads[@]}"

if [[ ! -s "$root/face-parse-bisent/79999_iter.pth" ]]; then
  command -v gdown >/dev/null || { echo "gdown is required" >&2; exit 1; }
  mkdir -p "$root/face-parse-bisent"
  gdown --continue 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
    -O "$root/face-parse-bisent/79999_iter.pth"
fi

if [[ ! -s "$root/face-parse-bisent/resnet18-5c106cde.pth" ]]; then
  curl \
    --fail \
    --location \
    --retry 100 \
    --retry-all-errors \
    --retry-delay 5 \
    --output "$root/face-parse-bisent/resnet18-5c106cde.pth" \
    https://download.pytorch.org/models/resnet18-5c106cde.pth
fi

echo "All MuseTalk runtime weights have been downloaded successfully."
