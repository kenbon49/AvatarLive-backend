#!/usr/bin/env bash
set -euo pipefail

endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
download_english="${OPENVOICE_DOWNLOAD_ENGLISH:-1}"
root="OpenVoice/checkpoints_v2"

download_file() {
  local repo="$1"
  local source="$2"
  local destination="$3"
  download_url "$endpoint/$repo/resolve/main/$source?download=true" "$destination"
}

download_url() {
  local url="$1"
  local destination="$2"
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
      "$url"
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
    "$url"
}

download_group() {
  while (( "$#" )); do
    download_file "$1" "$2" "$3"
    shift 3
  done
}

# The service preloads the Chinese path at startup. Download it first so the
# container can start while optional English assets are fetched separately.
download_group \
  myshell-ai/OpenVoiceV2 converter/config.json "$root/converter/config.json" \
  myshell-ai/OpenVoiceV2 converter/checkpoint.pth "$root/converter/checkpoint.pth" \
  myshell-ai/OpenVoiceV2 base_speakers/ses/zh.pth "$root/base_speakers/ses/zh.pth" \
  myshell-ai/MeloTTS-Chinese config.json "$root/melotts/pretrained_model/ZH/config.json" \
  myshell-ai/MeloTTS-Chinese checkpoint.pth "$root/melotts/pretrained_model/ZH/checkpoint.pth" \
  google-bert/bert-base-multilingual-uncased config.json "$root/melotts/bert_model/multilingual/config.json" \
  google-bert/bert-base-multilingual-uncased model.safetensors "$root/melotts/bert_model/multilingual/model.safetensors" \
  google-bert/bert-base-multilingual-uncased tokenizer.json "$root/melotts/bert_model/multilingual/tokenizer.json" \
  google-bert/bert-base-multilingual-uncased tokenizer_config.json "$root/melotts/bert_model/multilingual/tokenizer_config.json" \
  google-bert/bert-base-multilingual-uncased vocab.txt "$root/melotts/bert_model/multilingual/vocab.txt" \
  microsoft/deberta-v3-large config.json "$root/melotts/bert_model/english_bert/config.json" \
  microsoft/deberta-v3-large spm.model "$root/melotts/bert_model/english_bert/spm.model" \
  microsoft/deberta-v3-large tokenizer_config.json "$root/melotts/bert_model/english_bert/tokenizer_config.json"

vad_root="OpenVoice/checkpoints/auxiliary"
download_url \
  "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/v1.1.1/faster_whisper/assets/silero_encoder_v5.onnx" \
  "$vad_root/silero_encoder_v5.onnx"
download_url \
  "https://raw.githubusercontent.com/SYSTRAN/faster-whisper/v1.1.1/faster_whisper/assets/silero_decoder_v5.onnx" \
  "$vad_root/silero_decoder_v5.onnx"

echo "OpenVoice Chinese runtime weights have been downloaded successfully."

if [[ "$download_english" != "1" ]]; then
  exit 0
fi

download_group \
  myshell-ai/OpenVoiceV2 base_speakers/ses/en-default.pth "$root/base_speakers/ses/en-default.pth" \
  myshell-ai/OpenVoiceV2 base_speakers/ses/en-us.pth "$root/base_speakers/ses/en-us.pth" \
  myshell-ai/OpenVoiceV2 base_speakers/ses/en-br.pth "$root/base_speakers/ses/en-br.pth" \
  myshell-ai/OpenVoiceV2 base_speakers/ses/en-india.pth "$root/base_speakers/ses/en-india.pth" \
  myshell-ai/OpenVoiceV2 base_speakers/ses/en-au.pth "$root/base_speakers/ses/en-au.pth" \
  myshell-ai/MeloTTS-English config.json "$root/melotts/pretrained_model/EN/config.json" \
  myshell-ai/MeloTTS-English checkpoint.pth "$root/melotts/pretrained_model/EN/checkpoint.pth" \
  microsoft/deberta-v3-large pytorch_model.bin "$root/melotts/bert_model/english_bert/pytorch_model.bin"

echo "All OpenVoice weights have been downloaded successfully."
