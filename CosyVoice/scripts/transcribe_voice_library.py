"""Transcribe preset reference WAVs and update the voice manifest."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "Matcha-TTS"))

from voice_registry import MANIFEST_PATH, WhisperTranscriber  # noqa: E402


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    transcriber = WhisperTranscriber()
    for voice in manifest["voices"]:
        source = voice.get("source", {})
        if source.get("provider") != "Fish Audio":
            continue
        audio_path = (MANIFEST_PATH.parent / voice["prompt_wav"]).resolve()
        whisper_text = transcriber.transcribe(audio_path)
        voice["whisper_text"] = whisper_text
        print(f"{voice['voice_id']}: {whisper_text}")
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
