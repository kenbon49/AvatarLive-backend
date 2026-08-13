import argparse
from pathlib import Path

from melo import TTS


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Chinese/English MeloTTS inference")
    parser.add_argument("--text", required=True, help=" Are you a broke beauty like me that loves to gamble, but doesn't have this money? Then you have to download cash bash. This slot game has a super high payout.")
    parser.add_argument("--language", default="ZH", choices=["ZH", "EN"], help="Language to use")
    parser.add_argument("--speaker", help="Speaker name; defaults to the first available speaker")
    parser.add_argument("--output", type=Path, default=Path(r"D:\code\avatar\MuseTalk\data\input\audio\melo_demo.wav"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    model = TTS(args.language, device=args.device)
    speakers = model.hps.data.spk2id
    speaker = args.speaker or next(iter(speakers))
    if speaker not in speakers:
        raise ValueError(f"Unknown speaker {speaker!r}; choose from {list(speakers)}")
    model.tts_to_file(args.text, speakers[speaker], args.output, speed=args.speed)
    print(f"Saved {args.language}/{speaker} audio to {args.output.resolve()}")


if __name__ == "__main__":
    main()
