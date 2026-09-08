"""Start the configured local TTS provider."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Local TTS service launcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()
    provider = os.getenv("TTS_SERVICE", "melotts").strip().lower()
    applications = {
        "melotts": "melotts_server:app",
        "openvoice": "server:app",
    }
    application = applications.get(provider)
    if application is None:
        choices = ", ".join(sorted(applications))
        raise SystemExit(f"unsupported TTS_SERVICE={provider!r}; choose one of: {choices}")
    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
