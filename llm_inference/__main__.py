"""CLI entry point: ``python -m llm_inference '你的问题'``."""

from __future__ import annotations

import argparse
import json
import sys

from .client import LLMInferenceError, LiteLLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 LiteLLM 生成简洁、通俗的回答文本")
    parser.add_argument("question", nargs="*", help="需要回答的问题；不填写时进入交互输入")
    parser.add_argument("--model", help="临时覆盖 .env 中的 LITELLM_MODEL")
    parser.add_argument("--list-models", action="store_true", help="列出 LiteLLM 支持的模型")
    args = parser.parse_args()

    client = LiteLLMClient()
    try:
        if args.list_models:
            for model in client.get_models():
                print(model)
            return
        question = " ".join(args.question).strip()
        if not question:
            question = input("请输入问题：").strip()
        print(
            json.dumps(
                client.answer(question, model=args.model),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    except (ValueError, LLMInferenceError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
