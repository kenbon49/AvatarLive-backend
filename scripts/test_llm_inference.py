"""Test a LiteLLM response and print its timing."""

from pathlib import Path
import sys
import time

# Allow this file to be run directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_inference import LLMInferenceError, LiteLLMClient  # noqa: E402

# Modify these two values to change the test.
TEST_QUESTION = "你叫什么名字"
STREAM_RESPONSE = True


def main() -> None:
    client = LiteLLMClient()

    print(f"问题：{TEST_QUESTION}")
    print(f"模型：{client.config.model}")
    print(f"流式回复：{'开启' if STREAM_RESPONSE else '关闭'}")

    started = time.perf_counter()
    if STREAM_RESPONSE:
        first_chunk_seconds = None
        chunk_count = 0
        print("回答：", end="", flush=True)
        for text in client.stream_answer_text(TEST_QUESTION):
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - started
            chunk_count += 1
            print(text, end="", flush=True)
        print()
        print(f"首字耗时：{first_chunk_seconds:.3f} 秒")
        print(f"分块数量：{chunk_count}")
    else:
        answer = client.answer_text(TEST_QUESTION)
        print(f"回答：{answer}")

    print(f"总耗时：{time.perf_counter() - started:.3f} 秒")
    print("测试通过。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, LLMInferenceError) as exc:
        print(f"\n测试失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
