from __future__ import annotations

from types import SimpleNamespace
import unittest

from llm_inference.client import (
    LLMConfig,
    LLMInferenceError,
    LiteLLMClient,
    _api_base_url,
    _partial_json_answer,
    _parse_answer,
)
from llm_inference.prompts import CONCISE_ANSWER_SYSTEM_PROMPT


class FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content='{"answer":"这是一个简短回答。"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeStreamingCompletions(FakeCompletions):
    def create(self, **kwargs):
        self.request = kwargs
        parts = ['{"answer":"你', "好，", "世界", "。", '"}']
        return iter(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
            )
            for part in parts
        )


class LLMClientTests(unittest.TestCase):
    def test_base_url_adds_v1_once(self):
        self.assertEqual(_api_base_url("http://localhost:4000"), "http://localhost:4000/v1")
        self.assertEqual(_api_base_url("http://localhost:4000/v1/"), "http://localhost:4000/v1")

    def test_answer_uses_system_prompt_and_returns_clean_text(self):
        completions = FakeCompletions()
        fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client = LiteLLMClient(LLMConfig(model="test-model"), openai_client=fake)
        answer = client.answer("  什么是 AI？ ")
        self.assertEqual(answer, {"answer": "这是一个简短回答。"})
        self.assertEqual(completions.request["model"], "test-model")
        self.assertEqual(completions.request["messages"][0]["content"], CONCISE_ANSWER_SYSTEM_PROMPT)
        self.assertEqual(completions.request["messages"][1]["content"], "什么是 AI？")
        self.assertEqual(completions.request["response_format"], {"type": "json_object"})

    def test_parser_normalizes_markdown_and_plain_text(self):
        self.assertEqual(_parse_answer('```json\n{"answer":"你好"}\n```'), {"answer": "你好"})
        self.assertEqual(_parse_answer("普通文本"), {"answer": "普通文本"})

    def test_answer_text_returns_tts_ready_string(self):
        fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        client = LiteLLMClient(LLMConfig(), openai_client=fake)
        self.assertEqual(client.answer_text("介绍一下人工智能"), "这是一个简短回答。")

    def test_partial_json_answer_handles_chunked_escapes(self):
        self.assertEqual(_partial_json_answer('{"answer":"第一行\\n第'), "第一行\n第")
        self.assertEqual(_partial_json_answer('{"answer":"你好\\u4e'), "你好")
        self.assertEqual(_partial_json_answer('{"answer":"你好\\u4e16'), "你好世")

    def test_stream_answer_text_yields_only_answer_deltas(self):
        completions = FakeStreamingCompletions()
        fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client = LiteLLMClient(LLMConfig(model="stream-model"), openai_client=fake)

        deltas = list(client.stream_answer_text("打个招呼"))

        self.assertEqual(deltas, ["你", "好，", "世界", "。"])
        self.assertEqual("".join(deltas), "你好，世界。")
        self.assertTrue(completions.request["stream"])
        self.assertEqual(completions.request["response_format"], {"type": "json_object"})

    def test_parser_rejects_invalid_structured_answer(self):
        with self.assertRaisesRegex(LLMInferenceError, "non-empty answer"):
            _parse_answer('{"answer":"  "}')
        with self.assertRaisesRegex(LLMInferenceError, "must be an object"):
            _parse_answer('["不是对象"]')

    def test_empty_question_is_rejected(self):
        fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        client = LiteLLMClient(LLMConfig(), openai_client=fake)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            client.answer("   ")


if __name__ == "__main__":
    unittest.main()
