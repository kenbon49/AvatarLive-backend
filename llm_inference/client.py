"""LiteLLM/OpenAI-compatible client for concise question answering."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence, TypedDict

from openai import OpenAI

from .prompts import CONCISE_ANSWER_SYSTEM_PROMPT


class LLMInferenceError(RuntimeError):
    """Raised when the configured model cannot produce an answer."""


class AnswerPayload(TypedDict):
    """Stable structured output returned to downstream services."""

    answer: str


def load_env(path: str | Path | None = None) -> None:
    """Load a small .env file without replacing explicit process variables."""

    env_path = Path(path) if path is not None else Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _api_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("LITELLM_BASE_URL must not be empty")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:4000"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 256
    json_mode: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_env()
        return cls(
            base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
            api_key=os.getenv("LITELLM_API_KEY", ""),
            model=os.getenv("LITELLM_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LITELLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LITELLM_MAX_TOKENS", "256")),
            json_mode=os.getenv("LITELLM_JSON_MODE", "true").lower()
            not in {"0", "false", "no", "off"},
        )


def _parse_answer(content: str) -> AnswerPayload:
    """Normalize model output to ``{"answer": "..."}``."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = {"answer": cleaned}
    if not isinstance(payload, dict):
        raise LLMInferenceError("model JSON output must be an object")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise LLMInferenceError("model JSON output has no non-empty answer field")
    return {"answer": answer.strip()}


def _partial_json_answer(content: str) -> str | None:
    """Decode the complete prefix of an unfinished JSON ``answer`` string."""

    marker = '"answer"'
    marker_start = content.find(marker)
    if marker_start < 0:
        return None
    colon = content.find(":", marker_start + len(marker))
    if colon < 0:
        return None
    quote = colon + 1
    while quote < len(content) and content[quote].isspace():
        quote += 1
    if quote >= len(content) or content[quote] != '"':
        return None

    encoded = content[quote + 1 :]
    safe_end = 0
    index = 0
    while index < len(encoded):
        char = encoded[index]
        if char == '"':
            safe_end = index
            break
        if char != "\\":
            index += 1
            safe_end = index
            continue
        if index + 1 >= len(encoded):
            break
        escape = encoded[index + 1]
        if escape == "u":
            candidate = encoded[index + 2 : index + 6]
            if len(candidate) < 4 or any(
                char not in "0123456789abcdefABCDEF" for char in candidate
            ):
                break
            index += 6
        elif escape in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            index += 2
        else:
            break
        safe_end = index

    try:
        return json.loads('"' + encoded[:safe_end] + '"')
    except json.JSONDecodeError as exc:
        raise LLMInferenceError(f"could not decode streaming answer: {exc}") from exc


class LiteLLMClient:
    """Return concise answer text from a LiteLLM proxy."""

    def __init__(self, config: LLMConfig | None = None, *, openai_client: Any = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.client = openai_client or OpenAI(
            api_key=self.config.api_key or "not-required",
            base_url=_api_base_url(self.config.base_url),
        )

    def answer(
        self,
        question: str,
        *,
        model: str | None = None,
        system_prompt: str = CONCISE_ANSWER_SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> AnswerPayload:
        """Answer one user question as ``{"answer": "..."}``."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be empty")
        options = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }
        if self.config.json_mode and "response_format" not in options:
            options["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=model or self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": clean_question},
                ],
                **options,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise LLMInferenceError("model returned an empty answer")
            return _parse_answer(content)
        except LLMInferenceError:
            raise
        except Exception as exc:
            raise LLMInferenceError(f"LiteLLM request failed: {exc}") from exc

    def answer_text(self, question: str, **kwargs: Any) -> str:
        """Return the validated answer text for direct TTS input."""

        return self.answer(question, **kwargs)["answer"]

    def stream_answer_text(
        self,
        question: str,
        *,
        model: str | None = None,
        system_prompt: str = CONCISE_ANSWER_SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield answer-only text deltas from an OpenAI-compatible stream."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be empty")
        options = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }
        if self.config.json_mode and "response_format" not in options:
            options["response_format"] = {"type": "json_object"}
        structured = options.get("response_format") == {"type": "json_object"}
        try:
            stream = self.client.chat.completions.create(
                model=model or self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": clean_question},
                ],
                stream=True,
                **options,
            )
            raw_content = ""
            emitted = ""
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0].delta, "content", None)
                if not isinstance(delta, str) or not delta:
                    continue
                raw_content += delta
                partial = _partial_json_answer(raw_content) if structured else raw_content
                if partial is None:
                    continue
                if not partial.startswith(emitted):
                    raise LLMInferenceError("streaming answer changed already emitted text")
                new_text = partial[len(emitted) :]
                if new_text:
                    emitted = partial
                    yield new_text

            answer = _parse_answer(raw_content)["answer"]
            if not answer.startswith(emitted):
                raise LLMInferenceError("final answer differs from streamed text")
            remainder = answer[len(emitted) :]
            if remainder:
                yield remainder
        except (LLMInferenceError, ValueError):
            raise
        except Exception as exc:
            raise LLMInferenceError(f"LiteLLM streaming request failed: {exc}") from exc

    def answer_with_history(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> AnswerPayload:
        """Answer using existing dialogue history and the same concise prompt."""

        request_messages = [
            {"role": "system", "content": CONCISE_ANSWER_SYSTEM_PROMPT},
            *messages,
        ]
        options = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs,
        }
        if self.config.json_mode and "response_format" not in options:
            options["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=model or self.config.model,
                messages=request_messages,
                **options,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise LLMInferenceError("model returned an empty answer")
            return _parse_answer(content)
        except LLMInferenceError:
            raise
        except Exception as exc:
            raise LLMInferenceError(f"LiteLLM request failed: {exc}") from exc

    def get_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self.client.models.list().data)
        except Exception as exc:
            raise LLMInferenceError(f"could not list LiteLLM models: {exc}") from exc


def answer_question(question: str, **kwargs: Any) -> AnswerPayload:
    """Convenience entry point returning a structured answer object."""

    return LiteLLMClient().answer(question, **kwargs)


def answer_question_text(question: str, **kwargs: Any) -> str:
    """Return only the answer value for direct TTS integration."""

    return LiteLLMClient().answer_text(question, **kwargs)
