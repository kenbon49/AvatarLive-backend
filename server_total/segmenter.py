"""Incrementally split LLM text at speech-relevant punctuation."""

from __future__ import annotations

from dataclasses import dataclass


HARD_DELIMITERS = frozenset("。！？.!?…")
SOFT_DELIMITERS = frozenset("，、,；;：:—")
DELIMITERS = HARD_DELIMITERS | SOFT_DELIMITERS
CLOSING_MARKS = frozenset("”’）)]】》〉」』")


@dataclass(frozen=True)
class TextUnit:
    seq: int
    text: str
    delimiter: str


class PunctuationSegmenter:
    """Split immediately at sentence endings and coalesce short comma clauses."""

    def __init__(self, *, first_unit_min_chars: int = 12, target_unit_chars: int = 20) -> None:
        if first_unit_min_chars < 1 or target_unit_chars < 1:
            raise ValueError("segment lengths must be positive")
        self._buffer = ""
        self._next_seq = 0
        self.first_unit_min_chars = int(first_unit_min_chars)
        self.target_unit_chars = int(target_unit_chars)

    def feed(self, delta: str) -> list[TextUnit]:
        if delta:
            self._buffer += delta
        return self._drain(final=False)

    def finish(self) -> list[TextUnit]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[TextUnit]:
        units: list[TextUnit] = []
        while self._buffer:
            boundary = self._find_boundary(final=final)
            if boundary is None:
                break
            punctuation_start, end = boundary
            raw_text = self._buffer[:end]
            self._buffer = self._buffer[end:].lstrip()
            text = raw_text.strip()
            if not text:
                continue
            delimiter = raw_text[punctuation_start:end].strip()
            units.append(TextUnit(self._next_seq, text, delimiter))
            self._next_seq += 1

        if final:
            text = self._buffer.strip()
            self._buffer = ""
            if text:
                units.append(TextUnit(self._next_seq, text, ""))
                self._next_seq += 1
        return units

    def _find_boundary(self, *, final: bool) -> tuple[int, int] | None:
        for index, char in enumerate(self._buffer):
            if char not in DELIMITERS or self._is_protected(index):
                continue
            end = index + 1
            while end < len(self._buffer):
                candidate = self._buffer[end]
                if candidate in DELIMITERS and not self._is_protected(end):
                    end += 1
                    continue
                if candidate in CLOSING_MARKS:
                    end += 1
                    continue
                break
            if end == len(self._buffer) and not final:
                return None
            if char in SOFT_DELIMITERS:
                minimum = (
                    self.first_unit_min_chars if self._next_seq == 0 else self.target_unit_chars
                )
                if self._spoken_length(self._buffer[:index]) < minimum:
                    if final and end == len(self._buffer):
                        return index, end
                    continue
            return index, end
        return None

    @staticmethod
    def _spoken_length(text: str) -> int:
        return sum(
            1
            for char in text
            if not char.isspace() and char not in DELIMITERS and char not in CLOSING_MARKS
        )

    def _is_protected(self, index: int) -> bool:
        char = self._buffer[index]
        previous = self._buffer[index - 1] if index > 0 else ""
        following = self._buffer[index + 1] if index + 1 < len(self._buffer) else ""

        if char in ".," and previous.isdigit() and following.isdigit():
            return True
        if char == ":" and previous.isdigit() and following.isdigit():
            return True
        if char == ":" and following == "/" and self._buffer[index + 1 :].startswith("//"):
            return True
        if char == "." and previous.isalnum() and following.isalnum():
            return True
        if char == "." and previous == "@":
            return True
        return False
