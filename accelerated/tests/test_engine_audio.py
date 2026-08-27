from __future__ import annotations

from contextlib import nullcontext
import math
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from accelerated.engine import MuseTalkEngine


class FakeFeatureTensor:
    def to(self, **_kwargs):
        return self


class FakeChunkTensor:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("fake chunks only support slices")
        start, stop, step = key.indices(self.count)
        return FakeChunkTensor(len(range(start, stop, step)))

    def detach(self):
        return self

    def cpu(self):
        return self


class FakeAudioProcessor:
    def __init__(self, *, exit_during_chunking: bool = False) -> None:
        self.exit_during_chunking = exit_during_chunking
        self.feature_inputs: list[np.ndarray] = []
        self.chunk_audio_samples = 0

    def feature_extractor(self, samples, **_kwargs):
        self.feature_inputs.append(samples.copy())
        return SimpleNamespace(input_features=FakeFeatureTensor())

    def get_whisper_chunk(
        self,
        _features,
        _device,
        _dtype,
        _whisper,
        audio_samples,
        *,
        fps,
        **_kwargs,
    ):
        if self.exit_during_chunking:
            raise SystemExit
        self.chunk_audio_samples = audio_samples
        count = math.floor(audio_samples / 16000 * int(fps))
        return FakeChunkTensor(count)


def make_engine(processor: FakeAudioProcessor) -> MuseTalkEngine:
    engine = object.__new__(MuseTalkEngine)
    engine.audio_padding_left = 2
    engine.audio_padding_right = 2
    engine.weight_dtype = "float32"
    engine.device = "cuda:0"
    engine.whisper = object()
    engine.audio_processor = processor
    engine._lock = threading.RLock()
    engine.torch = SimpleNamespace(
        empty=lambda shape, **_kwargs: FakeChunkTensor(shape[0]),
        inference_mode=nullcontext,
    )
    return engine


class EngineAudioTests(unittest.TestCase):
    def test_short_single_frame_audio_is_padded_only_for_whisper(self):
        processor = FakeAudioProcessor()
        engine = make_engine(processor)
        pcm = np.ones(1066, dtype=np.float32)

        chunks = engine.extract_audio_features(pcm, 15)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(processor.chunk_audio_samples, 3200)
        self.assertEqual(len(processor.feature_inputs[0]), 3200)
        np.testing.assert_array_equal(processor.feature_inputs[0][: len(pcm)], pcm)
        self.assertFalse(np.any(processor.feature_inputs[0][len(pcm) :]))

    def test_non_aligned_audio_gets_a_final_covering_frame(self):
        processor = FakeAudioProcessor()
        engine = make_engine(processor)

        chunks = engine.extract_audio_features(
            np.zeros(8000, dtype=np.float32),
            15,
        )

        self.assertEqual(len(chunks), 8)
        self.assertEqual(processor.chunk_audio_samples, 8534)

    def test_upstream_system_exit_becomes_a_request_error(self):
        engine = make_engine(FakeAudioProcessor(exit_during_chunking=True))

        with self.assertRaisesRegex(RuntimeError, "feature extraction failed"):
            engine.extract_audio_features(np.zeros(1067, dtype=np.float32), 15)


if __name__ == "__main__":
    unittest.main()
