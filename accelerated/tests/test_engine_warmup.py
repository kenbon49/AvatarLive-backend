from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from accelerated.engine import MuseTalkEngine


class EngineWarmUpTests(unittest.TestCase):
    def test_warm_up_exercises_whisper_before_rendering(self):
        engine = object.__new__(MuseTalkEngine)
        engine.batch_size = 1
        engine.weight_dtype = "float32"
        fixed_prompts = object()
        engine.torch = SimpleNamespace(zeros=Mock(return_value=fixed_prompts))
        whisper_prompts = np.zeros((25, 50, 384), dtype=np.float32)
        engine.extract_audio_features = Mock(return_value=whisper_prompts)
        engine.render_batch = Mock()

        engine.warm_up(["frame"], iterations=2, fps=25.0)

        pcm, fps = engine.extract_audio_features.call_args.args
        self.assertEqual((pcm.shape, pcm.dtype, fps), ((16_000,), np.float32, 25.0))
        self.assertEqual(engine.render_batch.call_count, 3)
        first_prompts, first_frames = engine.render_batch.call_args_list[0].args
        self.assertIs(first_prompts.base, whisper_prompts)
        self.assertEqual(first_frames, ["frame"])
        self.assertIs(engine.render_batch.call_args_list[1].args[0], fixed_prompts)


if __name__ == "__main__":
    unittest.main()
