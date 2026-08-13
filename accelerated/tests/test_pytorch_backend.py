from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from accelerated.engine import MuseTalkEngine
from accelerated.runtime import MuseTalkRuntime, RuntimeConfig


class PyTorchBackendTests(unittest.TestCase):
    def test_engine_has_no_inference_backend_switch(self):
        parameters = inspect.signature(MuseTalkEngine).parameters

        self.assertNotIn("backend", parameters)
        self.assertNotIn("onnx_dir", parameters)

    def test_runtime_reports_the_fixed_torch_backend(self):
        runtime = MuseTalkRuntime(RuntimeConfig.defaults())

        self.assertEqual(runtime.status()["backend"], "torch")
        self.assertEqual(runtime.status()["inference_dtype"], "float32")

    def test_engine_uses_float32_inference(self):
        self.assertEqual(MuseTalkEngine.inference_dtype, "float32")
        self.assertNotIn("use_float16", inspect.signature(MuseTalkEngine).parameters)

    def test_low_level_musetalk_models_reject_float16(self):
        root = Path(__file__).resolve().parents[2]
        for relative, message in (
            ("musetalk/models/unet.py", "class UNet"),
            ("musetalk/models/vae.py", "class VAE"),
        ):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertIn(message, source)
            self.assertNotIn(".half()", source)
            self.assertNotIn("use_float16", source)


if __name__ == "__main__":
    unittest.main()
