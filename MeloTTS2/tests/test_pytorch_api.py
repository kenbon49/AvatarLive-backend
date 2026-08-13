import inspect
import unittest

from melo import TTS
from pydantic import ValidationError
from server import SynthesisRequest


class PyTorchApiTests(unittest.TestCase):
    def test_tts_has_no_inference_backend_switch(self):
        parameters = inspect.signature(TTS).parameters

        self.assertNotIn("bert_backend", parameters)
        self.assertNotIn("onnx_dir", parameters)

    def test_service_rejects_onnx_backend_requests(self):
        with self.assertRaises(ValidationError):
            SynthesisRequest(text="backend check", bert_backend="onnx")


if __name__ == "__main__":
    unittest.main()
