import unittest

from melo.onnx_api import _cuda_provider_options


class CudaProviderOptionsTests(unittest.TestCase):
    def test_options_are_supported_by_pinned_onnxruntime(self):
        options = _cuda_provider_options(2)

        self.assertEqual(
            options,
            {
                "device_id": "2",
                "do_copy_in_default_stream": "1",
            },
        )
        self.assertNotIn("user_compute_stream", options)


if __name__ == "__main__":
    unittest.main()
