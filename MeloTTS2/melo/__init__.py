from .api import TTS

__all__ = ["TTS", "OnnxBert", "OnnxTTS"]


def __getattr__(name):
    if name == "OnnxBert":
        from .onnx_api import OnnxBert

        return OnnxBert
    if name == "OnnxTTS":
        from .onnx_api import OnnxTTS

        return OnnxTTS
    raise AttributeError(name)
