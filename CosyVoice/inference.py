"""CosyVoice v1、v2 和 v3 的本地推理脚本。"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
import torchaudio


ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

MODEL_DIRS = {
    "v1": str(ROOT_DIR / "pretrained_models" / "CosyVoice-300M"),
    "v2": str(ROOT_DIR / "pretrained_models" / "CosyVoice2-0.5B"),
    "v3": str(ROOT_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
}
MODEL_CLASSES = {
    "v1": "CosyVoice",
    "v2": "CosyVoice2",
    "v3": "CosyVoice3",
}


@dataclass
class InferenceConfig:
    """统一管理所有推理参数；直接修改这里的默认值即可。"""

    model_version: str = "v3"  # 模型版本，可选 v1、v2 或 v3。
    mode: str = "zero_shot"  # 推理模式：zero_shot、cross_lingual、sft 或 instruct。
    text: str = "她走后，他每天对着空椅子说话，直到有一天，那椅子也不见了。"  # 待合成文本。
    prompt_wav: str = str(ROOT_DIR / "asset" / "zero_shot_prompt.wav")  # 目标音色参考音频。
    prompt_text: str = "希望你以后能够做得比我还好呦。"  # 参考音频的准确转录，zero_shot 必填。
    speaker: str | None = None  # SFT 说话人 ID；v1 的 sft 和 instruct 模式使用。
    instruct_text: str | None = None  # 情感、语速或方言等控制指令，instruct 模式必填。
    output: str = "output.wav"  # 生成音频的保存路径。
    speed: float = 1.0  # 语速倍率，必须大于 0；1.0 表示正常语速。
    stream: bool = False  # 是否启用流式音频生成。
    fp16: bool = False  # 是否启用 FP16 混合精度；模型权重仍以 FP32 常驻。
    load_jit: bool = False  # 是否启用 JIT 加速，仅支持 v1 和 v2。
    load_trt: bool = False  # 是否启用 TensorRT，需要预先生成匹配当前 GPU 的 Engine。
    load_vllm: bool = False  # 是否启用 vLLM，仅支持 v2 和 v3，并会增加显存占用。


CONFIG = InferenceConfig()


def validate_config(config: InferenceConfig) -> None:
    if config.model_version not in MODEL_DIRS:
        raise ValueError("model_version 只能是 v1、v2 或 v3")
    if config.mode not in {"zero_shot", "cross_lingual", "sft", "instruct"}:
        raise ValueError("mode 只能是 zero_shot、cross_lingual、sft 或 instruct")
    if not config.text.strip():
        raise ValueError("text 不能为空")
    if config.speed <= 0:
        raise ValueError("speed 必须大于 0")
    if config.mode == "zero_shot" and not config.prompt_text:
        raise ValueError("zero_shot 模式必须配置 prompt_text")
    if config.mode == "sft" and not config.speaker:
        raise ValueError("sft 模式必须配置 speaker")
    if config.mode == "sft" and config.model_version != "v1":
        raise ValueError("sft 模式仅支持 CosyVoice v1")
    if config.mode == "instruct" and not config.instruct_text:
        raise ValueError("instruct 模式必须配置 instruct_text")
    if config.mode == "instruct" and config.model_version == "v1" and not config.speaker:
        raise ValueError("CosyVoice v1 指令模式必须配置 speaker")
    if config.model_version == "v3" and config.load_jit:
        raise ValueError("CosyVoice v3 不支持 JIT")
    if config.model_version == "v1" and config.load_vllm:
        raise ValueError("CosyVoice v1 不支持 vLLM")


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(config: InferenceConfig):
    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = MODEL_DIRS[config.model_version]
    kwargs = {
        "model_dir": model_dir,
        "fp16": config.fp16,
        "load_trt": config.load_trt,
    }
    if config.model_version != "v3":
        kwargs["load_jit"] = config.load_jit
    if config.model_version != "v1":
        kwargs["load_vllm"] = config.load_vllm

    started_at = time.perf_counter()
    model = AutoModel(**kwargs)
    synchronize_cuda()
    load_seconds = time.perf_counter() - started_at

    actual_class = type(model).__name__
    expected_class = MODEL_CLASSES[config.model_version]
    if actual_class != expected_class:
        raise ValueError(
            f"模型版本 {config.model_version} 应加载 {expected_class}，"
            f"但目录 {model_dir!r} 中实际加载了 {actual_class}。"
        )
    return model, model_dir, load_seconds


def add_control_tokens(text: str, version: str, append_marker: bool = False) -> str:
    """为较新版本模型补充缺失的控制标记。"""
    marker = "<|endofprompt|>"
    if marker in text:
        return text
    if version == "v3":
        if append_marker:
            return f"You are a helpful assistant. {text}{marker}"
        return f"You are a helpful assistant.{marker}{text}"
    if version == "v2" and append_marker:
        return f"{text}{marker}"
    return text


def create_generator(
    model, config: InferenceConfig
) -> Iterable[Dict[str, torch.Tensor]]:
    common = {"stream": config.stream, "speed": config.speed}
    if config.mode == "zero_shot":
        prompt_text = add_control_tokens(config.prompt_text, config.model_version)
        return model.inference_zero_shot(
            config.text, prompt_text, config.prompt_wav, **common
        )
    if config.mode == "cross_lingual":
        text = add_control_tokens(config.text, config.model_version)
        return model.inference_cross_lingual(text, config.prompt_wav, **common)
    if config.mode == "sft":
        return model.inference_sft(config.text, config.speaker, **common)

    if config.model_version == "v1":
        return model.inference_instruct(
            config.text, config.speaker, config.instruct_text, **common
        )
    instruct_text = add_control_tokens(
        config.instruct_text, config.model_version, append_marker=True
    )
    return model.inference_instruct2(
        config.text, instruct_text, config.prompt_wav, **common
    )


def run_inference(model, config: InferenceConfig):
    generator = create_generator(model, config)
    chunks = []
    first_chunk_seconds = None
    synchronize_cuda()
    started_at = time.perf_counter()

    for result in generator:
        synchronize_cuda()
        if first_chunk_seconds is None:
            first_chunk_seconds = time.perf_counter() - started_at
        chunks.append(result["tts_speech"].detach().cpu())

    synchronize_cuda()
    inference_seconds = time.perf_counter() - started_at
    if not chunks:
        raise RuntimeError("模型没有生成任何音频")
    return torch.cat(chunks, dim=1), first_chunk_seconds, inference_seconds


def main() -> None:
    validate_config(CONFIG)
    model, model_dir, load_seconds = load_model(CONFIG)
    speech, first_chunk_seconds, inference_seconds = run_inference(model, CONFIG)

    output_path = Path(CONFIG.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), speech, model.sample_rate)

    audio_seconds = speech.shape[1] / model.sample_rate
    rtf = inference_seconds / audio_seconds
    print(f"模型版本      : {CONFIG.model_version}")
    print(f"模型目录      : {model_dir}")
    print(f"输出文件      : {output_path.resolve()}")
    print(f"模型加载时间  : {load_seconds:.3f} 秒")
    print(f"首包延迟      : {first_chunk_seconds:.3f} 秒")
    print(f"推理时间      : {inference_seconds:.3f} 秒")
    print(f"音频时长      : {audio_seconds:.3f} 秒")
    print(f"实时率 RTF    : {rtf:.4f}")


if __name__ == "__main__":
    main()
