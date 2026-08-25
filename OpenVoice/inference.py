"""OpenVoice V1/V2 简洁命令行推理脚本。"""

import argparse
import contextlib
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
V1_ROOT = ROOT / "checkpoints"
V2_ROOT = ROOT / "checkpoints_v2"
CACHE_ROOT = V2_ROOT / "runtime_cache"
CONFIG_PATH = ROOT / "model_config.yaml"

# 所有模型只使用项目内文件，禁止运行时联网下载。
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

# Directly invoking an environment's python.exe on Windows does not activate
# Conda's DLL search paths, which NVRTC needs for OpenVoice's scripted kernels.
_DLL_DIRECTORY_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    dll_paths = []
    for dll_dir in (Path(sys.prefix) / "bin", Path(sys.prefix) / "Library" / "bin"):
        if dll_dir.is_dir():
            dll_paths.append(str(dll_dir))
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    if dll_paths:
        os.environ["PATH"] = os.pathsep.join(dll_paths + [os.environ.get("PATH", "")])

import torch  # noqa: E402

from melo import TTS  # noqa: E402
from openvoice import se_extractor  # noqa: E402
from openvoice.api import BaseSpeakerTTS, ToneColorConverter  # noqa: E402


PATH_KEYS = {"config", "checkpoint", "default_embedding", "style_embedding", "embedding"}


def load_model_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    def resolve_paths(value):
        if isinstance(value, dict):
            return {
                key: ROOT / item if key in PATH_KEYS else resolve_paths(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [resolve_paths(item) for item in value]
        return value

    return resolve_paths(config)


MODEL_CONFIG = load_model_config()
V1_STYLES = tuple(MODEL_CONFIG["v1"]["styles"])


@contextlib.contextmanager
def suppress_model_output():
    """隐藏第三方模型的分句、音素、进度条等非核心输出。"""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少{name}：{path}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")
    return device


def load_converter(version: str, device: str):
    converter = MODEL_CONFIG[version]["converter"]
    config = converter["config"]
    checkpoint = converter["checkpoint"]
    require_file(config, "转换模型配置")
    require_file(checkpoint, "转换模型权重")

    model = ToneColorConverter(str(config), device=device, enable_watermark=False)
    model.load_ckpt(str(checkpoint))
    return model


def load_v1_synthesizer(args, device: str):
    model_config = MODEL_CONFIG["v1"]["languages"][args.language]
    config = model_config["config"]
    checkpoint = model_config["checkpoint"]
    embedding_key = "default_embedding" if args.style == "default" else "style_embedding"
    embedding = model_config[embedding_key]
    for path, name in ((config, "V1 配置"), (checkpoint, "V1 权重"), (embedding, "V1 音色向量")):
        require_file(path, name)

    model = BaseSpeakerTTS(str(config), device=device)
    model.load_ckpt(str(checkpoint))
    source_se = torch.load(embedding, map_location=device).to(device)
    return model, source_se


def synthesize_v1(args, model, audio_path: Path) -> None:
    model_config = MODEL_CONFIG["v1"]["languages"][args.language]
    model.tts(
        args.text, str(audio_path), speaker=args.style,
        language=model_config["name"], speed=args.speed,
    )


def load_v2_synthesizer(args, device: str):
    language_config = MODEL_CONFIG["v2"]["languages"][args.language]
    speaker = language_config["default_speaker"] if args.speaker == "auto" else args.speaker
    speaker_config = language_config["speakers"][speaker]
    embedding = speaker_config["embedding"]
    require_file(embedding, "V2 音色向量")

    model = TTS(language_config["melo_language"], device=device)
    source_se = torch.load(embedding, map_location=device).to(device)
    return model, source_se


def synthesize_v2(args, model, audio_path: Path) -> None:
    language_config = MODEL_CONFIG["v2"]["languages"][args.language]
    speaker = language_config["default_speaker"] if args.speaker == "auto" else args.speaker
    speaker_config = language_config["speakers"][speaker]
    model.tts_to_file(
        args.text,
        model.hps.data.spk2id[speaker_config["melo_speaker"]],
        str(audio_path),
        speed=args.speed,
        quiet=True,
    )


def synchronize_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def validate_args(args) -> None:
    if args.speed <= 0:
        raise ValueError("语速必须大于 0")
    if args.version == "v1":
        if args.language == "zh" and args.style != "default":
            raise ValueError("V1 中文仅支持 default 风格")
        if args.speaker != "auto":
            raise ValueError("speaker 参数仅用于 V2")
    else:
        if args.style != "default":
            raise ValueError("V2 不支持独立情感控制，请使用 default")
        language_config = MODEL_CONFIG["v2"]["languages"][args.language]
        speaker = language_config["default_speaker"] if args.speaker == "auto" else args.speaker
        if speaker not in language_config["speakers"]:
            choices = "、".join(language_config["speakers"])
            raise ValueError(f"当前语言可用的 V2 说话人：{choices}")


def run_inference(args) -> Path:
    validate_args(args)
    reference = (ROOT / args.reference).resolve() if not Path(args.reference).is_absolute() else Path(args.reference)
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    require_file(reference, "参考音频")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"使用设备：{device}")
    print(f"加载 {args.version.upper()} 模型……")

    with suppress_model_output():
        converter = load_converter(args.version, device)
        model, source_se = (
            load_v1_synthesizer(args, device)
            if args.version == "v1"
            else load_v2_synthesizer(args, device)
        )
        se_extractor.get_local_vad_model()

    print("提取参考音色……")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=output.parent, delete=False) as file:
            temp_path = Path(file.name)

        synchronize_device(device)
        inference_started = time.perf_counter()
        with suppress_model_output():
            target_se, _ = se_extractor.get_se(
                str(reference), converter, target_dir=str(ROOT / "processed"), vad=True
            )

        print("生成基础语音并转换音色……")
        with suppress_model_output():
            (
                synthesize_v1(args, model, temp_path)
                if args.version == "v1"
                else synthesize_v2(args, model, temp_path)
            )
            converter.convert(
                str(temp_path), source_se, target_se,
                output_path=str(output), message="@MyShell",
            )
        synchronize_device(device)
        inference_elapsed = time.perf_counter() - inference_started
        print(f"模型推理耗时：{inference_elapsed:.3f} 秒")
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenVoice 本地语音克隆推理",
        add_help=False,
        usage="python inference.py [参数]",
    )
    options = parser.add_argument_group("参数")
    options.add_argument("-h", "--help", action="help", help="显示帮助信息")
    options.add_argument("--version", choices=("v1", "v2"), default="v2", help="模型版本")
    options.add_argument("--language", choices=("zh", "en"), default="zh", help="生成语言")
    options.add_argument("--style", choices=V1_STYLES, default="default", help="V1 英文情感风格")
    options.add_argument("--speaker", default="auto", help="V2 说话人或口音，默认自动选择")
    options.add_argument("--speed", type=float, default=0.8, help="语速，大于 1 加快，小于 1 放慢")
    options.add_argument("--text", default=",,今天的天气很不错。", help="需要合成的文本")
    options.add_argument("--reference", default=r"D:\code\avatar\MuseTalk\data\input\audio\yongen.wav", help="参考音频")
    options.add_argument("--output", default="outputs/inference_yongen.wav", help="输出音频")
    options.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        output = run_inference(args)
        print(f"推理完成：{output}")
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"推理失败：{error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
