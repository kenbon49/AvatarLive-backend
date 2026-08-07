# MeloTTS 中英文离线推理精简版

本目录只保留中文（支持中英混合文本）和英文语音合成推理所需的代码、声学模型与 BERT 权重，不包含训练、数据预处理、Web 服务、语音增强及其他语言模型。

## 保留的模型

| 语言参数 | 声学模型 | BERT | 说话人 |
| --- | --- | --- | --- |
| `ZH` | `pretrained_model/ZH` | `bert_model/multilingual` | `ZH` |
| `EN` | `pretrained_model/EN` | `bert_model/english_bert` | `EN-US`、`EN-BR`、`EN_INDIA`、`EN-AU`、`EN-Default` |

`ZH` 权重在代码中映射为 `ZH_MIX_EN`。它的文本特征维度是 768，实际使用 `multilingual` BERT；原来的 `chinese_bert` 是 1024 维，不参与该权重的推理，因此已删除。

## 安装

本项目以 **Python 3.10** 为运行基线。建议创建独立环境：

```powershell
conda create -n melotts-infer python=3.10 -y
conda activate melotts-infer
python -m pip install -r requirements.txt
```

所有权重均从当前目录读取，推理过程不会联网下载模型。

Docker 服务默认使用 `cuda:0`。默认推理链为 ONNX Runtime BERT + PyTorch MeloTTS
声学模型。应用启动阶段会用 `MELOTTS_WARMUP_TEXT` 执行一次合成，将
`MELOTTS_DEFAULT_LANGUAGE` 对应的两个模型加载到显卡；
如果 CUDA 不可用，启动会直接失败，不会静默回退到 CPU。`/health` 中的
`gpu_ready`、`model_devices` 和 `bert_devices` 可用于确认权重驻留位置。

## 命令行推理

中文或中英混合：

```powershell
python infer.py --language ZH --text "你好，欢迎使用 MeloTTS. This is a test." --output output_zh.wav
```

英文：

```powershell
python infer.py --language EN --speaker EN-US --text "Hello from MeloTTS." --output output_en.wav
```

可选参数：

- `--device auto|cpu|cuda|cuda:0`：默认自动选择 CUDA 或 CPU。
- `--speed 1.0`：语速，必须大于 0。
- `--speaker`：不传时使用对应模型的第一个说话人。

输出采样率由模型配置决定，当前中英文模型均为 44100 Hz。

## Python 调用

```python
from melo import TTS

model = TTS("ZH", device="auto")
speaker_id = model.hps.data.spk2id["ZH"]
audio = model.tts_to_file(
    "你好，This is MeloTTS.",
    speaker_id,
    "output.wav",
    speed=1.0,
)
```

`tts_to_file` 同时返回 `float32` NumPy 音频数组；将输出路径设为 `None` 时只返回数组，不写文件。
`TTS` 默认使用 `bert_backend="onnx"`；仅在需要回退验证时显式传入
`bert_backend="pytorch"`。声学模型始终使用 PyTorch。

## ONNX 导出与推理

一次性导出中英文 BERT 和 MeloTTS 声学模型：

```powershell
python export_onnx.py --device cpu
```

默认输出到 `onnx_models/`：

```text
onnx_models/
  bert/
    english_bert.onnx
    multilingual_bert.onnx
  EN/
    encoder_duration.onnx
    flow.onnx
    generator.onnx
  ZH/
    encoder_duration.onnx
    flow.onnx
    generator.onnx
  manifest.json
```

也可以只导出指定语言或指定模型类型：

```powershell
python export_onnx.py --languages ZH --skip-acoustic
python export_onnx.py --languages EN --skip-bert
```

ONNX Runtime 命令行推理：

```powershell
python infer_onnx.py --language ZH --text "你好，This is MeloTTS." --output output_zh_onnx.wav --device cuda
python infer_onnx.py --language EN --speaker EN-US --text "Hello from ONNX." --output output_en_onnx.wav --device cuda
```

Python 调用：

```python
from melo import OnnxTTS

model = OnnxTTS("ZH", device="cuda:0", seed=1234)
speaker_id = model.hps.data.spk2id.ZH
audio = model.tts_to_file("你好，欢迎使用 ONNX。", speaker_id, "output.wav")
```

声学模型按 `encoder_duration -> 动态对齐/采样 -> flow -> generator` 分图。CUDA 推理默认启用 ONNX Runtime I/O Binding，BERT 输出、动态对齐以及三个声学子图之间通过 CUDA Tensor 直接传递，仅最终音频复制回 CPU。CUDA Provider 使用 ONNX Runtime 1.17.1 支持的默认计算流配置，避免因不兼容的 Provider option 触发 Session 重建。随机时长噪声作为显式输入，因此指定相同 `seed` 可以复现 ONNX 输出。文本归一化、G2P、tokenizer 和 WAV 写入仍在 CPU/Python 中，动态帧数分配需要读取一个 GPU 标量。ONNX 推理强制使用 `CUDAExecutionProvider` 并关闭执行回退，没有 GPU Provider 时直接报错；仅在显式传入 `device="cpu"` 时才使用 CPU。导出后应使用固定文本对比 PyTorch 与 ONNX 的发音、时长和音质。

## 推理流程

1. `split_utils.py` 按中英文标点切分长文本。
2. `text/` 完成数字与标点归一化、中文拼音/英文音素转换、声调和语言 ID 编码。
3. ONNX Runtime 在 GPU 上执行 `multilingual` 或 `english_bert`，并通过 I/O Binding 生成 CUDA 特征。
4. PyTorch `SynthesizerTrn.infer` 直接接收 CUDA BERT 特征，预测时长、隐变量和对齐并输出波形。
5. 多句音频之间插入 50 ms 静音，最终写为 WAV。

## 目录边界

- `infer.py`：唯一命令行入口。
- `melo/`：推理 API、文本前端及 VITS/MeloTTS 网络定义。
- `pretrained_model/EN`、`pretrained_model/ZH`：中英文声学模型。
- `bert_model/english_bert`、`bert_model/multilingual`：实际使用的 BERT 权重与 tokenizer。

训练脚本、训练数据、TensorBoard 日志、Web/Flask/Gradio 服务、日语/韩语/法语/西语前端、语音增强、测试模型及重复 checkpoint 均不属于推理闭包，已移除。
