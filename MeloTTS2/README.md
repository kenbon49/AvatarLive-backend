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

Docker 服务默认使用 `cuda:0`。BERT 与 MeloTTS 声学模型均直接使用 PyTorch 推理。
应用启动阶段会用 `MELOTTS_WARMUP_TEXT` 执行一次合成，将
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
服务没有推理后端开关，也不需要导出模型；`TTS` 始终加载本地 PyTorch 权重。

## 推理流程

1. `split_utils.py` 按中英文标点切分长文本。
2. `text/` 完成数字与标点归一化、中文拼音/英文音素转换、声调和语言 ID 编码。
3. PyTorch 在 GPU 上执行 `multilingual` 或 `english_bert`，生成 BERT 特征。
4. PyTorch `SynthesizerTrn.infer` 接收 CUDA BERT 特征，预测时长、隐变量和对齐并输出波形。
5. 多句音频之间插入 50 ms 静音，最终写为 WAV。

## 目录边界

- `infer.py`：唯一命令行入口。
- `melo/`：推理 API、文本前端及 VITS/MeloTTS 网络定义。
- `pretrained_model/EN`、`pretrained_model/ZH`：中英文声学模型。
- `bert_model/english_bert`、`bert_model/multilingual`：实际使用的 BERT 权重与 tokenizer。

训练脚本、训练数据、TensorBoard 日志、Web/Flask/Gradio 服务、日语/韩语/法语/西语前端、语音增强、测试模型及重复 checkpoint 均不属于推理闭包，已移除。
