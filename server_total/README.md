# LLM + MeloTTS2 + MuseTalk 流式总服务

`server_total` 将三个服务组成有背压的端到端流水线：

```text
LiteLLM token 流
  -> answer 正文增量
  -> 逐标点 text_unit
  -> 每个单元调用 ONNX BERT + PyTorch MeloTTS
  -> 每个 PCM 单元独立提交 MuseTalk
  -> 连续时间轴的 PCM/JPEG 媒体包
```

## 启动

复制并修改 LiteLLM 环境变量：

```powershell
Copy-Item llm_inference/.env.example llm_inference/.env
docker compose -f docker-compose.dev.yml up --build
```

三个容器分别为：

- `musetalk_dev`：GPU 数字人流式推理，内部端口 `8083`。
- `melotts_dev`：MeloTTS2 常驻语音合成，内部端口 `8084`。
- `server_total_dev`：流式编排 API，宿主机端口 `8080`。

`server_total` 向 MeloTTS2 请求时固定声明 `bert_backend=onnx`。因此默认语音链路使用
ONNX Runtime BERT GPU 推理，BERT CUDA 特征随后直接交给 PyTorch MeloTTS 声学模型。

MuseTalk 流式服务同样默认使用 ONNX Runtime CUDA：Whisper Encoder、MuseTalk UNet
和 VAE Decoder 均加载 `models/onnx/` 下的转换产物，固定使用已经验证的 batch 1。
`server_total` 会校验 MuseTalk WebSocket 握手中的 `backend=onnx`，避免误连到 PyTorch
后端。首次启动前先在宿主机导出模型：

```powershell
python scripts/export_musetalk_onnx.py --models all --batch-size 1
docker compose -f docker-compose.dev.yml up --build
```

根目录 `requirements.txt` 安装 `onnx` 和 `onnxruntime-gpu`；轻量的
`server_total` 网关不加载模型，所以它自己的 requirements 不重复安装 GPU runtime。

```text
GET  http://localhost:8080/health
GET  http://localhost:8080/v1/avatars
WS   ws://localhost:8080/v1/conversation
```

`GET /v1/avatars` 返回六个已在 MuseTalk 启动阶段完成预处理的公共形象：

- `chinese`（默认）：`data/public/chinese.mp4`
- `business_male_1`：`data/public/商务男1.mp4`
- `casual_male`：`data/public/休闲风.mp4`
- `middle_aged_male`：`data/public/中年.mp4`
- `casual_conversation`：`data/public/休闲交流.mp4`
- `casual_female`：`data/public/休闲2.mp4`

## 请求协议

连接收到 `ready` 后发送：

```json
{
  "type": "ask",
  "request_id": "可选的客户端请求ID",
  "question": "请简单介绍一下北京。",
  "profile": "chinese",
  "language": "ZH",
  "speed": 1.0
}
```

调用方通过 `profile` 选择本轮推理使用的数字人。省略该字段时使用 `chinese`；例如选择
商务男形象时传入 `"profile": "business_male_1"`。WebSocket 首次返回的 `ready`
事件也包含 `default_profile` 和 `avatars`，客户端可据此生成选择列表。

同一连接一次只运行一个回答。取消当前回答：

```json
{"type":"cancel","request_id":"对应的请求ID"}
```

连接本身不会触发 MuseTalk 推理。前端在未提问和回答播放完毕后显示所选源视频的首帧，
只有发送 `ask` 或 `speak` 后才会把真实语音提交给 MuseTalk 并接收流式媒体。服务不接受
`idle_start` / `idle_stop`，也不会用静音 PCM 生成静息画面。

## 流式事件

LLM 原始增量：

```json
{"type":"llm_delta","request_id":"...","delta":"它能处理数据、"}
```

达到切分条件后会产生一个不可再修改的结构化单元。该事件在对应媒体的
`stream_start` 前返回，并携带与二进制媒体包相同时间轴的 `pts_us`：

```json
{
  "type": "text_unit",
  "request_id": "...",
  "seq": 0,
  "text": "它能处理数据、",
  "delimiter": "、",
  "pts_us": 0
}
```

默认切分规则：

- `。！？.!?…` 立即切分。
- 首单元在 `，、,；;：:—` 处累计满 12 个正文字符后切分，以降低首包延迟。
- 后续单元在这些软标点处累计满 20 个正文字符后切分，通常保持在 20～40 字。
- 连续标点合并在同一个单元中；小数、版本号、时间和 URL 中的标点不会切分。
- 流结束时，剩余文字无论长短都会作为最后一个单元返回。

可通过 `PIPELINE_FIRST_UNIT_MIN_CHARS` 和 `PIPELINE_TARGET_UNIT_CHARS` 调整两个长度阈值。

每个 `text_unit` 都会独立经历：

```text
 tts_start -> tts_result -> segment_start
-> started/queued -> text_unit/stream_start -> binary media packets -> stream_end
-> segment_end
```

一轮回答最后返回：

```json
{
  "type": "conversation_end",
  "request_id": "...",
  "answer": "完整回答",
  "units": 4,
  "rendered_units": 4,
  "elapsed_ms": 3200,
  "cancelled": false
}
```

## 媒体协议

二进制媒体包使用 `MSTK/2` 会话语义，包头仍为小端结构 `<4sBBHIIQ>`：

- 包类型 `1`：JPEG 视频帧。
- 包类型 `2`：16 kHz 单声道 PCM s16le。
- 包序号在一轮回答内连续递增。
- PTS 单位为微秒，并在一轮回答的所有标点单元之间保持连续；下一轮回答从 0 重新开始。

MuseTalk 内部每个文本单元仍会从序号和 PTS 0 开始，`server_total` 会在转发前重写包头。
前端按包头 PTS 调度本轮媒体，并在回答播放结束后切回静态首帧。

## 背压与并发

- 文本队列默认最多缓存 4 个单元，可用 `PIPELINE_TEXT_QUEUE_SIZE` 调整。
- 音频队列默认最多缓存 2 个单元，可用 `PIPELINE_AUDIO_QUEUE_SIZE` 调整。
- 单个会话中的 TTS 和 MuseTalk 严格按 `seq` 顺序执行。
- 每个最终切出的单元都独立进行语音合成。
