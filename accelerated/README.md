# MuseTalk v1.5 流式加速服务

该目录提供独立的实时推理服务，不改动原有的离线 `/generate` 接口。服务启动后只加载一份
MuseTalk v1.5、Whisper、VAE 和 UNet；头像视频的脸框、VAE latent 与融合 mask 会写入
`cache/accelerated/`，后续启动直接复用。

## 公共形象

- `chinese`：`data/public/chinese.mp4`，默认公共形象。
- `business_male_1`：`data/public/商务男1.mp4`。

公共可选数字人视频统一存放在 `data/public/`。新增形象时应将源视频放入该目录，
并在 `accelerated/runtime.py` 的 `avatar_specs` 中注册对应的形象 ID。
服务启动时会依次为全部公共形象生成人脸框、VAE latent 和融合 mask 缓存，并完成
GPU warm-up；所有形象准备完毕后 `/readiness` 才会返回 `ready`。

为了兼顾实时速度与清晰度，Docker 流式服务使用 batch 4、输出 16 FPS，并把头像
等比缩放到最高 720p。16 FPS 会覆盖源视频的完整时间范围，不会降低人物动作速度；
可用 `--fps` 和 `--height` 调整，更改后缓存会自动失效并重建。

## 启动

```powershell
python -m accelerated.server --host 0.0.0.0 --port 8083
```

首次启动会执行姿态检测与 MuseTalk 素材预计算，时间明显长于后续启动。浏览器访问
`http://localhost:8083/` 可使用内置演示页；`GET /readiness` 返回 `ready=true` 后才能推理。

常用环境变量：`MUSETALK_PORT`、`MUSETALK_HEIGHT`、`MUSETALK_BATCH_SIZE`、
`MUSETALK_DEVICE`、`MUSETALK_BACKEND`。也可使用同名命令行参数覆盖高度、batch、
设备和推理后端。

导出模型后，可让 Whisper Encoder、UNet 和 VAE Decoder 走 ONNX Runtime CUDA：

```powershell
python scripts/export_musetalk_onnx.py --models all --batch-size 1
python -m accelerated.server --backend onnx --onnx-dir models/onnx --batch-size 1
```

ONNX 路径使用 CUDA I/O Binding，UNet 与 VAE 的中间张量不会绕行 CPU。头像素材的
VAE Encoder 只在缓存预计算阶段运行，仍使用 PyTorch；实时逐批链路使用 ONNX UNet
和 ONNX VAE Decoder。若 CUDA provider 不可用，服务会明确启动失败，不会静默回退 CPU。

## 测试客户端

服务的 `/readiness` 返回 `ready=true` 后，在另一个终端运行：

```powershell
python -m accelerated.client
```

客户端默认使用 `data/input/audio/demo2_audio.wav` 驱动 `chinese`，逐包校验序号、
音视频时间戳和 PCM 完整性，并将接收到的内容合成为
`data/output/accelerated_stream_test.mp4`。测试其他形象或音频：

```powershell
python -m accelerated.client --profile chinese --audio data/input/audio/demo1_audio.wav --output data/output/chinese_stream_test.mp4
```

远程服务可通过 `--url ws://SERVER:8083/v1/stream` 指定。`--timeout` 是等待每条服务端
消息的最大秒数，首次构建 `chinese` 缓存时可适当增大。

## WebSocket 协议

连接 `ws://HOST:8083/v1/stream`，服务返回 `ready` 后：

1. 发送 `{"type":"start","profile":"chinese"}`；可用人物由 `/v1/avatars` 返回，
   当前包括 `chinese`、`business_male_1`、`casual_male`、`middle_aged_male`、
   `casual_conversation` 和 `casual_female`。
2. 发送任意数量的二进制 PCM 数据（16 kHz、单声道、signed 16-bit little-endian）。
3. 发送 `{"type":"commit"}`。一段音频上限 120 秒。
4. 服务逐 batch 返回 PCM 音频和 JPEG 帧，最后返回 `stream_end`。

每个输出二进制消息都由 24 字节 little-endian 头和 payload 组成：

```text
<4sBBHIIQ
magic="MSTK", version=1, type, flags=0, sequence, payload_size, pts_us
```

`type=1` 是 JPEG RGB 视频帧，`type=2` 是 PCM s16le 音频。前端应使用 `pts_us`
同步 Web Audio 和画面，不能假设 WebSocket 消息到达间隔就是播放间隔。

同一连接可多次 `start -> binary audio -> commit`。发送 `cancel` 会清空尚未提交的音频。
GPU 推理通过全局队列串行执行，避免多个会话同时占用模型导致显存峰值。
