"""Simple end-to-end test for the server_total WebSocket API."""

import argparse
import asyncio
import json
import struct

import websockets


PACKET_HEADER = struct.Struct("<4sBBHIIQ")


async def test_server(url, question, profile, language, speed, timeout):
    audio_packets = 0
    video_frames = 0
    answer = ""
    text_units = []
    expected_media_sequence = 0
    last_pts = 0

    async with websockets.connect(url, open_timeout=timeout, max_size=None) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"服务未就绪: {ready}")

        await ws.send(
            json.dumps(
                {
                    "type": "ask",
                    "question": question,
                    "profile": profile,
                    "language": language,
                    "speed": speed,
                },
                ensure_ascii=False,
            )
        )
        print(f"问题已发送: {question}")

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout)
            if isinstance(message, str):
                event = json.loads(message)
                event_type = event.get("type", "unknown")
                print(f"事件: {event_type}")

                if event_type == "llm_result":
                    answer = event.get("answer", "")
                    print(f"大模型回答: {answer}")
                elif event_type == "text_unit":
                    text_units.append(event)
                    print(
                        f"文本单元 {event.get('seq')}: "
                        f"{event.get('text')} [{event.get('delimiter')}]"
                    )
                elif event_type == "error":
                    raise RuntimeError(event.get("message", "服务返回未知错误"))
                elif event_type == "conversation_end":
                    break
                continue

            if len(message) < PACKET_HEADER.size:
                raise RuntimeError("收到无效的媒体包")
            header = PACKET_HEADER.unpack_from(message)
            magic, _version, packet_type, _flags, sequence, size, pts = header
            if magic != b"MSTK" or len(message) - PACKET_HEADER.size != size:
                raise RuntimeError("媒体包格式不正确")
            if sequence != expected_media_sequence:
                raise RuntimeError(
                    f"媒体包序号不连续: 期望 {expected_media_sequence}, 收到 {sequence}"
                )
            if pts < last_pts:
                raise RuntimeError(f"媒体时间戳回退: {last_pts} -> {pts}")
            expected_media_sequence += 1
            last_pts = pts
            if packet_type == 1:
                video_frames += 1
            elif packet_type == 2:
                audio_packets += 1

    if not answer or not text_units or not audio_packets or not video_frames:
        raise RuntimeError("测试未收到完整的大模型、音频和视频结果")
    if [unit.get("seq") for unit in text_units] != list(range(len(text_units))):
        raise RuntimeError("文本单元序号不连续")
    if "".join(str(unit.get("text", "")) for unit in text_units) != answer:
        raise RuntimeError("文本单元无法还原完整回答")
    print(
        f"测试成功: 文本单元 {len(text_units)} 个，"
        f"音频包 {audio_packets} 个，视频帧 {video_frames} 帧"
    )


def main():
    parser = argparse.ArgumentParser(description="测试 server_total 完整推理链路")
    parser.add_argument("--url", default="ws://localhost:8080/v1/conversation")
    parser.add_argument("--question", default="请用一句话介绍北京。")
    parser.add_argument(
        "--profile", choices=("chinese", "business_male_1"), default="chinese"
    )
    parser.add_argument("--language", choices=("ZH", "EN"), default="ZH")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    try:
        asyncio.run(test_server(**vars(args)))
    except TimeoutError:
        raise SystemExit("测试失败: 等待服务响应超时") from None
    except Exception as exc:
        raise SystemExit(f"测试失败: {exc}") from None


if __name__ == "__main__":
    main()
