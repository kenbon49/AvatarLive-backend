from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from server_total.app import (
    AVATAR_CATALOG,
    AskRequest,
    IdleStartRequest,
    MediaTimeline,
    _render_units,
    _run_idle,
    avatars,
    conversation,
    run_pipeline,
    synthesize_speech,
)
from server_total.protocol import PACKET_HEADER, remap_media_packet


class FakeFrontend:
    def __init__(self) -> None:
        self.json_messages = []
        self.binary_messages = []

    async def send_json(self, value):
        self.json_messages.append(value)

    async def send_bytes(self, value):
        self.binary_messages.append(value)


class FakeConversationSocket(FakeFrontend):
    def __init__(self, messages) -> None:
        super().__init__()
        self.messages = iter(messages)

    async def accept(self):
        return None

    async def receive_json(self):
        try:
            return next(self.messages)
        except StopIteration as exc:
            raise WebSocketDisconnect from exc


class YieldingConversationSocket(FakeConversationSocket):
    """Let background idle tasks make progress between control messages."""

    async def receive_json(self):
        await asyncio.sleep(0.05)
        return await super().receive_json()


class FakeUpstream:
    def __init__(self, messages) -> None:
        self.messages = iter(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def recv(self):
        return next(self.messages)

    async def send(self, value):
        self.sent.append(value)


class CyclingUpstream(FakeUpstream):
    """Replay a fixed message list forever, yielding to the event loop each recv."""

    def __init__(self, messages) -> None:
        self.messages = list(messages)
        self.sent = []
        self.index = 0

    async def recv(self):
        await asyncio.sleep(0)
        message = self.messages[self.index % len(self.messages)]
        self.index += 1
        return message


class FakeHttpResponse:
    content = b"\x00\x00\x01\x00"
    headers = {"X-Duration-Seconds": "0.125"}

    def raise_for_status(self):
        return None


class FakeHttpClient:
    def __init__(self) -> None:
        self.request = None

    async def post(self, url, json):
        self.request = (url, json)
        return FakeHttpResponse()


class FakeLLM:
    def stream_answer_text(self, _question):
        yield "人工智能能够快速处理数据，识别"
        yield "图像。"


def media_packet(sequence: int = 9, pts_us: int = 0, payload: bytes = b"media") -> bytes:
    return PACKET_HEADER.pack(b"MSTK", 1, 1, 0, sequence, len(payload), pts_us) + payload


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_and_validates_all_public_avatars(self):
        response = await avatars()
        profile_ids = [item["id"] for item in response["avatars"]]
        self.assertEqual(
            profile_ids,
            [
                "chinese",
                "business_male_1",
                "casual_male",
                "middle_aged_male",
                "casual_conversation",
                "casual_female",
            ],
        )
        self.assertEqual(response["avatars"], AVATAR_CATALOG)
        for profile_id in profile_ids:
            request = AskRequest(type="ask", question="介绍能力", profile=profile_id)
            self.assertEqual(request.profile, profile_id)
        with self.assertRaises(ValidationError):
            AskRequest(type="ask", question="介绍能力", profile="unknown")

    async def test_rejects_non_onnx_musetalk_backend(self):
        upstream = FakeUpstream(['{"type":"ready","fps":25,"backend":"torch"}'])
        with patch("server_total.app.websockets.connect", return_value=upstream):
            with self.assertRaisesRegex(RuntimeError, "expected onnx, got torch"):
                await _render_units(
                    FakeFrontend(),
                    AskRequest(type="ask", question="backend check"),
                    "request-backend",
                    asyncio.Queue(),
                )

    async def test_synthesize_returns_validated_pcm(self):
        http = FakeHttpClient()
        fake_app = SimpleNamespace(state=SimpleNamespace(http=http))
        request = AskRequest(type="ask", question="你好")
        pcm, duration = await synthesize_speech(fake_app, request, "简短回答")
        self.assertEqual(pcm, FakeHttpResponse.content)
        self.assertEqual(duration, 0.125)
        self.assertEqual(http.request[1]["sample_rate"], 16000)
        self.assertEqual(http.request[1]["text"], "简短回答")
        self.assertEqual(http.request[1]["bert_backend"], "onnx")

    async def test_pipeline_streams_each_text_unit_through_tts_and_musetalk(self):
        upstream = FakeUpstream(
            [
                '{"type":"ready","fps":25,"backend":"onnx"}',
                '{"type":"started","profile":"business_male_1"}',
                '{"type":"queued","profile":"business_male_1"}',
                '{"type":"stream_start"}',
                media_packet(4, 0, b"first"),
                '{"type":"stream_end","packets":1}',
                '{"type":"started","profile":"business_male_1"}',
                '{"type":"queued","profile":"business_male_1"}',
                '{"type":"stream_start"}',
                media_packet(7, 20_000, b"second"),
                '{"type":"stream_end","packets":1}',
            ]
        )
        frontend = FakeFrontend()
        request = AskRequest(
            type="ask",
            question="介绍能力",
            request_id="request-1",
            profile="business_male_1",
        )
        pcm = bytes(3200)
        synthesize = AsyncMock(side_effect=[(pcm, 0.1), (pcm, 0.1)])

        from server_total import app as app_module

        previous_llm = app_module.app.state.llm if hasattr(app_module.app.state, "llm") else None
        app_module.app.state.llm = FakeLLM()
        try:
            with (
                patch("server_total.app.synthesize_speech", synthesize),
                patch("server_total.app.websockets.connect", return_value=upstream),
            ):
                await run_pipeline(frontend, request, "request-1")
        finally:
            if previous_llm is None:
                del app_module.app.state.llm
            else:
                app_module.app.state.llm = previous_llm

        self.assertEqual(
            [call.args[2] for call in synthesize.await_args_list],
            ["人工智能能够快速处理数据，", "识别图像。"],
        )
        text_units = [item for item in frontend.json_messages if item["type"] == "text_unit"]
        self.assertEqual(
            [(item["seq"], item["text"], item["delimiter"]) for item in text_units],
            [(0, "人工智能能够快速处理数据，", "，"), (1, "识别图像。", "。")],
        )
        self.assertEqual([item["pts_us"] for item in text_units], [0, 100_000])
        for text_unit in text_units:
            text_index = frontend.json_messages.index(text_unit)
            next_message = frontend.json_messages[text_index + 1]
            self.assertEqual(next_message["type"], "stream_start")
            self.assertEqual(next_message["segment_seq"], text_unit["seq"])
        self.assertEqual(len(frontend.binary_messages), 2)
        ready = next(item for item in frontend.json_messages if item["type"] == "musetalk_ready")
        self.assertEqual(ready["backend"], "onnx")
        first = PACKET_HEADER.unpack_from(frontend.binary_messages[0])
        second = PACKET_HEADER.unpack_from(frontend.binary_messages[1])
        self.assertEqual((first[4], first[6]), (0, 0))
        self.assertEqual((second[4], second[6]), (1, 120_000))
        self.assertEqual(frontend.json_messages[-1]["type"], "conversation_end")
        self.assertEqual(frontend.json_messages[-1]["units"], 2)
        commits = [item for item in upstream.sent if isinstance(item, str) and "commit" in item]
        self.assertEqual(len(commits), 2)
        starts = [item for item in upstream.sent if isinstance(item, str) and "start" in item]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all('"profile": "business_male_1"' in item for item in starts))

    async def test_pipeline_does_not_deadlock_when_musetalk_fails(self):
        upstream = FakeUpstream(
            [
                '{"type":"ready","fps":25,"backend":"onnx"}',
                '{"type":"started","profile":"chinese"}',
                '{"type":"queued","profile":"chinese"}',
                '{"type":"error","message":"render failed"}',
            ]
        )
        frontend = FakeFrontend()
        request = AskRequest(type="ask", question="介绍能力", request_id="request-2")

        from server_total import app as app_module

        previous_llm = app_module.app.state.llm if hasattr(app_module.app.state, "llm") else None
        app_module.app.state.llm = FakeLLM()
        try:
            with (
                patch(
                    "server_total.app.synthesize_speech",
                    AsyncMock(return_value=(bytes(3200), 0.1)),
                ),
                patch("server_total.app.websockets.connect", return_value=upstream),
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    await asyncio.wait_for(
                        run_pipeline(frontend, request, "request-2"), timeout=1
                    )
        finally:
            if previous_llm is None:
                del app_module.app.state.llm
            else:
                app_module.app.state.llm = previous_llm

    async def test_conversation_cancels_the_active_pipeline(self):
        websocket = FakeConversationSocket(
            [
                {"type": "ask", "question": "介绍能力", "request_id": "request-3"},
                {"type": "cancel", "request_id": "request-3"},
            ]
        )

        async def slow_pipeline(*_args):
            await asyncio.Event().wait()

        with patch("server_total.app._run_and_report", side_effect=slow_pipeline):
            await conversation(websocket)

        cancelled = [
            item
            for item in websocket.json_messages
            if item.get("type") == "conversation_end" and item.get("cancelled")
        ]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["request_id"], "request-3")

    async def test_conversation_streams_idle_until_idle_stop(self):
        upstream = CyclingUpstream(
            [
                '{"type":"ready","fps":25,"backend":"onnx"}',
                '{"type":"started","profile":"chinese"}',
                '{"type":"queued","profile":"chinese"}',
                '{"type":"stream_start"}',
                media_packet(4, 0, b"idle"),
                '{"type":"stream_end","packets":1}',
            ]
        )
        websocket = YieldingConversationSocket(
            [
                {"type": "idle_start"},
                {"type": "idle_stop"},
            ]
        )
        with patch("server_total.app.websockets.connect", return_value=upstream):
            await conversation(websocket)

        types = [item["type"] for item in websocket.json_messages]
        self.assertIn("idle_started", types)
        self.assertIn("idle_stopped", types)
        starts = [item for item in upstream.sent if isinstance(item, str) and '"start"' in item]
        self.assertGreaterEqual(len(starts), 1)
        self.assertIn('"profile": "chinese"', starts[0])

    async def test_idle_renderer_streams_silence_until_cancelled(self):
        upstream = CyclingUpstream(
            [
                '{"type":"ready","fps":25,"backend":"onnx"}',
                '{"type":"started","profile":"chinese"}',
                '{"type":"queued","profile":"chinese"}',
                '{"type":"stream_start"}',
                media_packet(4, 0, b"idle"),
                '{"type":"stream_end","packets":1}',
            ]
        )
        frontend = FakeFrontend()
        timeline = MediaTimeline()
        with patch("server_total.app.websockets.connect", return_value=upstream):
            task = asyncio.create_task(
                _run_idle(frontend, IdleStartRequest(type="idle_start"), "idle", timeline)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        types = [item["type"] for item in frontend.json_messages]
        self.assertIn("idle_started", types)
        self.assertGreaterEqual(types.count("idle_chunk"), 1)
        self.assertTrue(any(isinstance(item, bytes) for item in frontend.binary_messages))
        self.assertGreater(timeline.sequence, 0)

    async def test_idle_timeline_limits_media_generated_ahead_of_playback(self):
        timeline = MediaTimeline(pts_offset_us=2_000_000, playback_started_at=99.0)
        delay = AsyncMock()
        with (
            patch("server_total.app.IDLE_MAX_AHEAD_SECONDS", 1.5),
            patch("server_total.app.time.perf_counter", return_value=100.0),
            patch("server_total.app.asyncio.sleep", delay),
        ):
            await timeline.wait_for_idle_budget(1_000_000)

        delay.assert_awaited_once_with(0.5)

    def test_remap_media_packet_rewrites_sequence_and_pts(self):
        packet = media_packet(99, 25_000, b"jpeg")
        outgoing, packet_type, pts_us = remap_media_packet(
            packet, sequence=3, pts_offset_us=100_000
        )
        header = PACKET_HEADER.unpack_from(outgoing)
        self.assertEqual(packet_type, 1)
        self.assertEqual(pts_us, 125_000)
        self.assertEqual(header[4], 3)
        self.assertEqual(header[6], 125_000)
        self.assertEqual(outgoing[PACKET_HEADER.size :], b"jpeg")


if __name__ == "__main__":
    unittest.main()
