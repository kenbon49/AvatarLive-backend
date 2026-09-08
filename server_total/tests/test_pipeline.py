from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import AsyncMock
import wave

from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from server_total.app import (
    AVATAR_CATALOG,
    AskRequest,
    MediaSendPacer,
    SpeechPrepareRequest,
    SpeakRequest,
    _musetalk_avatar_catalog,
    _render_units,
    app,
    avatars,
    conversation,
    get_or_synthesize_speech,
    prepare_speech,
    run_pipeline,
    stream_speech_chunks,
    synthesize_speech,
    voice_preview,
    voices,
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


class FakeHttpResponse:
    content = bytes(4000)
    is_error = False
    headers = {
        "X-Audio-Sample-Rate": "16000",
        "X-Audio-Sample-Format": "s16le",
    }

    def raise_for_status(self):
        return None

    async def aiter_bytes(self, chunk_size=None):
        size = chunk_size or len(self.content)
        for offset in range(0, len(self.content), size):
            yield self.content[offset : offset + size]


class FakeCatalogResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        return False


class FakeHttpClient:
    def __init__(self) -> None:
        self.request = None

    async def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return FakeHttpResponse()

    def stream(self, _method, url, **kwargs):
        self.request = (url, kwargs)
        return FakeStreamContext(FakeHttpResponse())


class FakeLLM:
    def stream_answer_text(self, _question):
        yield "人工智能能够快速处理数据，识别"
        yield "图像。"


def media_packet(
    sequence: int = 9, pts_us: int = 0, payload: bytes = b"media"
) -> bytes:
    return (
        PACKET_HEADER.pack(b"MSTK", 1, 1, 0, sequence, len(payload), pts_us) + payload
    )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_media_sender_waits_when_pts_exceeds_send_ahead_window(self):
        pacer = MediaSendPacer(max_ahead_seconds=2.5, started_at=100.0)
        sleep = AsyncMock()

        with (
            patch("server_total.app.time.perf_counter", return_value=100.25),
            patch("server_total.app.asyncio.sleep", sleep),
        ):
            await pacer.wait_until_sendable(3_000_000)

        sleep.assert_awaited_once_with(0.25)

    async def test_media_sender_does_not_wait_inside_send_ahead_window(self):
        pacer = MediaSendPacer(max_ahead_seconds=2.5, started_at=100.0)
        sleep = AsyncMock()

        with (
            patch("server_total.app.time.perf_counter", return_value=100.25),
            patch("server_total.app.asyncio.sleep", sleep),
        ):
            await pacer.wait_until_sendable(2_500_000)

        sleep.assert_not_awaited()

    async def test_normalizes_melotts_speaker_catalog_for_web_clients(self):
        response = FakeCatalogResponse(
            {
                "default": "default",
                "speakers": [
                    {"id": "default", "name": "Default voice", "default": True},
                ],
            }
        )
        http = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(app.state, "http", http, create=True):
            catalog = await voices()

        self.assertEqual(catalog["default"], "default")
        self.assertEqual(
            catalog["voices"],
            [
                {
                    "voice_id": "default",
                    "name": "Default voice",
                    "kind": "preset",
                    "source": {
                        "provider": "MeloTTS",
                        "sample_url": "/v1/voices/default/preview",
                    },
                },
            ],
        )

    async def test_voice_preview_returns_browser_playable_wav(self):
        pcm = bytes(range(64))
        with patch(
            "server_total.app.synthesize_speech",
            AsyncMock(return_value=(pcm, len(pcm) / 32000)),
        ) as synthesize:
            response = await voice_preview("custom-voice")

        self.assertEqual(response.media_type, "audio/wav")
        self.assertEqual(response.headers["cache-control"], "public, max-age=86400")
        with wave.open(io.BytesIO(response.body), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.readframes(wav.getnframes()), pcm)
        self.assertEqual(synthesize.await_args.args[1].voice_id, "custom-voice")

    async def test_speak_splits_text_at_punctuation(self):
        from server_total.app import _produce_text_units

        frontend = FakeFrontend()
        queue = asyncio.Queue()
        request = SimpleNamespace(text="First sentence. Second sentence. Third sentence.")

        answer = await _produce_text_units(
            frontend, request, "request-speak", queue, kind="speak"
        )
        first = await queue.get()
        second = await queue.get()
        third = await queue.get()
        end = await queue.get()

        self.assertEqual(answer, request.text)
        self.assertEqual(
            [(first.seq, first.text), (second.seq, second.text), (third.seq, third.text)],
            [(0, "First sentence."), (1, "Second sentence."), (2, "Third sentence.")],
        )
        self.assertIsNotNone(end)

    async def test_exposes_and_validates_all_public_avatars(self):
        response = await avatars()
        profile_ids = [item["id"] for item in response["avatars"]]
        self.assertEqual(
            profile_ids,
            ["chinese", "business_male_1", "chen_yu"],
        )
        self.assertEqual(response["avatars"], AVATAR_CATALOG)
        for profile_id in profile_ids:
            request = AskRequest(type="ask", question="介绍能力", profile=profile_id)
            self.assertEqual(request.profile, profile_id)
        self.assertEqual(
            AskRequest(type="ask", question="介绍能力", profile="custom-avatar_v1").profile,
            "custom-avatar_v1",
        )
        with self.assertRaises(ValidationError):
            AskRequest(type="ask", question="介绍能力", profile="../unknown")

    async def test_catalog_only_exposes_published_custom_avatars(self):
        response = FakeCatalogResponse(
            [
                {"id": "custom-review_v1", "custom": True, "prepared": True, "status": "review"},
                {"id": "custom-cold_v1", "custom": True, "prepared": False, "status": "ready"},
                {
                    "id": "custom-ready_v1",
                    "name": "已发布形象",
                    "custom": True,
                    "prepared": True,
                    "status": "ready",
                },
            ]
        )
        http = SimpleNamespace(get=AsyncMock(return_value=response))
        with patch.object(app.state, "http", http, create=True):
            catalog = await _musetalk_avatar_catalog()

        self.assertEqual([item["id"] for item in catalog], ["custom-ready_v1"])

    async def test_conversation_rejects_unpublished_profile(self):
        socket = FakeConversationSocket(
            [{"type": "ask", "question": "测试", "profile": "custom-review_v1"}]
        )

        await conversation(socket)

        self.assertEqual(socket.json_messages[-1]["stage"], "request")
        self.assertIn("unknown or unpublished avatar", socket.json_messages[-1]["message"])

    async def test_rejects_non_torch_musetalk_backend(self):
        upstream = FakeUpstream(['{"type":"ready","fps":25,"backend":"onnx"}'])
        with patch("server_total.app.websockets.connect", return_value=upstream):
            with self.assertRaisesRegex(RuntimeError, "expected torch, got onnx"):
                await _render_units(
                    FakeFrontend(),
                    AskRequest(type="ask", question="backend check"),
                    "request-backend",
                    asyncio.Queue(),
                )

    async def test_rejects_non_float32_musetalk_inference(self):
        upstream = FakeUpstream(
            ['{"type":"ready","fps":25,"backend":"torch","inference_dtype":"float16"}']
        )
        with patch("server_total.app.websockets.connect", return_value=upstream):
            with self.assertRaisesRegex(RuntimeError, "expected float32, got float16"):
                await _render_units(
                    FakeFrontend(),
                    AskRequest(type="ask", question="dtype check"),
                    "request-dtype",
                    asyncio.Queue(),
                )

    async def test_synthesize_returns_validated_pcm(self):
        http = FakeHttpClient()
        fake_app = SimpleNamespace(state=SimpleNamespace(http=http))
        request = AskRequest(type="ask", question="你好")
        pcm, duration = await synthesize_speech(fake_app, request, "简短回答")
        self.assertEqual(pcm, FakeHttpResponse.content)
        self.assertEqual(duration, 0.125)
        self.assertTrue(http.request[0].endswith("/v1/tts"))
        self.assertNotIn("speaker_id", http.request[1]["data"])
        self.assertEqual(http.request[1]["data"]["language"], "zh")
        self.assertEqual(http.request[1]["data"]["tts_text"], "简短回答")
        self.assertEqual(http.request[1]["data"]["output_sample_rate"], 16000)

    async def test_speech_cache_reuses_pcm_and_separates_speed_and_voice(self):
        pcm = bytes(3200)
        synthesize = AsyncMock(return_value=(pcm, 0.1))
        base = SpeakRequest(type="speak", text="缓存测试。", voice_id="voice-a")
        faster = base.model_copy(update={"speed": 1.2})
        another_voice = base.model_copy(update={"voice_id": "voice-b"})

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("server_total.app.TTS_CACHE_DIR", Path(directory)),
            patch("server_total.app.synthesize_speech", synthesize),
        ):
            first = await get_or_synthesize_speech(app, base, base.text)
            second = await get_or_synthesize_speech(app, base, base.text)
            await get_or_synthesize_speech(app, faster, faster.text)
            await get_or_synthesize_speech(app, another_voice, another_voice.text)

        self.assertFalse(first[2])
        self.assertTrue(second[2])
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(synthesize.await_count, 3)

    async def test_speech_cache_write_failure_falls_back_to_live_audio(self):
        request = SpeakRequest(type="speak", text="继续播放。")
        pcm = bytes(3200)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("server_total.app.TTS_CACHE_DIR", Path(directory)),
            patch(
                "server_total.app.synthesize_speech",
                AsyncMock(return_value=(pcm, 0.1)),
            ),
            patch(
                "server_total.app._write_cached_pcm",
                side_effect=OSError("read only"),
            ),
        ):
            result = await get_or_synthesize_speech(app, request, request.text)

        self.assertEqual(result, (pcm, 0.1, False))

    async def test_prepare_speech_uses_playback_segmentation_and_reports_hits(self):
        prepare = AsyncMock(
            side_effect=[(bytes(3200), 0.1, False), (bytes(3200), 0.1, True)]
        )
        request = SpeechPrepareRequest(texts=["第一句。第二句。", "第一句。"])

        with patch("server_total.app.get_or_synthesize_speech", prepare):
            result = await prepare_speech(request)

        self.assertEqual(result["requested_texts"], 2)
        self.assertEqual(result["units"], 2)
        self.assertEqual(result["generated"], 1)
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(
            {call.args[2] for call in prepare.await_args_list},
            {"第一句。", "第二句。"},
        )

    async def test_streaming_uses_a_short_first_chunk_then_larger_chunks(self):
        class ChunkedResponse(FakeHttpResponse):
            content = bytes(56_000)

            async def aiter_bytes(self, chunk_size=None):
                del chunk_size
                for offset in range(0, len(self.content), 3_000):
                    yield self.content[offset : offset + 3_000]

        class ChunkedClient(FakeHttpClient):
            def stream(self, _method, url, **kwargs):
                self.request = (url, kwargs)
                return FakeStreamContext(ChunkedResponse())

        http = ChunkedClient()
        fake_app = SimpleNamespace(state=SimpleNamespace(http=http))
        request = AskRequest(type="ask", question="chunking")

        with (
            patch("server_total.app.TTS_STREAM_CHUNK_SECONDS", 0.5),
            patch("server_total.app.TTS_STREAM_STEADY_CHUNK_SECONDS", 1.0),
        ):
            chunks = [
                item async for item in stream_speech_chunks(fake_app, request, "text")
            ]

        self.assertEqual(
            [len(pcm) for pcm, _duration in chunks], [16_000, 32_000, 8_000]
        )
        self.assertEqual(
            [duration for _pcm, duration in chunks], [0.5, 1.0, 0.25]
        )

    async def test_synthesize_resamples_melotts_pcm_and_ignores_legacy_voice_id(self):
        class NativeRateResponse(FakeHttpResponse):
            content = bytes(4800)
            headers = {
                "X-Audio-Sample-Rate": "24000",
                "X-Audio-Sample-Format": "s16le",
            }

        class NativeRateClient(FakeHttpClient):
            def stream(self, _method, url, **kwargs):
                self.request = (url, kwargs)
                return FakeStreamContext(NativeRateResponse())

        http = NativeRateClient()
        fake_app = SimpleNamespace(state=SimpleNamespace(http=http))
        request = AskRequest(
            type="ask",
            question="你好",
            voice_id="customer_service_female",
        )

        pcm, duration = await synthesize_speech(fake_app, request, "采样率测试")

        self.assertEqual(len(pcm), 3200)
        self.assertEqual(duration, 0.1)
        self.assertNotIn("speaker_id", http.request[1]["data"])

    async def test_openvoice_switch_forwards_registered_clone_voice_id(self):
        http = FakeHttpClient()
        fake_app = SimpleNamespace(state=SimpleNamespace(http=http))
        request = AskRequest(
            type="ask",
            question="你好",
            voice_id="my-cloned-voice",
        )

        with patch("server_total.app.TTS_SERVICE", "openvoice"):
            await synthesize_speech(fake_app, request, "克隆音色测试")

        self.assertTrue(http.request[0].endswith("/v1/voice-clone"))
        self.assertEqual(http.request[1]["data"]["speaker_id"], "my-cloned-voice")

    async def test_pipeline_submits_each_melotts_text_unit_once_to_musetalk(self):
        upstream = FakeUpstream(
            [
                '{"type":"ready","fps":25,"backend":"torch","inference_dtype":"float32"}',
                '{"type":"started","profile":"business_male_1"}',
                '{"type":"queued","profile":"business_male_1"}',
                '{"type":"stream_start"}',
                media_packet(4, 0, b"first"),
                '{"type":"stream_end","packets":1}',
                '{"type":"started","profile":"business_male_1"}',
                '{"type":"queued","profile":"business_male_1"}',
                '{"type":"stream_start"}',
                media_packet(8, 0, b"second"),
                '{"type":"stream_end","packets":1}',
            ]
        )
        frontend = FakeFrontend()
        request = AskRequest(
            type="ask",
            question="介绍能力",
            request_id="request-1",
            profile="business_male_1",
            source_time_seconds=0.55,
        )
        pcm = bytes(3200)
        synthesized_texts = []

        async def synthesize_stream(_app, _request, text, **_kwargs):
            synthesized_texts.append(text)
            yield pcm, 0.1
            if len(synthesized_texts) == 1:
                yield pcm, 0.1

        from server_total import app as app_module

        previous_llm = (
            app_module.app.state.llm if hasattr(app_module.app.state, "llm") else None
        )
        app_module.app.state.llm = FakeLLM()
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch("server_total.app.TTS_CACHE_DIR", Path(directory)),
                patch("server_total.app.stream_speech_chunks", synthesize_stream),
                patch("server_total.app.websockets.connect", return_value=upstream),
            ):
                await run_pipeline(frontend, request, "request-1")
        finally:
            if previous_llm is None:
                del app_module.app.state.llm
            else:
                app_module.app.state.llm = previous_llm

        self.assertEqual(
            synthesized_texts,
            ["人工智能能够快速处理数据，", "识别图像。"],
        )
        text_units = [
            item for item in frontend.json_messages if item["type"] == "text_unit"
        ]
        self.assertEqual(
            [(item["seq"], item["text"], item["delimiter"]) for item in text_units],
            [
                (0, "人工智能能够快速处理数据，", "，"),
                (1, "识别图像。", "。"),
            ],
        )
        self.assertEqual([item["pts_us"] for item in text_units], [0, 200_000])
        for text_unit in text_units:
            text_index = frontend.json_messages.index(text_unit)
            next_message = frontend.json_messages[text_index + 1]
            self.assertEqual(next_message["type"], "stream_start")
            self.assertEqual(next_message["segment_seq"], text_unit["seq"])
        self.assertEqual(len(frontend.binary_messages), 2)
        ready = next(
            item for item in frontend.json_messages if item["type"] == "musetalk_ready"
        )
        self.assertEqual(ready["backend"], "torch")
        self.assertEqual(ready["inference_dtype"], "float32")
        first = PACKET_HEADER.unpack_from(frontend.binary_messages[0])
        second = PACKET_HEADER.unpack_from(frontend.binary_messages[1])
        self.assertEqual((first[4], first[6]), (0, 0))
        self.assertEqual((second[4], second[6]), (1, 200_000))
        self.assertEqual(frontend.json_messages[-1]["type"], "conversation_end")
        self.assertEqual(frontend.json_messages[-1]["units"], 2)
        commits = [
            item for item in upstream.sent if isinstance(item, str) and "commit" in item
        ]
        self.assertEqual(len(commits), 2)
        starts = [
            item for item in upstream.sent if isinstance(item, str) and "start" in item
        ]
        self.assertEqual(len(starts), 2)
        self.assertTrue(all('"profile": "business_male_1"' in item for item in starts))
        parsed_starts = [json.loads(item) for item in starts]
        self.assertTrue(all(item["start_position"] == 13 for item in parsed_starts))
        self.assertFalse(parsed_starts[0]["continue_from_previous"])
        self.assertTrue(all(item["continue_from_previous"] for item in parsed_starts[1:]))

    async def test_pipeline_does_not_deadlock_when_musetalk_fails(self):
        upstream = FakeUpstream(
            [
                '{"type":"ready","fps":25,"backend":"torch","inference_dtype":"float32"}',
                '{"type":"started","profile":"chinese"}',
                '{"type":"queued","profile":"chinese"}',
                '{"type":"error","message":"render failed"}',
            ]
        )
        frontend = FakeFrontend()
        request = AskRequest(type="ask", question="介绍能力", request_id="request-2")

        from server_total import app as app_module

        previous_llm = (
            app_module.app.state.llm if hasattr(app_module.app.state, "llm") else None
        )
        app_module.app.state.llm = FakeLLM()
        try:
            async def synthesize_stream(_app, _request, _text, **_kwargs):
                yield bytes(3200), 0.1

            with (
                tempfile.TemporaryDirectory() as directory,
                patch("server_total.app.TTS_CACHE_DIR", Path(directory)),
                patch(
                    "server_total.app.stream_speech_chunks",
                    synthesize_stream,
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

    async def test_idle_messages_do_not_start_musetalk_inference(self):
        websocket = FakeConversationSocket(
            [{"type": "idle_start", "profile": "chinese"}]
        )
        with patch("server_total.app.websockets.connect") as connect:
            await conversation(websocket)

        connect.assert_not_called()
        error = next(
            item for item in websocket.json_messages if item.get("type") == "error"
        )
        self.assertIn("unknown message type", error["message"])

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
