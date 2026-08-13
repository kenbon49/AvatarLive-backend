from __future__ import annotations

import unittest

import numpy as np

from accelerated.avatar import AvatarFrame, AvatarProfile, AvatarSpec
from accelerated.streaming import (
    PACKET_JPEG,
    StreamingRenderer,
    decode_packet,
    encode_packet,
    pcm16le_to_float32,
)


class FakeEngine:
    batch_size = 2

    def __init__(self):
        self.feature_inputs = []

    def extract_audio_features(self, pcm16k, fps):
        self.feature_inputs.append(pcm16k.copy())
        count = int(len(pcm16k) / 16000 * fps)
        return np.zeros((count, 50, 384), dtype=np.float32)

    def render_batch(self, audio_features, frames):
        return np.stack([frame.frame for frame in frames])


def make_profile() -> AvatarProfile:
    frames = [
        AvatarFrame(
            "american",
            "loop",
            index,
            np.full((2, 3, 3), index, dtype=np.uint8),
            (0, 0, 2, 2),
        )
        for index in range(3)
    ]
    return AvatarProfile(AvatarSpec("american", __file__), frames, 25)


class StreamingTests(unittest.TestCase):
    def test_packet_round_trip(self):
        encoded = encode_packet(PACKET_JPEG, 7, 120_000, b"image")
        self.assertEqual(decode_packet(encoded), (PACKET_JPEG, 7, 120_000, b"image"))

    def test_rejects_incomplete_pcm_sample(self):
        with self.assertRaisesRegex(ValueError, "complete 16-bit"):
            pcm16le_to_float32(b"\x00")

    def test_avatar_uses_ping_pong_cycle(self):
        profile = make_profile()
        self.assertEqual([item.frame_index for item in profile.sequence(0, 7)], [0, 1, 2, 1, 0, 1, 2])

    def test_avatar_can_repeat_directly_from_the_first_frame(self):
        profile = make_profile()
        profile.spec = AvatarSpec("chinese", __file__, clip_end_seconds=2.0, ping_pong=False)
        self.assertEqual([item.frame_index for item in profile.sequence(0, 7)], [0, 1, 2, 0, 1, 2, 0])
        self.assertEqual(profile.cycle_length, 3)

    def test_renderer_yields_fixed_batches_and_matching_audio(self):
        profile = make_profile()
        engine = FakeEngine()
        renderer = StreamingRenderer(engine, fps=25)
        pcm = np.zeros(4 * 640, dtype="<i2").tobytes()
        batches = list(renderer.render(pcm, profile))
        self.assertEqual([len(batch.frames) for batch in batches], [2, 2])
        self.assertEqual([len(batch.pcm16) for batch in batches], [2 * 640 * 2, 2 * 640 * 2])
        self.assertEqual([batch.start_frame for batch in batches], [0, 2])
        self.assertEqual(len(engine.feature_inputs), 1)
        self.assertFalse(np.any(engine.feature_inputs[0]))

    def test_renderer_does_not_carry_position_between_requests(self):
        profile = make_profile()
        engine = FakeEngine()
        renderer = StreamingRenderer(engine, fps=25)
        pcm = np.zeros(2 * 640, dtype="<i2").tobytes()

        first = list(renderer.render(pcm, profile))
        second = list(renderer.render(pcm, profile))

        self.assertEqual(first[0].frames[0, 0, 0, 0], 0)
        self.assertEqual(second[0].frames[0, 0, 0, 0], 0)

    def test_renderer_uses_the_session_position_requested_by_the_server(self):
        profile = make_profile()
        engine = FakeEngine()
        renderer = StreamingRenderer(engine, fps=25)
        pcm = np.zeros(2 * 640, dtype="<i2").tobytes()

        batches = list(renderer.render(pcm, profile, start_position=2))

        self.assertEqual(batches[0].frames[0, 0, 0, 0], 2)


if __name__ == "__main__":
    unittest.main()
