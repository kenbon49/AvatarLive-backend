from __future__ import annotations

import unittest

from accelerated.server import _advance_profile_phase


class ServerTests(unittest.TestCase):
    def test_profile_phase_advances_by_audio_duration_across_chunks(self):
        phase = 3.0
        byte_counts = [16_000, 10_000, 2_912]

        for byte_count in byte_counts:
            phase = _advance_profile_phase(phase, byte_count, 15, 100)

        expected = 3.0 + sum(byte_counts) / 2 / 16000 * 15
        self.assertAlmostEqual(phase, expected)


if __name__ == "__main__":
    unittest.main()
