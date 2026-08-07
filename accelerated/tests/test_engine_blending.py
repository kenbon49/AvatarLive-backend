from __future__ import annotations

import unittest

import cv2
import numpy as np

from accelerated.engine import MuseTalkEngine


def reference_blend(source_rgb, generated_bgr, face_box, blend_mask):
    x1, y1, x2, y2 = face_box
    source_bgr = np.ascontiguousarray(source_rgb[:, :, ::-1])
    overlay = source_bgr.copy()
    overlay[y1:y2, x1:x2] = cv2.resize(
        generated_bgr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4
    )
    active = blend_mask > 0
    result = source_bgr.copy()
    alpha = blend_mask[active].astype(np.float32)[:, None] / 255.0
    values = (
        overlay[active].astype(np.float32) * alpha
        + source_bgr[active].astype(np.float32) * (1.0 - alpha)
    )
    result[active] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result[:, :, ::-1])


class BlendFaceTests(unittest.TestCase):
    def test_cropped_blend_matches_full_frame_reference(self):
        random = np.random.default_rng(7)
        source = random.integers(0, 256, (180, 120, 3), dtype=np.uint8)
        face = random.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((180, 120), dtype=np.uint8)
        mask[35:145, 12:108] = random.integers(
            0, 256, (110, 96), dtype=np.uint8
        )
        box = (28, 52, 94, 134)

        expected = reference_blend(source, face, box, mask)
        actual = MuseTalkEngine.blend_face(source, face, box, mask)

        np.testing.assert_array_equal(actual, expected)

    def test_empty_mask_returns_unchanged_copy(self):
        source = np.full((20, 16, 3), 91, dtype=np.uint8)
        face = np.zeros((256, 256, 3), dtype=np.uint8)

        actual = MuseTalkEngine.blend_face(
            source, face, (2, 3, 12, 18), np.zeros((20, 16), dtype=np.uint8)
        )

        np.testing.assert_array_equal(actual, source)
        self.assertIsNot(actual, source)


if __name__ == "__main__":
    unittest.main()
