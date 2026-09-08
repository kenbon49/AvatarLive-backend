from __future__ import annotations

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image

from server_total.app import (
    VideoPrepareRequest,
    _prepared_video_items,
    _prepared_video_key,
)
from server_total.prepared_video import (
    artifact_paths,
    cache_key,
    encode_transparent_webm,
    ffmpeg_executable,
    jpeg_to_background_removed_rgba,
    read_manifest,
    write_manifest,
)


def jpeg_frame(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        output, format="JPEG", quality=100, subsampling=0
    )
    return output.getvalue()


class PreparedVideoTests(unittest.IsolatedAsyncioTestCase):
    def test_cache_key_is_canonical_and_configuration_specific(self):
        self.assertEqual(
            cache_key({"text": "hello", "speed": 1}),
            cache_key({"speed": 1, "text": "hello"}),
        )
        request = VideoPrepareRequest(
            texts=["固定话术"], profile="chinese", voice_id="voice-a", speed=1
        )
        changed_voice = VideoPrepareRequest(
            texts=["固定话术"], profile="chinese", voice_id="voice-b", speed=1
        )
        self.assertNotEqual(
            _prepared_video_key(request, request.texts[0], "avatar-v1"),
            _prepared_video_key(changed_voice, changed_voice.texts[0], "avatar-v1"),
        )
        self.assertNotEqual(
            _prepared_video_key(request, request.texts[0], "avatar-v1"),
            _prepared_video_key(request, request.texts[0], "avatar-v2"),
        )

    def test_removes_edge_connected_background_but_preserves_enclosed_white(self):
        image = np.full((96, 96, 3), 250, dtype=np.uint8)
        image[24:72, 24:72] = (30, 45, 60)
        image[36:60, 36:60] = (250, 250, 250)

        rgba_bytes, width, height = jpeg_to_background_removed_rgba(jpeg_frame(image))
        rgba = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)

        self.assertLess(rgba[2, 2, 3], 10)
        self.assertGreater(rgba[48, 48, 3], 245)
        self.assertGreater(rgba[28, 28, 3], 245)

    def test_manifest_requires_both_artifacts(self):
        key = cache_key({"text": "manifest"})
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            video_path, _manifest_path = artifact_paths(cache_dir, key)
            video_path.parent.mkdir(parents=True)
            video_path.write_bytes(bytes(2048))
            write_manifest(cache_dir, key, {"duration_seconds": 1.25})
            self.assertEqual(read_manifest(cache_dir, key)["duration_seconds"], 1.25)
            video_path.unlink()
            self.assertIsNone(read_manifest(cache_dir, key))

    async def test_prepare_items_reports_ready_and_schedules_only_missing(self):
        request = VideoPrepareRequest(texts=["已经生成", "等待生成"])

        def info(index, key):
            if index == 0:
                return {"index": index, "key": key, "status": "ready", "url": "/ready.webm"}
            return {"index": index, "key": key, "status": "missing"}

        with (
            patch(
                "server_total.app._prepared_video_profile_version",
                AsyncMock(return_value="avatar-v1"),
            ),
            patch("server_total.app._prepared_video_info", side_effect=info),
            patch("server_total.app._schedule_prepared_video") as schedule,
        ):
            items = await _prepared_video_items(request, schedule_missing=True)

        self.assertEqual([item["status"] for item in items], ["ready", "preparing"])
        schedule.assert_called_once()
        self.assertEqual(schedule.call_args.args[2], "等待生成")

    def test_encoder_produces_playable_vp9_video_with_alpha(self):
        frames = []
        for offset in range(3):
            image = np.full((96, 128, 3), 250, dtype=np.uint8)
            image[20:86, 38 + offset : 90 + offset] = (25, 90, 150)
            frames.append(jpeg_frame(image))
        pcm = bytes(2 * 16_000)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.webm"
            metadata = encode_transparent_webm(frames, pcm, 3, output)
            decoded = subprocess.run(
                [
                    ffmpeg_executable(),
                    "-v",
                    "error",
                    "-c:v",
                    "libvpx-vp9",
                    "-i",
                    str(output),
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
            rgba = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(96, 128, 4)

        self.assertTrue(metadata["background_removed"])
        self.assertLess(rgba[2, 2, 3], 20)
        self.assertGreater(rgba[50, 64, 3], 235)


if __name__ == "__main__":
    unittest.main()
