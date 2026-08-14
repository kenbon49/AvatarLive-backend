from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from accelerated.avatar import AvatarLoader
from accelerated.runtime import MuseTalkRuntime, PUBLIC_AVATAR_FILES, RuntimeConfig


class RuntimeTests(unittest.TestCase):
    def test_avatar_loader_limits_large_frames_without_upscaling(self):
        loader = AvatarLoader("cache", max_frame_height=720)

        large = np.zeros((1280, 720, 3), dtype=np.uint8)
        small = np.zeros((720, 406, 3), dtype=np.uint8)

        self.assertEqual(loader._resize_frame(large).shape, (720, 405, 3))
        self.assertIs(loader._resize_frame(small), small)

    def test_public_avatar_registry_uses_the_public_directory(self):
        root = Path("project-root").resolve()
        runtime = MuseTalkRuntime(RuntimeConfig.defaults(root))

        self.assertEqual(
            list(runtime.avatar_specs),
            ["chinese", "business_male_1", "chen_yu"],
        )
        for profile_id, filename in PUBLIC_AVATAR_FILES.items():
            self.assertEqual(
                runtime.avatar_specs[profile_id].video_path,
                root / "data" / "public" / filename,
            )
            # Public avatars must include the video's first decoded frame.
            self.assertEqual(runtime.avatar_specs[profile_id].clip_start_seconds, 0.0)

    @patch("accelerated.runtime.MuseTalkEngine")
    def test_initialize_prepares_and_warms_every_public_avatar(self, engine_class):
        with tempfile.TemporaryDirectory() as directory:
            runtime = MuseTalkRuntime(RuntimeConfig.defaults(directory))
            profiles = {
                profile_id: SimpleNamespace(frames=[profile_id])
                for profile_id in runtime.avatar_specs
            }
            with patch.object(
                runtime,
                "get_profile",
                side_effect=lambda profile_id: profiles[profile_id],
            ) as get_profile:
                runtime.initialize()

        self.assertEqual(
            [call.args[0] for call in get_profile.call_args_list],
            ["chinese", "business_male_1", "chen_yu"],
        )
        self.assertEqual(
            [call.args[0] for call in engine_class.return_value.warm_up.call_args_list],
            [["chinese"], ["business_male_1"], ["chen_yu"]],
        )
        self.assertTrue(runtime.ready)


if __name__ == "__main__":
    unittest.main()
