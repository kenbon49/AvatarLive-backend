"""Lifecycle and profile registry for the streaming MuseTalk service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import threading

from .avatar import AvatarLoader, AvatarProfile, AvatarSpec
from .engine import MuseTalkEngine
from .streaming import StreamingRenderer


log = logging.getLogger(__name__)

PUBLIC_AVATAR_FILES = {
    "chinese": "chinese2.mp4",
    "business_male_1": "商务男确定.mp4",
    "chen_yu": "陈屿.mp4",
}


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    cache_dir: Path
    public_avatar_dir: Path | None = None
    default_avatar: str = "chinese"
    fps: float = 25.0
    bbox_shift: int = 5
    detection_stride: int = 5
    batch_size: int = 1
    device: str = "cuda:0"

    @classmethod
    def defaults(cls, root: str | Path | None = None) -> "RuntimeConfig":
        project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        return cls(
            root=project_root,
            cache_dir=project_root / "cache" / "accelerated",
            public_avatar_dir=project_root / "data" / "public",
        )


class MuseTalkRuntime:
    """Own the single model instance and lazily prepared avatar profiles."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        public_avatar_dir = config.public_avatar_dir or config.root / "data" / "public"
        self.engine: MuseTalkEngine | None = None
        self.renderer: StreamingRenderer | None = None
        self._profiles: dict[str, AvatarProfile] = {}
        self._profile_lock = threading.RLock()
        self.ready = False
        self.initialization_error: str | None = None
        self.avatar_specs = {
            profile_id: AvatarSpec(
                profile_id,
                public_avatar_dir / filename,
                ping_pong=False,
            )
            for profile_id, filename in PUBLIC_AVATAR_FILES.items()
        }
        self.loader = AvatarLoader(
            config.cache_dir,
            target_fps=config.fps,
            bbox_shift=config.bbox_shift,
            detection_stride=config.detection_stride,
        )

    def _profile_cache_key(self, spec: AvatarSpec) -> str:
        digest = hashlib.sha256(b"musetalk-v15-stream-profile-v2-float32\0")
        video = spec.video_path.resolve(strict=True)
        stat = video.stat()
        digest.update(f"{video}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        for relative in (
            "models/musetalkV15/unet.pth",
            "models/musetalkV15/musetalk.json",
            "models/sd-vae/diffusion_pytorch_model.bin",
            "models/face-parse-bisent/79999_iter.pth",
        ):
            path = self.config.root / relative
            model_stat = path.stat()
            digest.update(f"{relative}:{model_stat.st_size}:{model_stat.st_mtime_ns}".encode("utf-8"))
        digest.update(
            f"{self.config.fps}:preserve_source_resolution:{self.config.bbox_shift}:"
            f"{self.config.detection_stride}".encode("ascii")
        )
        digest.update(
            f":{spec.clip_start_seconds}:{spec.clip_end_seconds}:{spec.ping_pong}".encode("ascii")
        )
        return digest.hexdigest()

    def initialize(self) -> None:
        """Load models and eagerly prepare every selectable public avatar."""

        try:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            self.engine = MuseTalkEngine(
                self.config.root,
                cache_path=self.config.cache_dir / "unused.npz",
                batch_size=self.config.batch_size,
                extra_margin=8,
                parsing_mode="jaw",
                left_cheek_width=60,
                right_cheek_width=60,
                device=self.config.device,
            )
            self.renderer = StreamingRenderer(self.engine, fps=self.config.fps)
            for profile_id in self.avatar_specs:
                profile = self.get_profile(profile_id)
                self.engine.warm_up(profile.frames, fps=self.config.fps)
                log.info("avatar %s is warmed up", profile_id)
            self.ready = True
            log.info(
                "streaming runtime ready; default avatar=%s; prepared avatars=%s",
                self.config.default_avatar,
                ",".join(self.avatar_specs),
            )
        except Exception as exc:
            self.initialization_error = str(exc)
            log.exception("failed to initialize accelerated MuseTalk runtime")
            raise

    def get_profile(self, profile_id: str) -> AvatarProfile:
        if profile_id not in self.avatar_specs:
            raise KeyError(f"unknown avatar {profile_id!r}")
        if self.engine is None:
            raise RuntimeError("MuseTalk runtime has not loaded its engine")
        with self._profile_lock:
            existing = self._profiles.get(profile_id)
            if existing is not None:
                return existing
            spec = self.avatar_specs[profile_id]
            profile = self.loader.load(spec)
            self.engine.prepare_frames(
                profile.frames,
                cache_key=self._profile_cache_key(spec),
                cache_path=self.config.cache_dir / f"{profile_id}-musetalk-v15.npz",
            )
            self._profiles[profile_id] = profile
            log.info("avatar %s is ready with %d frames", profile_id, len(profile.frames))
            return profile

    def status(self) -> dict[str, object]:
        model_devices: dict[str, str] = {}
        model_dtypes: dict[str, str] = {}
        if self.engine is not None:
            model_devices = {
                "unet": str(next(self.engine.unet.model.parameters()).device),
                "vae": str(next(self.engine.vae.vae.parameters()).device),
                "whisper": str(next(self.engine.whisper.parameters()).device),
            }
            model_dtypes = {
                "unet": str(next(self.engine.unet.model.parameters()).dtype).removeprefix("torch."),
                "vae": str(next(self.engine.vae.vae.parameters()).dtype).removeprefix("torch."),
                "whisper": str(next(self.engine.whisper.parameters()).dtype).removeprefix("torch."),
            }
        gpu_ready = bool(model_devices) and all(
            device.startswith("cuda") for device in model_devices.values()
        )
        return {
            "ready": self.ready,
            "error": self.initialization_error,
            "model_version": "1.5",
            "device": self.config.device,
            "backend": "torch",
            "inference_dtype": MuseTalkEngine.inference_dtype,
            "fps": self.config.fps,
            "batch_size": self.config.batch_size,
            "model_devices": model_devices,
            "model_dtypes": model_dtypes,
            "gpu_ready": gpu_ready,
            "default_avatar": self.config.default_avatar,
            "avatars": [
                {
                    "id": profile_id,
                    "default": profile_id == self.config.default_avatar,
                    "prepared": profile_id in self._profiles,
                    "source": str(spec.video_path),
                    "clip_start_seconds": spec.clip_start_seconds,
                }
                for profile_id, spec in self.avatar_specs.items()
            ],
        }
