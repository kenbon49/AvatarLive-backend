"""Lifecycle and profile registry for the streaming MuseTalk service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
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
CUSTOM_PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
CUSTOM_AVATAR_STATUSES = {"review", "ready"}


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    cache_dir: Path
    public_avatar_dir: Path | None = None
    custom_avatar_dir: Path | None = None
    default_avatar: str = "chinese"
    fps: float = 25.0
    bbox_shift: int = 5
    detection_stride: int = 5
    max_frame_height: int = 0
    batch_size: int = 12
    device: str = "cuda:0"

    @classmethod
    def defaults(cls, root: str | Path | None = None) -> "RuntimeConfig":
        project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        return cls(
            root=project_root,
            cache_dir=project_root / "cache" / "accelerated",
            public_avatar_dir=project_root / "data" / "public",
            custom_avatar_dir=Path(
                os.getenv("MUSETALK_CUSTOM_AVATAR_DIR", project_root / "data" / "custom")
            ).expanduser().resolve(),
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
        self._avatar_metadata: dict[str, dict[str, object]] = {}
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
        self.static_avatar_ids = tuple(self.avatar_specs)
        self.loader = AvatarLoader(
            config.cache_dir,
            target_fps=config.fps,
            bbox_shift=config.bbox_shift,
            detection_stride=config.detection_stride,
            max_frame_height=config.max_frame_height,
        )

    def _custom_manifest_spec(
        self, manifest_path: Path
    ) -> tuple[AvatarSpec, dict[str, object]] | None:
        custom_root = (self.config.custom_avatar_dir or self.config.root / "data" / "custom").resolve()
        try:
            manifest_path = manifest_path.resolve(strict=True)
            manifest_path.relative_to(custom_root)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            log.warning("ignoring invalid custom avatar manifest %s", manifest_path)
            return None
        if not isinstance(payload, dict) or payload.get("status") not in CUSTOM_AVATAR_STATUSES:
            return None
        profile_id = str(payload.get("profile_id") or "")
        if not CUSTOM_PROFILE_PATTERN.fullmatch(profile_id) or profile_id in PUBLIC_AVATAR_FILES:
            return None
        source_name = str(payload.get("source_video") or "")
        if not source_name or Path(source_name).is_absolute():
            return None
        try:
            source_path = (manifest_path.parent / source_name).resolve(strict=True)
            source_path.relative_to(manifest_path.parent)
        except (OSError, ValueError):
            return None
        if not source_path.is_file():
            return None
        metadata = {
            "id": profile_id,
            "name": str(payload.get("name") or profile_id),
            "avatar_id": str(payload.get("avatar_id") or ""),
            "version": str(payload.get("version") or manifest_path.parent.name),
            "status": str(payload["status"]),
            "idle_video": str(payload.get("idle_video") or ""),
            "custom": True,
        }
        return AvatarSpec(profile_id, source_path, ping_pong=False), metadata

    def refresh_custom_avatar_specs(self) -> None:
        custom_root = (self.config.custom_avatar_dir or self.config.root / "data" / "custom").resolve()
        if not custom_root.is_dir():
            return
        discovered: dict[str, tuple[AvatarSpec, dict[str, object]]] = {}
        for manifest_path in custom_root.glob("avatars/*/*/manifest.json"):
            item = self._custom_manifest_spec(manifest_path)
            if item is None:
                continue
            spec, metadata = item
            discovered[spec.profile_id] = (spec, metadata)
        with self._profile_lock:
            for profile_id in set(self._avatar_metadata) - set(discovered):
                self.avatar_specs.pop(profile_id, None)
                self._profiles.pop(profile_id, None)
                self._avatar_metadata.pop(profile_id, None)
            for profile_id, (spec, metadata) in discovered.items():
                existing = self.avatar_specs.get(profile_id)
                if existing is not None and existing.video_path != spec.video_path:
                    self._profiles.pop(profile_id, None)
                self.avatar_specs[profile_id] = spec
                self._avatar_metadata[profile_id] = metadata

    def prepare_custom_profile(self, profile_id: str) -> AvatarProfile:
        if not CUSTOM_PROFILE_PATTERN.fullmatch(profile_id) or profile_id in PUBLIC_AVATAR_FILES:
            raise KeyError(f"unknown custom avatar {profile_id!r}")
        self.refresh_custom_avatar_specs()
        if profile_id not in self._avatar_metadata:
            raise KeyError(f"unknown custom avatar {profile_id!r}")
        profile = self.get_profile(profile_id)
        if self.engine is None:
            raise RuntimeError("MuseTalk runtime has not loaded its engine")
        self.engine.warm_up(profile.frames, fps=self.config.fps)
        return profile

    def is_profile_streamable(self, profile_id: str) -> bool:
        self.refresh_custom_avatar_specs()
        if profile_id not in self.avatar_specs:
            return False
        metadata = self._avatar_metadata.get(profile_id)
        return metadata is None or metadata.get("status") == "ready"

    def _profile_cache_key(self, spec: AvatarSpec) -> str:
        digest = hashlib.sha256(b"musetalk-v15-stream-profile-v3-float32\0")
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
            f"{self.config.fps}:max_height={self.config.max_frame_height}:"
            f"{self.config.bbox_shift}:{self.config.detection_stride}".encode("ascii")
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
            for profile_id in self.static_avatar_ids:
                profile = self.get_profile(profile_id)
                self.engine.warm_up(profile.frames, fps=self.config.fps)
                log.info("avatar %s is warmed up", profile_id)
            self.refresh_custom_avatar_specs()
            for profile_id, metadata in self._avatar_metadata.items():
                if metadata.get("status") != "ready":
                    continue
                profile = self.get_profile(profile_id)
                self.engine.warm_up(profile.frames, fps=self.config.fps)
                log.info("published custom avatar %s is warmed up", profile_id)
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
            self.refresh_custom_avatar_specs()
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
        self.refresh_custom_avatar_specs()
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
            "tf32": self.engine.allow_tf32 if self.engine is not None else None,
            "fps": self.config.fps,
            "batch_size": self.config.batch_size,
            "max_frame_height": self.config.max_frame_height,
            "model_devices": model_devices,
            "model_dtypes": model_dtypes,
            "gpu_ready": gpu_ready,
            "default_avatar": self.config.default_avatar,
            "avatars": [
                {
                    "id": profile_id,
                    "name": self._avatar_metadata.get(profile_id, {}).get("name", profile_id),
                    "default": profile_id == self.config.default_avatar,
                    "prepared": profile_id in self._profiles,
                    "source": str(spec.video_path),
                    "clip_start_seconds": spec.clip_start_seconds,
                    "custom": profile_id in self._avatar_metadata,
                    "status": self._avatar_metadata.get(profile_id, {}).get("status", "ready"),
                    "avatar_id": self._avatar_metadata.get(profile_id, {}).get("avatar_id", profile_id),
                    "version": self._avatar_metadata.get(profile_id, {}).get("version", ""),
                }
                for profile_id, spec in self.avatar_specs.items()
            ],
        }
