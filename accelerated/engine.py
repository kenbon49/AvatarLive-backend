"""MuseTalk 1.5 inference over pre-recorded action frames.

The target action frame is the single geometry source. MuseTalk only redraws
the parsed lower-face area, so hair, head pose, neck, body, and hands remain
pixel-identical outside the blend mask.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


log = logging.getLogger(__name__)

RGBFrame = NDArray[np.uint8]
FrameKey = tuple[str, str, int]
_LEGACY_PROFILE_ID = "__legacy__"


class MuseTalkSetupError(RuntimeError):
    """Raised when models, frames, or the preprocessing cache are invalid."""


@dataclass(frozen=True)
class PreparedFrame:
    """Model material associated with one immutable target action frame."""

    profile_id: str
    action: str
    frame_index: int
    frame: RGBFrame
    face_box: tuple[int, int, int, int]
    latent: Any
    blend_mask: NDArray[np.uint8]


def build_cache_key(manifest_path: str | Path, model_root: str | Path) -> str:
    """Build a cheap cache key from all source assets and model metadata."""

    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    root = Path(model_root).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(b"synlive-musetalk-v15-cache-v3\0")
    digest.update(manifest.read_bytes())
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in parsed.get("actions", {}).values():
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            continue
        asset = (manifest.parent / entry["file"]).resolve(strict=True)
        stat = asset.stat()
        digest.update(str(asset).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    for relative in (
        "models/musetalkV15/unet.pth",
        "models/musetalkV15/musetalk.json",
        "models/sd-vae/diffusion_pytorch_model.bin",
        "models/whisper/pytorch_model.bin",
        "models/face-parse-bisent/79999_iter.pth",
        "models/face-parse-bisent/resnet18-5c106cde.pth",
    ):
        path = root / relative
        stat = path.stat()
        digest.update(relative.encode("ascii"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


class MuseTalkEngine:
    """Load MuseTalk 1.5 and render mouth-only RGB frames."""

    model_version = "1.5"

    def __init__(
        self,
        model_root: str | Path,
        *,
        cache_path: str | Path,
        batch_size: int = 1,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        upper_boundary_ratio: float = 0.55,
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        audio_padding_left: int = 2,
        audio_padding_right: int = 2,
        device: str = "cuda:0",
        backend: str = "onnx",
        onnx_dir: str | Path | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if extra_margin < 0:
            raise ValueError("extra_margin must not be negative")
        if parsing_mode not in {"raw", "jaw", "neck"}:
            raise ValueError(f"unsupported parsing mode: {parsing_mode}")
        if not 0.0 <= upper_boundary_ratio <= 1.0:
            raise ValueError("upper_boundary_ratio must be between 0 and 1")
        if backend not in {"torch", "onnx"}:
            raise ValueError(f"unsupported inference backend: {backend}")

        self.root = Path(model_root).expanduser().resolve(strict=True)
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.batch_size = int(batch_size)
        self._blend_executor = ThreadPoolExecutor(
            max_workers=min(self.batch_size, 4),
            thread_name_prefix="musetalk-blend",
        )
        self.extra_margin = int(extra_margin)
        self.parsing_mode = parsing_mode
        self.upper_boundary_ratio = float(upper_boundary_ratio)
        self.left_cheek_width = int(left_cheek_width)
        self.right_cheek_width = int(right_cheek_width)
        self.audio_padding_left = int(audio_padding_left)
        self.audio_padding_right = int(audio_padding_right)
        self.device_name = device
        self.backend = backend
        self.onnx_dir = Path(onnx_dir or Path(model_root) / "models/onnx").resolve()
        self._prepared: dict[FrameKey, PreparedFrame] = {}
        self._lock = threading.RLock()
        self._load_models()

    def _require_file(self, relative: str) -> Path:
        path = (self.root / relative).resolve(strict=True)
        if not path.is_file():
            raise MuseTalkSetupError(f"required model file is missing: {path}")
        return path

    def _load_models(self) -> None:
        import torch
        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor

        if not torch.cuda.is_available():
            raise MuseTalkSetupError("MuseTalk requires CUDA")
        self.torch = torch
        self.device = torch.device(self.device_name)
        # Fixed 256x256 batches are faster and avoid multi-second first-shape
        # autotune stalls on RTX 3090 with the current CUDA/PyTorch stack.
        torch.backends.cudnn.benchmark = False
        vae_dir = self.root / "models/sd-vae"
        whisper_dir = self.root / "models/whisper"
        log.info("loading MuseTalk 1.5 %s backend from %s", self.backend, self.root)
        self.vae = VAE(model_path=str(vae_dir))
        self.pe = PositionalEncoding(d_model=384)
        self.vae.vae = self.vae.vae.to(device=self.device, dtype=torch.float16).eval()
        self.vae.vae.requires_grad_(False)
        self.pe = self.pe.to(device=self.device, dtype=torch.float16).eval()
        self.weight_dtype = torch.float16
        self.timesteps = torch.tensor([0], device=self.device)
        self.audio_processor = AudioProcessor(feature_extractor_path=str(whisper_dir))

        if self.backend == "onnx":
            from musetalk.onnx_inference import (
                OnnxUNet,
                OnnxVAEDecoder,
                OnnxWhisperEncoder,
            )

            unet = OnnxUNet(self.onnx_dir / "musetalk_unet.onnx", self.device)
            self.unet = SimpleNamespace(model=unet)
            self.decoder = OnnxVAEDecoder(
                self.onnx_dir / "musetalk_vae_decoder.onnx", self.device
            )
            for name, exported_batch in (
                ("UNet", unet.batch_size),
                ("VAE Decoder", self.decoder.batch_size),
            ):
                if isinstance(exported_batch, int) and exported_batch != self.batch_size:
                    raise MuseTalkSetupError(
                        f"ONNX {name} was exported for batch {exported_batch}, but the "
                        f"service uses batch {self.batch_size}; re-export or change "
                        "--batch-size"
                    )
            self.whisper = SimpleNamespace(
                encoder=OnnxWhisperEncoder(
                    self.onnx_dir / "musetalk_whisper_encoder.onnx", self.device
                )
            )
        else:
            from transformers import WhisperModel

            unet_config = self._require_file("models/musetalkV15/musetalk.json")
            unet_weights = self._require_file("models/musetalkV15/unet.pth")
            self.unet = UNet(
                unet_config=str(unet_config),
                model_path=str(unet_weights),
                device=self.device,
            )
            self.unet.model = self.unet.model.to(
                device=self.device, dtype=self.weight_dtype
            ).eval()
            self.unet.model.requires_grad_(False)
            self.decoder = self.vae
            self.whisper = WhisperModel.from_pretrained(str(whisper_dir))
            self.whisper = self.whisper.to(
                device=self.device, dtype=self.weight_dtype
            ).eval()
            self.whisper.requires_grad_(False)
        log.info("MuseTalk 1.5 %s model load complete", self.backend)

    def _load_face_parser(self):
        """Load trusted legacy parser weights under PyTorch 2.6+ safely.

        The upstream ResNet checkpoint uses PyTorch's legacy tar container and
        therefore cannot be read with the new weights_only default. We decode
        that one known local checkpoint explicitly, then rewrite only its state
        dict into the modern tensor-only format consumed by upstream BiSeNet.
        """

        torch = self.torch
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.face_parsing.model import BiSeNet

        legacy_resnet = self._require_file(
            "models/face-parse-bisent/resnet18-5c106cde.pth"
        )
        parser_weights = self._require_file("models/face-parse-bisent/79999_iter.pth")
        converted = Path(tempfile.gettempdir()) / (
            "synlive-musetalk-resnet18-" + hashlib.sha256(
                f"{legacy_resnet.stat().st_size}:{legacy_resnet.stat().st_mtime_ns}".encode(
                    "ascii"
                )
            ).hexdigest()[:16] + ".pth"
        )
        if not converted.is_file():
            state = torch.load(
                legacy_resnet,
                map_location="cpu",
                weights_only=False,
            )
            torch.save(state, converted)

        device = self.device

        class CompatibleFaceParsing(FaceParsing):
            def model_init(inner_self):
                net = BiSeNet(str(converted))
                state = torch.load(
                    parser_weights,
                    map_location="cpu",
                    weights_only=True,
                )
                net.load_state_dict(state)
                return net.to(device).eval()

        return CompatibleFaceParsing(
            left_cheek_width=self.left_cheek_width,
            right_cheek_width=self.right_cheek_width,
        )

    @staticmethod
    def _profile_id(frame: Any) -> str:
        if not hasattr(frame, "profile_id"):
            return _LEGACY_PROFILE_ID
        profile_id = frame.profile_id
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise MuseTalkSetupError("frame profile_id must be a non-empty string")
        return profile_id.strip()

    @classmethod
    def _frame_key(cls, frame: Any) -> FrameKey:
        return cls._profile_id(frame), str(frame.action), int(frame.frame_index)

    @classmethod
    def _single_profile_id(cls, frames: Sequence[Any]) -> str:
        profiles = {cls._profile_id(frame) for frame in frames}
        if len(profiles) != 1:
            raise MuseTalkSetupError(
                "prepare each MuseTalk profile separately with its own cache path"
            )
        return next(iter(profiles))

    def _install_prepared(
        self,
        profile_id: str,
        prepared: dict[FrameKey, PreparedFrame],
    ) -> None:
        if any(key[0] != profile_id for key in prepared):
            raise MuseTalkSetupError("prepared frame profile does not match the batch")
        with self._lock:
            merged = {
                key: value
                for key, value in self._prepared.items()
                if key[0] != profile_id
            }
            merged.update(prepared)
            self._prepared = merged

    def _resolve_cache_path(self, cache_path: str | Path | None) -> Path:
        if cache_path is None:
            return self.cache_path
        return Path(cache_path).expanduser().resolve()

    def _normalize_box(self, frame: Any) -> tuple[int, int, int, int]:
        height, width = frame.frame.shape[:2]
        values = np.asarray(frame.face_box, dtype=np.float32).reshape(4)
        if not np.isfinite(values).all():
            raise MuseTalkSetupError(f"non-finite face box for {self._frame_key(frame)}")
        x1 = max(0, min(width - 2, int(math.floor(float(values[0])))))
        y1 = max(0, min(height - 2, int(math.floor(float(values[1])))))
        x2 = max(x1 + 2, min(width, int(math.ceil(float(values[2])))))
        y2 = max(
            y1 + 2,
            min(height, int(math.ceil(float(values[3]))) + self.extra_margin),
        )
        if x2 - x1 < 16 or y2 - y1 < 16:
            raise MuseTalkSetupError(f"face box is too small for {self._frame_key(frame)}")
        return x1, y1, x2, y2

    @staticmethod
    def _expand_mask_to_frame(
        mask: NDArray[np.uint8],
        crop_box: Sequence[int],
        shape: tuple[int, int],
    ) -> NDArray[np.uint8]:
        height, width = shape
        full = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = (int(value) for value in crop_box)
        dst_x1, dst_y1 = max(0, x1), max(0, y1)
        dst_x2, dst_y2 = min(width, x2), min(height, y2)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return full
        src_x1, src_y1 = dst_x1 - x1, dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        full[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
        return full

    def prepare_frames(
        self,
        frames: Iterable[Any],
        *,
        cache_key: str,
        cache_path: str | Path | None = None,
    ) -> None:
        source_frames = list(frames)
        if not source_frames:
            raise MuseTalkSetupError("action runtime contains no frames")
        profile_id = self._single_profile_id(source_frames)
        resolved_cache_path = self._resolve_cache_path(cache_path)
        keys = [self._frame_key(frame) for frame in source_frames]
        if len(set(keys)) != len(keys):
            raise MuseTalkSetupError("action runtime contains duplicate frame keys")
        shape = tuple(int(value) for value in source_frames[0].frame.shape[:2])
        for frame in source_frames:
            if frame.frame.dtype != np.uint8 or frame.frame.shape != (*shape, 3):
                raise MuseTalkSetupError("all action frames must be same-size uint8 RGB")

        if self._try_load_cache(
            source_frames,
            cache_key,
            resolved_cache_path,
            profile_id,
        ):
            return

        from musetalk.utils.blending import get_image_prepare_material

        face_parser = self._load_face_parser()
        prepared: list[PreparedFrame] = []
        total = len(source_frames)
        log.info("precomputing MuseTalk material for %s action frames", total)
        try:
            with self.torch.inference_mode():
                for position, frame in enumerate(source_frames, start=1):
                    box = self._normalize_box(frame)
                    x1, y1, x2, y2 = box
                    bgr = np.ascontiguousarray(frame.frame[:, :, ::-1])
                    crop = bgr[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latent = self.vae.get_latents_for_unet(crop)
                    latent = latent.detach().to(device="cpu", dtype=self.torch.float16)[0]
                    raw_mask, crop_box = get_image_prepare_material(
                        bgr,
                        box,
                        upper_boundary_ratio=self.upper_boundary_ratio,
                        mode=self.parsing_mode,
                        fp=face_parser,
                    )
                    full_mask = self._expand_mask_to_frame(
                        np.asarray(raw_mask, dtype=np.uint8),
                        crop_box,
                        shape,
                    )
                    if not np.any(full_mask):
                        raise MuseTalkSetupError(
                            f"face parser returned an empty mask for {self._frame_key(frame)}"
                        )
                    prepared.append(
                        PreparedFrame(
                            profile_id=profile_id,
                            action=str(frame.action),
                            frame_index=int(frame.frame_index),
                            frame=frame.frame,
                            face_box=box,
                            latent=latent,
                            blend_mask=full_mask,
                        )
                    )
                    if position == 1 or position % 25 == 0 or position == total:
                        log.info("MuseTalk frame preparation %s/%s", position, total)
        finally:
            del face_parser
            gc.collect()
            self.torch.cuda.empty_cache()

        prepared_by_key = {
            (item.profile_id, item.action, item.frame_index): item
            for item in prepared
        }
        self._install_prepared(profile_id, prepared_by_key)
        self._save_cache(prepared, cache_key, resolved_cache_path)

    def _cache_metadata(self, cache_key: str, count: int, shape: Sequence[int]) -> dict[str, Any]:
        return {
            "format": 2,
            "cache_key": cache_key,
            "model_version": self.model_version,
            "count": count,
            "shape": list(shape),
            "extra_margin": self.extra_margin,
            "parsing_mode": self.parsing_mode,
            "upper_boundary_ratio": self.upper_boundary_ratio,
            "left_cheek_width": self.left_cheek_width,
            "right_cheek_width": self.right_cheek_width,
        }

    def _try_load_cache(
        self,
        frames: Sequence[Any],
        cache_key: str,
        cache_path: Path,
        profile_id: str,
    ) -> bool:
        if not cache_path.is_file():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata"].item()))
                expected = self._cache_metadata(
                    cache_key,
                    len(frames),
                    frames[0].frame.shape[:2],
                )
                if metadata != expected:
                    log.info("ignoring stale MuseTalk frame cache: %s", cache_path)
                    return False
                actions = cache["actions"].tolist()
                indices = cache["indices"].astype(np.int64, copy=False).tolist()
                boxes = cache["boxes"].astype(np.int32, copy=False)
                latents = cache["latents"].astype(np.float16, copy=False)
                masks = cache["masks"].astype(np.uint8, copy=False)
                if not (
                    len(actions)
                    == len(indices)
                    == len(boxes)
                    == len(latents)
                    == len(masks)
                    == len(frames)
                ):
                    return False
                prepared: dict[FrameKey, PreparedFrame] = {}
                for pos, frame in enumerate(frames):
                    frame_key = self._frame_key(frame)
                    stored_key = (str(actions[pos]), int(indices[pos]))
                    if stored_key != frame_key[1:]:
                        return False
                    prepared[frame_key] = PreparedFrame(
                        profile_id=profile_id,
                        action=frame_key[1],
                        frame_index=frame_key[2],
                        frame=frame.frame,
                        face_box=tuple(int(value) for value in boxes[pos]),
                        latent=self.torch.from_numpy(latents[pos].copy()),
                        blend_mask=masks[pos].copy(),
                    )
            self._install_prepared(profile_id, prepared)
            log.info(
                "loaded %s precomputed MuseTalk frames for %s from %s",
                len(prepared),
                profile_id,
                cache_path,
            )
            return True
        except Exception:
            log.exception("failed to load MuseTalk cache; rebuilding it")
            return False

    def _save_cache(
        self,
        prepared: Sequence[PreparedFrame],
        cache_key: str,
        cache_path: Path,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._cache_metadata(
            cache_key,
            len(prepared),
            prepared[0].frame.shape[:2],
        )
        actions = np.asarray([item.action for item in prepared])
        indices = np.asarray([item.frame_index for item in prepared], dtype=np.int32)
        boxes = np.asarray([item.face_box for item in prepared], dtype=np.int32)
        latents = np.stack([item.latent.numpy() for item in prepared]).astype(np.float16)
        masks = np.stack([item.blend_mask for item in prepared]).astype(np.uint8)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                actions=actions,
                indices=indices,
                boxes=boxes,
                latents=latents,
                masks=masks,
            )
        os.replace(temporary, cache_path)
        log.info("saved MuseTalk frame cache to %s", cache_path)

    def get_prepared(self, frame: Any) -> PreparedFrame:
        key = self._frame_key(frame)
        with self._lock:
            try:
                return self._prepared[key]
            except KeyError as exc:
                raise MuseTalkSetupError(f"frame was not precomputed: {key}") from exc

    def extract_audio_features(self, pcm16k: NDArray[np.float32], fps: float):
        """Return one CPU Whisper prompt per output video frame."""

        pcm = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        frame_count = math.floor(len(pcm) / 16000.0 * int(fps))
        if frame_count <= 0:
            return self.torch.empty((0, 50, 384), dtype=self.weight_dtype)
        segment_size = 30 * 16000
        features = []
        with self._lock, self.torch.inference_mode():
            for start in range(0, len(pcm), segment_size):
                segment = pcm[start : start + segment_size]
                feature = self.audio_processor.feature_extractor(
                    segment,
                    return_tensors="pt",
                    sampling_rate=16000,
                ).input_features
                features.append(feature.to(dtype=self.weight_dtype))
            chunks = self.audio_processor.get_whisper_chunk(
                features,
                self.device,
                self.weight_dtype,
                self.whisper,
                len(pcm),
                fps=fps,
                audio_padding_length_left=self.audio_padding_left,
                audio_padding_length_right=self.audio_padding_right,
            )
        return chunks.detach().cpu()

    @staticmethod
    def blend_face(
        source_rgb: RGBFrame,
        generated_bgr: RGBFrame,
        face_box: Sequence[int],
        blend_mask: NDArray[np.uint8],
    ) -> RGBFrame:
        """Blend a generated BGR face while preserving every zero-mask pixel."""

        x1, y1, x2, y2 = (int(value) for value in face_box)
        resized = cv2.resize(
            generated_bgr.astype(np.uint8, copy=False),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_LANCZOS4,
        )
        result = source_rgb.copy()
        nonzero = cv2.findNonZero(blend_mask)
        if nonzero is None:
            return result

        # The prepared mask only covers the face crop. Blending that rectangle avoids
        # three full-frame copies and a full-frame boolean gather for every output frame.
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(nonzero)
        mask_x2, mask_y2 = mask_x + mask_w, mask_y + mask_h
        source_roi = source_rgb[mask_y:mask_y2, mask_x:mask_x2]
        overlay_roi = source_roi.copy()

        paste_x1, paste_y1 = max(x1, mask_x), max(y1, mask_y)
        paste_x2, paste_y2 = min(x2, mask_x2), min(y2, mask_y2)
        if paste_x2 > paste_x1 and paste_y2 > paste_y1:
            face_x1, face_y1 = paste_x1 - x1, paste_y1 - y1
            face_x2, face_y2 = paste_x2 - x1, paste_y2 - y1
            overlay_roi[
                paste_y1 - mask_y : paste_y2 - mask_y,
                paste_x1 - mask_x : paste_x2 - mask_x,
            ] = resized[face_y1:face_y2, face_x1:face_x2, ::-1]

        alpha = blend_mask[mask_y:mask_y2, mask_x:mask_x2].astype(np.float32)[..., None]
        alpha /= 255.0
        values = overlay_roi.astype(np.float32) * alpha
        values += source_roi.astype(np.float32) * (1.0 - alpha)
        result[mask_y:mask_y2, mask_x:mask_x2] = np.clip(
            np.rint(values), 0, 255
        ).astype(np.uint8)
        return np.ascontiguousarray(result)

    def render_batch(self, audio_features: Any, frames: Sequence[Any]) -> NDArray[np.uint8]:
        if len(frames) == 0:
            if self._prepared:
                sample = next(iter(self._prepared.values())).frame
                return np.empty((0, *sample.shape), dtype=np.uint8)
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        if len(audio_features) != len(frames):
            raise ValueError("audio feature and target frame counts differ")
        if len(frames) > self.batch_size:
            raise ValueError(f"batch exceeds configured size {self.batch_size}")

        actual_count = len(frames)
        items = [self.get_prepared(frame) for frame in frames]
        if actual_count < self.batch_size:
            pad_count = self.batch_size - actual_count
            items.extend([items[-1]] * pad_count)
            padding = audio_features[-1:].repeat(
                (pad_count,) + (1,) * (audio_features.ndim - 1)
            )
            audio_features = self.torch.cat([audio_features, padding], dim=0)
        with self._lock, self.torch.inference_mode():
            prompts = audio_features.to(device=self.device, dtype=self.weight_dtype)
            prompts = self.pe(prompts)
            latents = self.torch.stack([item.latent for item in items])
            latents = latents.to(device=self.device, dtype=self.weight_dtype)
            predictions = self.unet.model(
                latents,
                self.timesteps,
                encoder_hidden_states=prompts,
            ).sample
            faces = self.decoder.decode_latents(predictions)

        blend_inputs = list(zip(items[:actual_count], faces[:actual_count]))
        if actual_count == 1:
            item, face = blend_inputs[0]
            output = [self.blend_face(item.frame, face, item.face_box, item.blend_mask)]
        else:
            futures = [
                self._blend_executor.submit(
                    self.blend_face,
                    item.frame,
                    face,
                    item.face_box,
                    item.blend_mask,
                )
                for item, face in blend_inputs
            ]
            output = [future.result() for future in futures]
        return np.stack(output).astype(np.uint8, copy=False)

    def warm_up(
        self,
        frames: Sequence[Any],
        iterations: int = 2,
        fps: float = 25.0,
    ) -> float:
        """Warm Whisper and the fixed UNet/VAE batch shape."""

        if not frames:
            raise ValueError("at least one frame is required for warm-up")
        if fps <= 0:
            raise ValueError("warm-up fps must be positive")
        selected = [frames[index % len(frames)] for index in range(self.batch_size)]
        prompts = self.torch.zeros(
            (self.batch_size, 50, 384),
            dtype=self.weight_dtype,
        )
        started = time.perf_counter()
        silent_features = self.extract_audio_features(
            np.zeros(16_000, dtype=np.float32),
            fps,
        )
        if len(silent_features) < self.batch_size:
            raise MuseTalkSetupError("audio warm-up returned too few Whisper features")
        self.render_batch(silent_features[: self.batch_size], selected)
        for _ in range(max(1, int(iterations))):
            self.render_batch(prompts, selected)
        elapsed = time.perf_counter() - started
        log.info("MuseTalk Whisper/UNet/VAE warm-up complete in %.3fs", elapsed)
        return elapsed
