from __future__ import annotations

from enum import Enum

import numpy as np
import torch


class LandmarksType(Enum):
    """Type of landmarks returned by the native PyTorch face detector."""

    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Landmark detection backed by the original PyTorch SFD model."""

    def __init__(
        self,
        landmarks_type,
        network_size=NetworkSize.LARGE,
        device="cuda",
        flip_input=False,
        face_detector="sfd",
        verbose=False,
    ):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose
        int(network_size)

        if "cuda" in device:
            torch.backends.cudnn.benchmark = True

        face_detector_module = __import__(
            "face_detection.detection." + face_detector,
            globals(),
            locals(),
            [face_detector],
            0,
        )
        self.face_detector = face_detector_module.FaceDetector(
            device=device,
            verbose=verbose,
        )

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []
        for detections in detected_faces:
            if len(detections) == 0:
                results.append(None)
                continue
            detection = np.clip(detections[0], 0, None)
            x1, y1, x2, y2 = map(int, detection[:-1])
            results.append((x1, y1, x2, y2))
        return results
