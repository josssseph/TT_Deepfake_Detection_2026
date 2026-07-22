"""Tipos compartidos por el preprocesamiento, la inferencia y la interfaz."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class ModelSpec:
    project_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    detector_prototxt_path: Path
    detector_prototxt_sha256: str
    detector_weights_path: Path
    detector_weights_sha256: str
    backbone_weights: str
    base_frames: int
    num_frames: int
    image_size: int
    num_dct: int
    face_confidence: float
    threshold: float
    positive_class: str
    negative_class: str
    spatial_dim: int
    spectral_dim: int
    lstm_hidden: int


@dataclass(frozen=True)
class SequenceDiagnostics:
    total_video_frames: int
    requested_positions: int
    decoded_positions: int
    valid_faces: int
    padded_positions: int
    selected_positions: int


@dataclass(frozen=True)
class PreprocessedVideo:
    faces_bgr: Any
    dct_coefficients: Any
    diagnostics: SequenceDiagnostics


@dataclass(frozen=True)
class InferenceResult:
    label: str
    fake_score: float
    real_score: float
    threshold: float
    logits: tuple[float, float]
    diagnostics: SequenceDiagnostics
    device: str
    elapsed_seconds: float


@dataclass
class ResourceBundle:
    spec: ModelSpec
    face_detector: Any
    backbone: Any
    classifier: Any
    device: Any
    lock: RLock
