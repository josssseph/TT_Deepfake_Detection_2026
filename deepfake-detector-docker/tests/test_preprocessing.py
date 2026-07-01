from dataclasses import replace

import numpy as np
import pytest

from deepfake_app.config import load_model_spec
from deepfake_app.errors import NoFaceDetectedError, VideoValidationError
from deepfake_app.preprocessing import (
    dct_features,
    detect_best_face,
    finalize_sequence,
    uniform_indices,
    zigzag_indices,
)


class FakeFaceDetector:
    def __init__(self, detections: np.ndarray):
        self.detections = detections
        self.last_blob = None

    def setInput(self, blob: np.ndarray) -> None:
        self.last_blob = blob

    def forward(self) -> np.ndarray:
        return self.detections


def test_uniform_indices_match_training_semantics() -> None:
    observed = uniform_indices(100, 5)
    assert np.array_equal(observed, np.array([0, 24, 49, 74, 99]))


def test_uniform_indices_reject_empty_video() -> None:
    with pytest.raises(VideoValidationError):
        uniform_indices(0, 60)


def test_zigzag_starts_at_low_frequencies() -> None:
    assert zigzag_indices(4, 8) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
    ]


def test_dct_constant_face_has_only_dc_component() -> None:
    face = np.full((8, 8, 3), 0.5, dtype=np.float32)
    coefficients = dct_features(face, 8)

    assert coefficients.dtype == np.float32
    assert coefficients.shape == (8,)
    assert coefficients[0] == pytest.approx(4.0, abs=1e-5)
    assert np.max(np.abs(coefficients[1:])) < 1e-5


def test_detector_uses_highest_confidence_and_clamps_box() -> None:
    detections = np.zeros((1, 1, 2, 7), dtype=np.float32)
    detections[0, 0, 0, 2] = 0.70
    detections[0, 0, 0, 3:7] = [0.1, 0.1, 0.5, 0.5]
    detections[0, 0, 1, 2] = 0.95
    detections[0, 0, 1, 3:7] = [-0.2, 0.2, 1.2, 0.8]
    detector = FakeFaceDetector(detections)
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)

    face = detect_best_face(frame, detector, 0.5, 224)

    assert face is not None
    assert face.shape == (224, 224, 3)
    assert face.dtype == np.float32
    assert face.min() == pytest.approx(128 / 255.0)
    assert detector.last_blob is not None


def test_finalize_sequence_pads_before_selecting_frames() -> None:
    spec = replace(load_model_spec(), base_frames=6, num_frames=4, num_dct=3)
    faces = [np.full((2, 2, 3), value, dtype=np.float32) for value in (0.1, 0.2)]
    vectors = [np.full(3, value, dtype=np.float32) for value in (1.0, 2.0)]

    result = finalize_sequence(
        faces,
        vectors,
        spec,
        total_video_frames=20,
        decoded_positions=6,
    )

    assert result.faces_bgr.shape == (4, 2, 2, 3)
    assert result.dct_coefficients.shape == (4, 3)
    assert result.diagnostics.valid_faces == 2
    assert result.diagnostics.padded_positions == 4
    assert result.faces_bgr[-1, 0, 0, 0] == pytest.approx(0.2)


def test_finalize_sequence_rejects_sequence_without_faces() -> None:
    with pytest.raises(NoFaceDetectedError):
        finalize_sequence([], [], load_model_spec(), 10, 10)
