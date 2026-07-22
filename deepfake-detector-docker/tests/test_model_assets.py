import cv2
import torch

from deepfake_app.config import load_model_spec
from scripts.model import PrecomputedDeepfakeDetector


def test_detector_files_load_with_opencv() -> None:
    spec = load_model_spec()
    detector = cv2.dnn.readNetFromCaffe(
        str(spec.detector_prototxt_path), str(spec.detector_weights_path)
    )
    assert detector is not None


def test_checkpoint_loads_strictly_into_selected_architecture() -> None:
    spec = load_model_spec()
    state = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    model = PrecomputedDeepfakeDetector(
        dct_input_dim=spec.num_dct,
        spectral_hidden_dim=spec.spectral_dim,
        lstm_hidden=spec.lstm_hidden,
        lstm_layers=1,
        use_spatial=True,
        use_spectral=True,
        use_metrics=False,
    )
    model.load_state_dict(state, strict=True)
