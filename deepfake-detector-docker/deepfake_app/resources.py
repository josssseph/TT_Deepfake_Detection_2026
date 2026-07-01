"""Carga verificada y reutilizable de los modelos de inferencia."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock

import cv2
import torch
from torchvision.models import ResNet18_Weights, resnet18

from scripts.model import PrecomputedDeepfakeDetector

from .config import load_model_spec
from .errors import ResourceValidationError
from .types import ModelSpec, ResourceBundle


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise ResourceValidationError(f"No se encontro {label}: {path}")
    observed = sha256_file(path)
    if observed.lower() != expected_sha256.lower():
        raise ResourceValidationError(
            f"La suma SHA-256 de {label} no coincide con el manifiesto."
        )


def verify_assets(spec: ModelSpec) -> None:
    _verify_file(spec.checkpoint_path, spec.checkpoint_sha256, "el checkpoint")
    _verify_file(
        spec.detector_prototxt_path,
        spec.detector_prototxt_sha256,
        "la definicion del detector facial",
    )
    _verify_file(
        spec.detector_weights_path,
        spec.detector_weights_sha256,
        "los pesos del detector facial",
    )


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ResourceValidationError(
            f"No se pudo leer el checkpoint: {path.name}"
        ) from exc

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ResourceValidationError("El checkpoint no contiene un state_dict valido.")
    return state


def load_resources(spec: ModelSpec | None = None) -> ResourceBundle:
    spec = spec or load_model_spec()
    verify_assets(spec)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        face_detector = cv2.dnn.readNetFromCaffe(
            str(spec.detector_prototxt_path), str(spec.detector_weights_path)
        )
    except cv2.error as exc:
        raise ResourceValidationError("OpenCV no pudo cargar el detector facial.") from exc

    if spec.backbone_weights != "IMAGENET1K_V1":
        raise ResourceValidationError(
            f"Backbone no compatible: {spec.backbone_weights}"
        )

    try:
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise ResourceValidationError(
            "No se pudieron cargar los pesos IMAGENET1K_V1 de ResNet-18. "
            "Ejecuta la preparacion con acceso a Internet."
        ) from exc
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.to(device)

    classifier = PrecomputedDeepfakeDetector(
        dct_input_dim=spec.num_dct,
        spectral_hidden_dim=spec.spectral_dim,
        lstm_hidden=spec.lstm_hidden,
        lstm_layers=1,
        use_spatial=True,
        use_spectral=True,
        use_metrics=False,
    ).to(device)
    try:
        classifier.load_state_dict(
            _load_state_dict(spec.checkpoint_path, device), strict=True
        )
    except RuntimeError as exc:
        raise ResourceValidationError(
            "El checkpoint no coincide con la arquitectura de la interfaz."
        ) from exc
    classifier.eval()

    return ResourceBundle(
        spec=spec,
        face_detector=face_detector,
        backbone=backbone,
        classifier=classifier,
        device=device,
        lock=RLock(),
    )
