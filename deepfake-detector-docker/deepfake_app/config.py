"""Carga y validacion de la configuracion congelada de inferencia."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ResourceValidationError
from .types import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "model_manifest.json"


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ResourceValidationError(f"Falta '{key}' en {context}.")
    return mapping[key]


def _project_path(project_root: Path, value: str) -> Path:
    candidate = (project_root / value).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ResourceValidationError(
            f"La ruta del manifiesto sale del proyecto: {value}"
        ) from exc
    return candidate


def load_model_spec(manifest_path: Path | None = None) -> ModelSpec:
    manifest_path = manifest_path or DEFAULT_MANIFEST
    if not manifest_path.is_file():
        raise ResourceValidationError(f"No existe el manifiesto: {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceValidationError(
            f"No se pudo leer el manifiesto: {manifest_path}"
        ) from exc

    if data.get("schema_version") != 1:
        raise ResourceValidationError("Version de manifiesto no compatible.")

    project_root = manifest_path.resolve().parent
    checkpoint = _required(data, "checkpoint", "el manifiesto")
    detector = _required(data, "face_detector", "el manifiesto")
    backbone = _required(data, "backbone", "el manifiesto")
    model = _required(data, "model", "el manifiesto")

    spec = ModelSpec(
        project_root=project_root,
        checkpoint_path=_project_path(
            project_root, _required(checkpoint, "path", "checkpoint")
        ),
        checkpoint_sha256=str(_required(checkpoint, "sha256", "checkpoint")).lower(),
        detector_prototxt_path=_project_path(
            project_root, _required(detector, "prototxt", "face_detector")
        ),
        detector_prototxt_sha256=str(
            _required(detector, "prototxt_sha256", "face_detector")
        ).lower(),
        detector_weights_path=_project_path(
            project_root, _required(detector, "weights", "face_detector")
        ),
        detector_weights_sha256=str(
            _required(detector, "weights_sha256", "face_detector")
        ).lower(),
        backbone_weights=str(_required(backbone, "weights", "backbone")),
        base_frames=int(_required(model, "base_frames", "model")),
        num_frames=int(_required(model, "num_frames", "model")),
        image_size=int(_required(model, "image_size", "model")),
        num_dct=int(_required(model, "num_dct", "model")),
        face_confidence=float(_required(model, "face_confidence", "model")),
        threshold=float(_required(model, "threshold", "model")),
        positive_class=str(_required(model, "positive_class", "model")),
        negative_class=str(_required(model, "negative_class", "model")),
        spatial_dim=int(_required(model, "spatial_dim", "model")),
        spectral_dim=int(_required(model, "spectral_dim", "model")),
        lstm_hidden=int(_required(model, "lstm_hidden", "model")),
    )

    if not 0.0 < spec.face_confidence < 1.0:
        raise ResourceValidationError("La confianza facial debe estar entre 0 y 1.")
    if not 0.0 < spec.threshold < 1.0:
        raise ResourceValidationError("El umbral debe estar entre 0 y 1.")
    if spec.num_frames > spec.base_frames:
        raise ResourceValidationError("num_frames no puede superar base_frames.")
    if spec.num_dct > spec.image_size * spec.image_size:
        raise ResourceValidationError("num_dct excede los coeficientes disponibles.")

    return spec
