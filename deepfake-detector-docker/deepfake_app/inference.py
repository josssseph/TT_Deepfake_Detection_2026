"""Orquestacion de la inferencia sobre un video MP4."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .errors import InferenceError
from .preprocessing import preprocess_video
from .types import InferenceResult, ProgressCallback, ResourceBundle


def label_from_score(score: float, threshold: float) -> int:
    if not math.isfinite(score):
        raise InferenceError("El modelo produjo un puntaje no finito.")
    return int(score >= threshold)


def _extract_spatial_features(
    faces_bgr: np.ndarray,
    resources: ResourceBundle,
    progress_callback: ProgressCallback | None,
    chunk_size: int = 10,
) -> torch.Tensor:
    faces_rgb = np.ascontiguousarray(faces_bgr[..., ::-1])
    frames = torch.from_numpy(faces_rgb).float().permute(0, 3, 1, 2).contiguous()

    chunks: list[torch.Tensor] = []
    total = frames.shape[0]
    with torch.inference_mode():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            batch = frames[start:end].to(resources.device)
            chunks.append(resources.backbone(batch))
            if progress_callback is not None:
                progress_callback(
                    "Extrayendo caracteristicas espaciales", end / total
                )
    return torch.cat(chunks, dim=0).unsqueeze(0)


def analyze_video(
    video_path: Path,
    resources: ResourceBundle,
    progress_callback: ProgressCallback | None = None,
) -> InferenceResult:
    started = perf_counter()

    def preprocessing_progress(message: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(message, 0.05 + 0.55 * fraction)

    def feature_progress(message: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(message, 0.60 + 0.32 * fraction)

    with resources.lock:
        if progress_callback is not None:
            progress_callback("Validando video", 0.02)

        processed = preprocess_video(
            Path(video_path),
            resources.face_detector,
            resources.spec,
            progress_callback=preprocessing_progress,
        )
        spatial = _extract_spatial_features(
            processed.faces_bgr,
            resources,
            progress_callback=feature_progress,
        )

        spec = resources.spec
        dct_tensor = torch.from_numpy(processed.dct_coefficients).float()
        dct_tensor = dct_tensor.unsqueeze(0).to(resources.device)
        metrics_shape = (1, spec.num_frames, 1)
        ssim = torch.zeros(metrics_shape, dtype=torch.float32, device=resources.device)
        jitter = torch.zeros(metrics_shape, dtype=torch.float32, device=resources.device)

        if progress_callback is not None:
            progress_callback("Ejecutando clasificacion", 0.95)

        with torch.inference_mode():
            logits_tensor = resources.classifier(spatial, dct_tensor, ssim, jitter)
            probabilities = torch.softmax(logits_tensor, dim=1)

        logits_array = logits_tensor[0].detach().cpu().numpy().astype(float)
        fake_score = float(probabilities[0, 1].detach().cpu().item())
        if not np.all(np.isfinite(logits_array)):
            raise InferenceError("El modelo produjo logits no finitos.")

        predicted = label_from_score(fake_score, spec.threshold)
        label = spec.positive_class if predicted == 1 else spec.negative_class

    if progress_callback is not None:
        progress_callback("Analisis completado", 1.0)

    return InferenceResult(
        label=label,
        fake_score=fake_score,
        real_score=1.0 - fake_score,
        threshold=spec.threshold,
        logits=(float(logits_array[0]), float(logits_array[1])),
        diagnostics=processed.diagnostics,
        device=str(resources.device),
        elapsed_seconds=perf_counter() - started,
    )
