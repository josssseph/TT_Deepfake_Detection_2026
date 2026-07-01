"""Preprocesamiento de videos nuevos, equivalente al flujo experimental."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.fftpack import dct

from .errors import NoFaceDetectedError, VideoValidationError
from .types import ModelSpec, PreprocessedVideo, ProgressCallback, SequenceDiagnostics


def uniform_indices(total_frames: int, count: int) -> np.ndarray:
    if total_frames <= 0:
        raise VideoValidationError("El video no contiene fotogramas decodificables.")
    if count <= 0:
        raise ValueError("La cantidad de posiciones debe ser positiva.")
    return np.linspace(0, total_frames - 1, count, dtype=np.int64)


def zigzag_indices(size: int, limit: int) -> list[tuple[int, int]]:
    if size <= 0 or limit <= 0 or limit > size * size:
        raise ValueError("Dimensiones invalidas para el recorrido zigzag.")

    indices: list[tuple[int, int]] = []
    for diagonal in range(2 * size - 1):
        if diagonal % 2 == 0:
            row = min(diagonal, size - 1)
            column = diagonal - row
            while row >= 0 and column < size:
                indices.append((row, column))
                if len(indices) == limit:
                    return indices
                row -= 1
                column += 1
        else:
            column = min(diagonal, size - 1)
            row = diagonal - column
            while column >= 0 and row < size:
                indices.append((row, column))
                if len(indices) == limit:
                    return indices
                row += 1
                column -= 1
    return indices


def detect_best_face(
    frame: np.ndarray,
    face_detector: Any,
    confidence_threshold: float,
    image_size: int,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=[104, 117, 123],
        swapRB=False,
        crop=False,
    )
    face_detector.setInput(blob)
    detections = face_detector.forward()

    if detections.size == 0 or detections.shape[2] == 0:
        return None

    confidences = detections[0, 0, :, 2]
    best_index = int(np.argmax(confidences))
    if float(confidences[best_index]) <= confidence_threshold:
        return None

    box = detections[0, 0, best_index, 3:7] * np.array(
        [width, height, width, height], dtype=np.float32
    )
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    resized = cv2.resize(
        face, (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    return resized.astype(np.float32) / 255.0


def dct_features(
    face_bgr: np.ndarray,
    num_coefficients: int,
    indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    blue = face_bgr[:, :, 0]
    green = face_bgr[:, :, 1]
    red = face_bgr[:, :, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    transformed = dct(dct(luminance.T, norm="ortho").T, norm="ortho")

    indices = indices or zigzag_indices(face_bgr.shape[0], num_coefficients)
    coefficients = np.asarray(
        [transformed[row, column] for row, column in indices], dtype=np.float32
    )
    return coefficients


def finalize_sequence(
    faces: list[np.ndarray],
    dct_vectors: list[np.ndarray],
    spec: ModelSpec,
    total_video_frames: int,
    decoded_positions: int,
) -> PreprocessedVideo:
    if not faces:
        raise NoFaceDetectedError(
            "No se detecto ningun rostro valido en las posiciones muestreadas."
        )
    if len(faces) != len(dct_vectors):
        raise ValueError("Las secuencias facial y DCT no estan alineadas.")

    valid_faces = len(faces)
    while len(faces) < spec.base_frames:
        faces.append(faces[-1].copy())
        dct_vectors.append(dct_vectors[-1].copy())

    faces_array = np.stack(faces[: spec.base_frames]).astype(np.float32)
    dct_array = np.stack(dct_vectors[: spec.base_frames]).astype(np.float32)
    selected = uniform_indices(spec.base_frames, spec.num_frames)

    diagnostics = SequenceDiagnostics(
        total_video_frames=total_video_frames,
        requested_positions=spec.base_frames,
        decoded_positions=decoded_positions,
        valid_faces=valid_faces,
        padded_positions=max(0, spec.base_frames - valid_faces),
        selected_positions=spec.num_frames,
    )
    return PreprocessedVideo(
        faces_bgr=np.ascontiguousarray(faces_array[selected]),
        dct_coefficients=np.ascontiguousarray(
            dct_array[selected, : spec.num_dct]
        ),
        diagnostics=diagnostics,
    )


def preprocess_video(
    video_path: Path,
    face_detector: Any,
    spec: ModelSpec,
    progress_callback: ProgressCallback | None = None,
) -> PreprocessedVideo:
    if video_path.suffix.lower() != ".mp4":
        raise VideoValidationError("La primera version admite videos MP4.")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise VideoValidationError("El archivo de video no existe o esta vacio.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoValidationError("OpenCV no pudo abrir el video seleccionado.")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_indices = uniform_indices(total_frames, spec.base_frames)
        dct_order = zigzag_indices(spec.image_size, spec.num_dct)
        faces: list[np.ndarray] = []
        dct_vectors: list[np.ndarray] = []
        decoded_positions = 0

        for position, frame_index in enumerate(sampled_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if success and frame is not None:
                decoded_positions += 1
                face = detect_best_face(
                    frame,
                    face_detector,
                    spec.face_confidence,
                    spec.image_size,
                )
                if face is not None:
                    faces.append(face)
                    dct_vectors.append(
                        dct_features(face, spec.num_dct, indices=dct_order)
                    )

            if progress_callback is not None:
                progress_callback(
                    "Detectando y normalizando rostros",
                    (position + 1) / spec.base_frames,
                )
    finally:
        capture.release()

    if decoded_positions == 0:
        raise VideoValidationError(
            "No se pudo decodificar ninguna de las posiciones seleccionadas."
        )

    return finalize_sequence(
        faces,
        dct_vectors,
        spec,
        total_video_frames=total_frames,
        decoded_positions=decoded_positions,
    )
