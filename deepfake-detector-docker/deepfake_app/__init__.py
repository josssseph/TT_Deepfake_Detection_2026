"""Aplicacion local para inferencia sobre videos nuevos."""

from .inference import analyze_video
from .types import InferenceResult

__all__ = ["InferenceResult", "analyze_video"]
