"""Errores controlados que pueden mostrarse de forma segura en la interfaz."""


class DeepfakeAppError(RuntimeError):
    """Error base de la aplicacion."""


class ResourceValidationError(DeepfakeAppError):
    """Un recurso del modelo no existe, esta corrupto o es incompatible."""


class VideoValidationError(DeepfakeAppError):
    """El archivo no puede procesarse como un video compatible."""


class NoFaceDetectedError(DeepfakeAppError):
    """No se detecto ningun rostro valido en las posiciones muestreadas."""


class InferenceError(DeepfakeAppError):
    """La red no pudo producir una salida numericamente valida."""
