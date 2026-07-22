"""Comprobacion de recursos y compatibilidad del modelo de interfaz."""

from __future__ import annotations

import argparse

from .config import load_model_spec
from .resources import load_resources, verify_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica los recursos de la interfaz")
    parser.add_argument(
        "--hashes-only",
        action="store_true",
        help="Comprueba archivos y hashes sin cargar ResNet ni el checkpoint",
    )
    args = parser.parse_args()

    spec = load_model_spec()
    verify_assets(spec)
    print("Recursos locales y sumas SHA-256: correctos")

    if not args.hashes_only:
        resources = load_resources(spec)
        print(f"Checkpoint compatible: {spec.checkpoint_path.name}")
        print(f"Dispositivo seleccionado: {resources.device}")
        print(f"Configuracion: T={spec.num_frames}, DCT={spec.num_dct}, tau={spec.threshold}")


if __name__ == "__main__":
    main()
