#!/usr/bin/env python3
"""
Cuenta los parametros entrenables y totales de PrecomputedDeepfakeDetector.

Uso previsto:
- Verificar el tamano real del modelo ganador de la tesis.
- Generar una cifra estable para reportar en el capitulo de resultados.
"""

import argparse

from model import PrecomputedDeepfakeDetector


def build_model(args):
    return PrecomputedDeepfakeDetector(
        dct_input_dim=args.num_dct,
        spectral_hidden_dim=args.spectral_hidden_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        use_spatial=args.use_spatial,
        use_spectral=args.use_spectral,
        use_metrics=args.use_metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Cuenta parametros entrenables del modelo de deepfakes")
    parser.add_argument("--num_dct", type=int, required=True)
    parser.add_argument("--spectral_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--spatial", dest="use_spatial", action="store_true")
    parser.add_argument("--no_spatial", dest="use_spatial", action="store_false")
    parser.add_argument("--spectral", dest="use_spectral", action="store_true")
    parser.add_argument("--no_spectral", dest="use_spectral", action="store_false")
    parser.add_argument("--metrics", dest="use_metrics", action="store_true")
    parser.add_argument("--no_metrics", dest="use_metrics", action="store_false")
    parser.set_defaults(use_spatial=True, use_spectral=True, use_metrics=True)
    args = parser.parse_args()

    model = build_model(args)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())

    spectral_params = sum(
        parameter.numel() for parameter in model.spectral_branch.parameters() if parameter.requires_grad
    )
    lstm_params = sum(parameter.numel() for parameter in model.lstm.parameters() if parameter.requires_grad)
    classifier_params = sum(parameter.numel() for parameter in model.classifier.parameters() if parameter.requires_grad)

    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Spectral branch: {spectral_params:,}")
    print(f"LSTM: {lstm_params:,}")
    print(f"Classifier: {classifier_params:,}")


if __name__ == "__main__":
    main()
