#!/usr/bin/env python3
"""
train_tuning.py – Entrenamiento con características precomputadas.
Utiliza el modelo ligero PrecomputedDeepfakeDetector y el dataset
que lee únicamente los vectores espaciales, DCT, SSIM y Jitter.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
import csv
import h5py
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC

from dataset import DeepfakeHDF5Dataset
from model import PrecomputedDeepfakeDetector

# ============================================================
# PARÁMETROS POR LÍNEA DE COMANDOS
# ============================================================
parser = argparse.ArgumentParser(description="Entrenamiento y evaluación del detector de deepfakes")

# Datos
parser.add_argument('--h5_path', type=str, required=True, help='Ruta al archivo HDF5 precomputado')
parser.add_argument('--train_idx', type=str, required=True, help='Archivo .npy con índices de train')
parser.add_argument('--val_idx', type=str, required=True, help='Archivo .npy con índices de val')
parser.add_argument('--test_idx', type=str, default=None, help='(Opcional) Archivo .npy con índices de test para evaluación final')

# Submuestreo
parser.add_argument('--num_frames', type=int, default=30, help='Número de frames por vídeo')
parser.add_argument('--num_dct', type=int, default=1024, help='Número de coeficientes DCT')

# Arquitectura (flags de ablación)
parser.add_argument('--spatial', dest='use_spatial', action='store_true', default=True,
                    help='Activar rama espacial')
parser.add_argument('--no_spatial', dest='use_spatial', action='store_false',
                    help='Desactivar rama espacial')
parser.add_argument('--spectral', dest='use_spectral', action='store_true', default=True,
                    help='Activar rama espectral (DCT)')
parser.add_argument('--no_spectral', dest='use_spectral', action='store_false',
                    help='Desactivar rama espectral')
parser.add_argument('--metrics', dest='use_metrics', action='store_true', default=True,
                    help='Activar SSIM y Jitter')
parser.add_argument('--no_metrics', dest='use_metrics', action='store_false',
                    help='Desactivar SSIM y Jitter')

parser.add_argument('--spectral_hidden_dim', type=int, default=128, help='Neuronas en la salida de la rama espectral')
parser.add_argument('--lstm_hidden', type=int, default=256, help='Tamaño oculto de la LSTM')
parser.add_argument('--lstm_layers', type=int, default=1, help='Número de capas de la LSTM')

# Entrenamiento
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay de AdamW')
parser.add_argument('--patience', type=int, default=10, help='Paciencia para early stopping')
parser.add_argument('--early_stop_metric', type=str, default='auc',
                    choices=['recall', 'auc'], help='Métrica para early stopping')

# Otros
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='experimento', help='Nombre identificativo del experimento')
parser.add_argument('--results_csv', type=str, default='resultados_experimentos.csv', help='Archivo CSV para resultados')
parser.add_argument('--save_model', type=str, default='mejor_modelo.pth', help='Ruta para guardar el mejor modelo')

args = parser.parse_args()

if not (args.use_spatial or args.use_spectral or args.use_metrics):
    parser.error("Debe activarse al menos una modalidad: spatial, spectral o metrics.")

# ============================================================
# 1. DISPOSITIVO Y CONFIGURACIÓN DE AMP
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
print(f"Dispositivo: {device}  |  AMP habilitado: {use_amp}")

train_idx = np.load(args.train_idx)
val_idx   = np.load(args.val_idx)
print(f"Videos de entrenamiento: {len(train_idx)} | Validación: {len(val_idx)}")

# Construir datasets con las banderas de ablación
train_dataset = DeepfakeHDF5Dataset(
    args.h5_path, train_idx,
    num_frames=args.num_frames,
    num_dct=args.num_dct,
    load_spatial=args.use_spatial,
    load_spectral=args.use_spectral,
    load_metrics=args.use_metrics
)
val_dataset = DeepfakeHDF5Dataset(
    args.h5_path, val_idx,
    num_frames=args.num_frames,
    num_dct=args.num_dct,
    load_spatial=args.use_spatial,
    load_spectral=args.use_spectral,
    load_metrics=args.use_metrics
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=False, persistent_workers=False)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False, persistent_workers=False)

# ============================================================
# 2. CONSTRUCCIÓN DEL MODELO LIGERO
# ============================================================
print(f"\nConfiguración del modelo (precomputed features):")
print(f"  Espacial: {args.use_spatial}, Espectral: {args.use_spectral}, Métricas: {args.use_metrics}")
print(f"  Frames: {args.num_frames}, DCT: {args.num_dct}")
print(f"  LSTM: {args.lstm_layers} capa(s), hidden={args.lstm_hidden}")
print(f"  LR: {args.lr}, Batch: {args.batch_size}, Paciencia: {args.patience}")
print(f"  Early stopping métrica: {args.early_stop_metric}")

model = PrecomputedDeepfakeDetector(
    dct_input_dim=args.num_dct,
    spectral_hidden_dim=args.spectral_hidden_dim,
    lstm_hidden=args.lstm_hidden,
    lstm_layers=args.lstm_layers,
    use_spatial=args.use_spatial,
    use_spectral=args.use_spectral,
    use_metrics=args.use_metrics
).to(device)

# ============================================================
# 3. OPTIMIZADOR, LOSS Y MÉTRICAS
# ============================================================
with h5py.File(args.h5_path, 'r') as f:
    train_labels = f['Y'][:]
train_labels = train_labels[train_idx]

n_real = np.sum(train_labels == 0)
n_fake = np.sum(train_labels == 1)
total = n_real + n_fake
weight_real = total / (2.0 * n_real) if n_real > 0 else 1.0
weight_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float).to(device)
print(f"Pesos de clase (real, fake): {class_weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Métricas
acc_metric = BinaryAccuracy().to(device)
recall_metric = BinaryRecall().to(device)
auc_metric = BinaryAUROC().to(device)

# ============================================================
# 4. FUNCIÓN PARA PREPARAR EL BATCH
# ============================================================
def prepare_batch(spatial, dct, ssim, jitter, labels, device):
    """
    Convierte los tensores vacíos (cuando una modalidad está desactivada)
    en tensores de ceros con la forma adecuada para que el modelo los reciba siempre.
    """
    B = labels.size(0)

    # Determinar T a partir de alguna modalidad activa
    if args.use_spatial:
        T = spatial.size(1)
    elif args.use_spectral:
        T = dct.size(1)
    elif args.use_metrics:
        T = ssim.size(1)
    else:
        raise RuntimeError("Al menos una modalidad debe estar activa.")

    if args.use_spatial:
        spatial = spatial.to(device)
    else:
        spatial = torch.zeros(B, T, 512, device=device)

    if args.use_spectral:
        dct = dct.to(device)
    else:
        dct = torch.zeros(B, T, args.num_dct, device=device)

    if args.use_metrics:
        ssim = ssim.to(device)
        jitter = jitter.to(device)
    else:
        ssim = torch.zeros(B, T, 1, device=device)
        jitter = torch.zeros(B, T, 1, device=device)

    labels = labels.to(device)
    return spatial, dct, ssim, jitter, labels

# ============================================================
# 5. BUCLE DE ENTRENAMIENTO CON EARLY STOPPING
# ============================================================
best_metric_value = float('-inf')
best_epoch = 0
patience_counter = 0
best_val_loss = None
best_val_acc = None
best_val_recall = None
best_val_auc = None

for epoch in range(args.epochs):
    start_time = time.time()

    # --- Entrenamiento ---
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        spatial, dct, ssim, jitter, labels = batch
        spatial, dct, ssim, jitter, labels = prepare_batch(spatial, dct, ssim, jitter, labels, device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(spatial, dct, ssim, jitter)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * labels.size(0)

    epoch_train_loss = running_loss / len(train_dataset)

    # --- Validación ---
    model.eval()
    val_loss = 0.0
    acc_metric.reset()
    recall_metric.reset()
    auc_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            spatial, dct, ssim, jitter, labels = batch
            spatial, dct, ssim, jitter, labels = prepare_batch(spatial, dct, ssim, jitter, labels, device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(spatial, dct, ssim, jitter)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            acc_metric.update(probs, labels)
            recall_metric.update(probs, labels)
            auc_metric.update(probs, labels)

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = acc_metric.compute().item()
    epoch_val_recall = recall_metric.compute().item()
    epoch_val_auc = auc_metric.compute().item()

    end_time = time.time()
    print(f"Época [{epoch+1}/{args.epochs}] - {end_time-start_time:.1f}s")
    print(f"  Train Loss: {epoch_train_loss:.4f}")
    print(f"  Val   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | Recall: {epoch_val_recall:.4f} | AUC: {epoch_val_auc:.4f}")

    # Seleccionar métrica de early stopping
    current_metric = epoch_val_auc if args.early_stop_metric == 'auc' else epoch_val_recall

    if current_metric > best_metric_value:
        best_metric_value = current_metric
        best_epoch = epoch + 1
        patience_counter = 0
        best_val_loss = epoch_val_loss
        best_val_acc = epoch_val_acc
        best_val_recall = epoch_val_recall
        best_val_auc = epoch_val_auc
        torch.save(model.state_dict(), args.save_model)
        print(f"  >>> Mejor modelo guardado ({args.early_stop_metric}={best_metric_value:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping activado tras {args.patience} épocas sin mejora en {args.early_stop_metric}.")
            break

    print("-" * 50)

# ============================================================
# 6. EVALUACIÓN FINAL EN TEST (OPCIONAL)
# ============================================================
test_results = {}
if args.test_idx is not None:
    print("\nEvaluando en conjunto de test...")
    test_idx = np.load(args.test_idx)

    test_dataset = DeepfakeHDF5Dataset(
        args.h5_path, test_idx,
        num_frames=args.num_frames,
        num_dct=args.num_dct,
        load_spatial=args.use_spatial,
        load_spectral=args.use_spectral,
        load_metrics=args.use_metrics
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False, persistent_workers=False)

    model.load_state_dict(torch.load(args.save_model, map_location=device))
    model.eval()

    test_loss = 0.0
    acc_metric.reset()
    recall_metric.reset()
    auc_metric.reset()

    with torch.no_grad():
        for batch in test_loader:
            spatial, dct, ssim, jitter, labels = batch
            spatial, dct, ssim, jitter, labels = prepare_batch(spatial, dct, ssim, jitter, labels, device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(spatial, dct, ssim, jitter)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            acc_metric.update(probs, labels)
            recall_metric.update(probs, labels)
            auc_metric.update(probs, labels)

    test_loss = test_loss / len(test_dataset)
    test_acc = acc_metric.compute().item()
    test_recall = recall_metric.compute().item()
    test_auc = auc_metric.compute().item()
    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Recall: {test_recall:.4f} | AUC: {test_auc:.4f}")
    test_results = {'test_loss': test_loss, 'test_acc': test_acc, 'test_recall': test_recall, 'test_auc': test_auc}

# ============================================================
# 7. REGISTRO EN CSV
# ============================================================
csv_file = args.results_csv
if not os.path.isfile(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['exp_name', 'num_frames', 'num_dct', 'lr', 'batch_size',
                  'lstm_layers', 'lstm_hidden', 'spatial', 'spectral', 'metrics',
                  'best_epoch', 'val_loss', 'val_acc', 'val_recall', 'val_auc']
        if args.test_idx is not None:
            header += ['test_loss', 'test_acc', 'test_recall', 'test_auc']
        writer.writerow(header)

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    row = [args.exp_name, args.num_frames, args.num_dct, args.lr, args.batch_size,
           args.lstm_layers, args.lstm_hidden, int(args.use_spatial), int(args.use_spectral), int(args.use_metrics),
           best_epoch, best_val_loss, best_val_acc, best_val_recall, best_val_auc]
    if args.test_idx is not None:
        row += [test_results['test_loss'], test_results['test_acc'], test_results['test_recall'], test_results['test_auc']]
    writer.writerow(row)

print(f"\nResultados añadidos a {csv_file}")
