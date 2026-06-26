#!/usr/bin/env python3
"""
finetuning_expert.py
==================================================================
Script especializado para el Fine-Tuning y Calibracion de los mejores
modelos preentrenados. Incorpora optimizacion quirurgica (LR bajo,
Weight Decay regulado y ReduceLROnPlateau) junto con un abanico
completo de metricas forenses (AUC, Recall, Precision, F1, Accuracy).
"""

import random
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

# Importacion de todas las metricas estandar de la literatura
from torchmetrics.classification import (
    BinaryAccuracy, BinaryRecall, BinaryAUROC,
    BinaryPrecision, BinaryF1Score
)

from dataset import DeepfakeHDF5Dataset
from model import PrecomputedDeepfakeDetector

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Fine-tuning quirurgico de detectores Deepfake")

# Rutas y Pesos Iniciales
parser.add_argument('--h5_path', type=str, required=True)
parser.add_argument('--train_idx', type=str, required=True)
parser.add_argument('--val_idx', type=str, required=True)
parser.add_argument('--pretrained_weights', type=str, required=True, help='Ruta al .pth del tuning inicial')

# Configuracion estructural (Debe coincidir exactamente con el modelo a cargar)
parser.add_argument('--num_frames', type=int, required=True)
parser.add_argument('--num_dct', type=int, required=True)
parser.add_argument('--spatial', dest='use_spatial', action='store_true')
parser.add_argument('--no_spatial', dest='use_spatial', action='store_false')
parser.add_argument('--spectral', dest='use_spectral', action='store_true')
parser.add_argument('--no_spectral', dest='use_spectral', action='store_false')
parser.add_argument('--metrics', dest='use_metrics', action='store_true')
parser.add_argument('--no_metrics', dest='use_metrics', action='store_false')

# Hiperparametros de Fine-Tuning (Valores por defecto optimizados)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=2e-5, help='LR muy bajo para Fine-Tuning')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Mayor regularizacion L2')
parser.add_argument('--patience', type=int, default=8)
parser.add_argument('--seed', type=int, default=42, help='Semilla de reproducibilidad')

# Registro
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--results_csv', type=str, default='resultados_finetuning_expert.csv')
parser.add_argument('--save_model', type=str, required=True)

args = parser.parse_args()

# Fijar la semilla para reproducibilidad
set_seed(args.seed)

# Configuracion de Entorno y Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
print(f"Iniciando Fine-Tuning para: {args.exp_name} en {device}")

# Carga de Indices
train_idx = np.load(args.train_idx)
val_idx = np.load(args.val_idx)

# Construccion de Datasets
train_dataset = DeepfakeHDF5Dataset(
    args.h5_path, train_idx, num_frames=args.num_frames, num_dct=args.num_dct,
    load_spatial=args.use_spatial, load_spectral=args.use_spectral, load_metrics=args.use_metrics
)
val_dataset = DeepfakeHDF5Dataset(
    args.h5_path, val_idx, num_frames=args.num_frames, num_dct=args.num_dct,
    load_spatial=args.use_spatial, load_spectral=args.use_spectral, load_metrics=args.use_metrics
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Inicializacion del Modelo Ligero
model = PrecomputedDeepfakeDetector(
    dct_input_dim=args.num_dct, spectral_hidden_dim=128, lstm_hidden=256, lstm_layers=1,
    use_spatial=args.use_spatial, use_spectral=args.use_spectral, use_metrics=args.use_metrics
).to(device)

# Cargar los pesos obtenidos en la macro-exploracion
print(f"Cargando pesos preentrenados desde: {args.pretrained_weights}")
model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))

# Pesos de Clase para Mitigar Desbalances
with h5py.File(args.h5_path, 'r') as f:
    train_labels = f['Y'][:][train_idx]
n_real, n_fake = np.sum(train_labels == 0), np.sum(train_labels == 1)
class_weights = torch.tensor([len(train_labels)/(2.0*n_real), len(train_labels)/(2.0*n_fake)], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Instanciacion del Scheduler dinamico para controlar la perdida (Calibracion)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Inicializacion del ecosistema completo de metricas academicas
metrics = {
    'acc': BinaryAccuracy().to(device),
    'recall': BinaryRecall().to(device),
    'auc': BinaryAUROC().to(device),
    'precision': BinaryPrecision().to(device),
    'f1': BinaryF1Score().to(device)
}

def prepare_batch(spatial, dct, ssim, jitter, labels, device):
    B, T = labels.size(0), args.num_frames
    spatial = spatial.to(device) if args.use_spatial else torch.zeros(B, T, 512, device=device)
    dct = dct.to(device) if args.use_spectral else torch.zeros(B, T, args.num_dct, device=device)
    if args.use_metrics:
        ssim, jitter = ssim.to(device), jitter.to(device)
    else:
        ssim, jitter = torch.zeros(B, T, 1, device=device), torch.zeros(B, T, 1, device=device)
    return spatial, dct, ssim, jitter, labels.to(device)

# --- NUEVO: Inicializacion del archivo de historial por epoca ---
history_csv = args.save_model.replace('.pth', '_history.csv')
with open(history_csv, 'w', newline='') as f:
    history_writer = csv.writer(f)
    history_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'val_recall', 'val_precision', 'val_f1'])

# Bucle de Optimizacion Quirurgica
best_val_loss = float('inf')
patience_counter = 0
best_epoch_stats = {}

for epoch in range(args.epochs):
    start_time = time.time()

    # Fase de Entrenamiento
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

    # Fase de Validacion y Calibracion
    model.eval()
    val_loss = 0.0
    for m in metrics.values(): m.reset()

    with torch.no_grad():
        for batch in val_loader:
            spatial, dct, ssim, jitter, labels = batch
            spatial, dct, ssim, jitter, labels = prepare_batch(spatial, dct, ssim, jitter, labels, device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(spatial, dct, ssim, jitter)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            for m in metrics.values(): m.update(probs, labels)

    epoch_val_loss = val_loss / len(val_dataset)

    # Actualizacion del Scheduler en base a la Perdida de Validacion
    scheduler.step(epoch_val_loss)

    # Extraccion de valores calculados
    stats = {k: v.compute().item() for k, v in metrics.items()}

    # --- NUEVO: Registrar cada epoca en el historial contiguo ---
    with open(history_csv, 'a', newline='') as f:
        history_writer = csv.writer(f)
        history_writer.writerow([
            epoch + 1, epoch_train_loss, epoch_val_loss,
            stats['acc'], stats['auc'], stats['recall'], stats['precision'], stats['f1']
        ])

    end_time = time.time()

    print(f"Epoca [{epoch+1}/{args.epochs}] - Time: {end_time-start_time:.1f}s | Train Loss: {epoch_train_loss:.4f}")
    print(f"   Val Loss: {epoch_val_loss:.4f} | ACC: {stats['acc']:.4f} | AUC: {stats['auc']:.4f} | Recall: {stats['recall']:.4f} | Prc: {stats['precision']:.4f} | F1: {stats['f1']:.4f}")

    # Criterio de guardado basado en MINIMIZAR LA PERDIDA (Garantiza estabilidad)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch_stats = stats.copy()
        best_epoch_stats['best_epoch'] = epoch + 1
        best_epoch_stats['val_loss'] = epoch_val_loss
        torch.save(model.state_dict(), args.save_model)
        print("   >>> Nuevo minimo de perdida encontrado. Modelo guardado de forma estable.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping activado por estancamiento de la perdida de validacion.")
            break
    print("-" * 60)

# Registro estructurado en el CSV Academico
csv_file = args.results_csv
header = ['exp_name', 'num_frames', 'num_dct', 'best_epoch', 'val_loss', 'val_acc', 'val_auc', 'val_recall', 'val_precision', 'val_f1']
file_exists = os.path.isfile(csv_file)

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow([
        args.exp_name, args.num_frames, args.num_dct, best_epoch_stats['best_epoch'], best_epoch_stats['val_loss'],
        best_epoch_stats['acc'], best_epoch_stats['auc'], best_epoch_stats['recall'], best_epoch_stats['precision'], best_epoch_stats['f1']
    ])
print(f"Proceso finalizado. Registro guardado en {csv_file} y historial por epoca en {history_csv}")
