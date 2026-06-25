#!/usr/bin/env python3
"""
train_tuning_v2.py – Búsqueda Macro (Grid Search) con Reproducibilidad.
Implementa:
- Semillas fijadas (Reproducibilidad total).
- Checkpointing Dual (Guarda el mejor por Val_Loss y el mejor por Val_AUC).
- Métricas completas (ACC, AUC, Recall, Precision, F1).
- Early Stopping estricto basado en val_loss.
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
import random
import h5py

from torchmetrics.classification import (
    BinaryAccuracy, BinaryRecall, BinaryAUROC, 
    BinaryPrecision, BinaryF1Score
)

from dataset import DeepfakeHDF5Dataset
from model import PrecomputedDeepfakeDetector

# ============================================================
# 0. FIJACIÓN DE SEMILLAS PARA REPRODUCIBILIDAD
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42) # Semilla universal para la tesis

# ============================================================
# PARÁMETROS POR LÍNEA DE COMANDOS
# ============================================================
parser = argparse.ArgumentParser(description="Grid Search V2 para Deepfakes")

parser.add_argument('--h5_path', type=str, required=True)
parser.add_argument('--train_idx', type=str, required=True)
parser.add_argument('--val_idx', type=str, required=True)

parser.add_argument('--num_frames', type=int, default=30)
parser.add_argument('--num_dct', type=int, default=1024)

parser.add_argument('--spatial', dest='use_spatial', action='store_true', default=True)
parser.add_argument('--no_spatial', dest='use_spatial', action='store_false')
parser.add_argument('--spectral', dest='use_spectral', action='store_true', default=True)
parser.add_argument('--no_spectral', dest='use_spectral', action='store_false')
parser.add_argument('--metrics', dest='use_metrics', action='store_true', default=True)
parser.add_argument('--no_metrics', dest='use_metrics', action='store_false')

parser.add_argument('--spectral_hidden_dim', type=int, default=128)
parser.add_argument('--lstm_hidden', type=int, default=256)
parser.add_argument('--lstm_layers', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=10, help='Paciencia basada en val_loss')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--results_csv', type=str, default='resultados_tuning_v2.csv')
parser.add_argument('--save_model_dir', type=str, default='models/aligned_v2')

args = parser.parse_args()

# ============================================================
# 1. DISPOSITIVO Y DATASETS
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()

os.makedirs(args.save_model_dir, exist_ok=True)

train_idx = np.load(args.train_idx)
val_idx = np.load(args.val_idx)

train_dataset = DeepfakeHDF5Dataset(
    args.h5_path, train_idx, num_frames=args.num_frames, num_dct=args.num_dct,
    load_spatial=args.use_spatial, load_spectral=args.use_spectral, load_metrics=args.use_metrics
)
val_dataset = DeepfakeHDF5Dataset(
    args.h5_path, val_idx, num_frames=args.num_frames, num_dct=args.num_dct,
    load_spatial=args.use_spatial, load_spectral=args.use_spectral, load_metrics=args.use_metrics
)

# shuffle=True ahora es determinista gracias a set_seed(42)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# ============================================================
# 2. MODELO Y OPTIMIZADOR
# ============================================================
model = PrecomputedDeepfakeDetector(
    dct_input_dim=args.num_dct, spectral_hidden_dim=args.spectral_hidden_dim,
    lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
    use_spatial=args.use_spatial, use_spectral=args.use_spectral, use_metrics=args.use_metrics
).to(device)

with h5py.File(args.h5_path, 'r') as f:
    train_labels = f['Y'][:][train_idx]
n_real, n_fake = np.sum(train_labels == 0), np.sum(train_labels == 1)
class_weights = torch.tensor([len(train_labels)/(2.0*n_real), len(train_labels)/(2.0*n_fake)], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Métricas
torch_metrics = {
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

# ============================================================
# 3. BUCLE CON CHECKPOINTING DUAL
# ============================================================
history_csv = os.path.join(args.save_model_dir, f"{args.exp_name}_history.csv")
with open(history_csv, 'w', newline='') as f:
    history_writer = csv.writer(f)
    history_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_auc', 'val_recall', 'val_precision', 'val_f1'])

best_val_loss = float('inf')
best_val_auc = float('-inf')
patience_counter = 0

best_loss_stats = {}
best_auc_stats = {}

path_best_loss = os.path.join(args.save_model_dir, f"{args.exp_name}_best_loss.pth")
path_best_auc = os.path.join(args.save_model_dir, f"{args.exp_name}_best_auc.pth")

print(f"--- Iniciando V2: {args.exp_name} ---")

for epoch in range(args.epochs):
    start_time = time.time()
    
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
    
    model.eval()
    val_loss = 0.0
    for m in torch_metrics.values(): m.reset()
    
    with torch.no_grad():
        for batch in val_loader:
            spatial, dct, ssim, jitter, labels = batch
            spatial, dct, ssim, jitter, labels = prepare_batch(spatial, dct, ssim, jitter, labels, device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(spatial, dct, ssim, jitter)
                loss = criterion(outputs, labels)
                
            val_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            for m in torch_metrics.values(): m.update(probs, labels)
            
    epoch_val_loss = val_loss / len(val_dataset)
    stats = {k: v.compute().item() for k, v in torch_metrics.items()}
    
    with open(history_csv, 'a', newline='') as f:
        history_writer = csv.writer(f)
        history_writer.writerow([
            epoch + 1, epoch_train_loss, epoch_val_loss,
            stats['acc'], stats['auc'], stats['recall'], stats['precision'], stats['f1']
        ])

    print(f"Ep [{epoch+1}/{args.epochs}] Loss: {epoch_val_loss:.4f} | AUC: {stats['auc']:.4f} | Recall: {stats['recall']:.4f}")

    # --- CHECKPOINT 1: Mejor AUC (Para discriminación máxima) ---
    if stats['auc'] > best_val_auc:
        best_val_auc = stats['auc']
        best_auc_stats = stats.copy()
        best_auc_stats['epoch'] = epoch + 1
        best_auc_stats['val_loss'] = epoch_val_loss
        torch.save(model.state_dict(), path_best_auc)

    # --- CHECKPOINT 2: Mejor Loss (Para estabilidad matemática y Early Stopping) ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_loss_stats = stats.copy()
        best_loss_stats['epoch'] = epoch + 1
        best_loss_stats['val_loss'] = epoch_val_loss
        torch.save(model.state_dict(), path_best_loss)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping en época {epoch+1}.")
            break

# ============================================================
# 4. REGISTRO EN CSV
# ============================================================
file_exists = os.path.isfile(args.results_csv)
with open(args.results_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'exp_name', 'num_frames', 'num_dct', 'spatial', 'spectral', 'metrics',
            'stop_reason', 'best_loss_epoch', 'min_val_loss', 'loss_auc', 'loss_recall', 'loss_precision', 'loss_f1',
            'best_auc_epoch', 'auc_val_loss', 'max_val_auc', 'auc_recall', 'auc_precision', 'auc_f1'
        ])
    
    writer.writerow([
        args.exp_name, args.num_frames, args.num_dct, int(args.use_spatial), int(args.use_spectral), int(args.use_metrics),
        'early_stopping', best_loss_stats.get('epoch', -1), best_loss_stats.get('val_loss', -1), best_loss_stats.get('auc', -1), best_loss_stats.get('recall', -1), best_loss_stats.get('precision', -1), best_loss_stats.get('f1', -1),
        best_auc_stats.get('epoch', -1), best_auc_stats.get('val_loss', -1), best_auc_stats.get('auc', -1), best_auc_stats.get('recall', -1), best_auc_stats.get('precision', -1), best_auc_stats.get('f1', -1)
    ])
print("Experimento finalizado y registrado.")

