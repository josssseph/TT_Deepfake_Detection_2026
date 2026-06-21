#!/usr/bin/env python3
"""
train_tuning.py – Entrenamiento configurable para ablación y tuning del detector de deepfakes.
Incluye:
  - Argumentos por línea de comandos para todos los hiperparámetros.
  - Métricas: Accuracy, Recall, AUROC.
  - Early stopping basado en Recall de validación (prioriza falsos negativos).
  - Escritura de resultados en un fichero CSV.
  - Compatibilidad con Mixed Precision (AMP) y HPC.
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
import h5py  # Movido al principio para usarlo en los pesos de clase
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC

# Módulos propios
from dataset import DeepfakeHDF5Dataset
from model import DeepfakeDetector

# ============================================================
# PARÁMETROS POR LÍNEA DE COMANDOS
# ============================================================
parser = argparse.ArgumentParser(description="Entrenamiento y evaluación del detector de deepfakes")

# Datos
parser.add_argument('--h5_path', type=str, required=True, help='Ruta al archivo HDF5')
parser.add_argument('--train_idx', type=str, required=True, help='Archivo .npy con índices de train')
parser.add_argument('--val_idx', type=str, required=True, help='Archivo .npy con índices de val')
parser.add_argument('--test_idx', type=str, default=None, help='(Opcional) Archivo .npy con índices de test para evaluación final')

# Submuestreo
parser.add_argument('--num_frames', type=int, default=30, help='Número de frames por vídeo')
parser.add_argument('--num_dct', type=int, default=1024, help='Número de coeficientes DCT')

# Arquitectura
parser.add_argument('--spatial', dest='use_spatial', action='store_true', default=True,
                    help='Activar rama espacial (ResNet)')
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

# Otros
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='experimento', help='Nombre identificativo del experimento')
parser.add_argument('--results_csv', type=str, default='resultados_experimentos.csv', help='Archivo CSV donde se añadirá una fila con los resultados')
parser.add_argument('--save_model', type=str, default='mejor_modelo.pth', help='Ruta para guardar el mejor modelo')
parser.add_argument('--normalize', action='store_true', default=False, help='Aplicar normalización ImageNet (si se descongela ResNet)')

args = parser.parse_args()

# ============================================================
# 1. DISPOSITIVO Y CONFIGURACIÓN DE AMP
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()  # Protección para CPU
print(f"Dispositivo: {device}  |  AMP habilitado: {use_amp}")

train_idx = np.load(args.train_idx)
val_idx   = np.load(args.val_idx)

print(f"Videos de entrenamiento: {len(train_idx)} | Validación: {len(val_idx)}")

# ### NUEVO: Determinar si necesitamos cargar los pesados tensores de imágenes
need_frames = args.use_spatial or args.use_metrics

# ### MODIFICADO: Pasar la bandera load_frames a los datasets
train_dataset = DeepfakeHDF5Dataset(args.h5_path, train_idx,
                                    num_frames=args.num_frames,
                                    num_dct=args.num_dct,
                                    load_frames=need_frames)
val_dataset   = DeepfakeHDF5Dataset(args.h5_path, val_idx,
                                    num_frames=args.num_frames,
                                    num_dct=args.num_dct,
                                    load_frames=need_frames)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

# ============================================================
# 2. CONSTRUCCIÓN DEL MODELO SEGÚN ABLACIÓN
# ============================================================
print(f"\nConfiguración del modelo:")
print(f"  Espacial: {args.use_spatial}, Espectral: {args.use_spectral}, Métricas: {args.use_metrics}")
print(f"  Frames: {args.num_frames}, DCT: {args.num_dct}")
print(f"  LSTM: {args.lstm_layers} capa(s), hidden={args.lstm_hidden}")
print(f"  LR: {args.lr}, Batch: {args.batch_size}, Paciencia: {args.patience}")

model = DeepfakeDetector(dct_input_dim=args.num_dct,
                         spectral_hidden_dim=args.spectral_hidden_dim,
                         lstm_hidden=args.lstm_hidden,
                         lstm_layers=args.lstm_layers).to(device)

# Flags para ablación (se usarán en el forward personalizado)
model.use_spatial = args.use_spatial
model.use_spectral = args.use_spectral
model.use_metrics = args.use_metrics

# ### MODIFICADO: Forward adaptado para evitar errores de shape si los frames están vacíos
def custom_forward(frames, dct_coeffs):
    # Tomamos las dimensiones del batch (B) y tiempo (T) desde la DCT 
    # ya que siempre estará presente, mientras que 'frames' puede venir vacío.
    B, T, _ = dct_coeffs.shape
    
    # Rama espacial
    if model.use_spatial:
        _, _, C, H, W = frames.shape
        frames_reshaped = frames.view(B * T, C, H, W)
        spatial_feat = model.spatial_branch(frames_reshaped).view(B, T, 512)
    else:
        spatial_feat = torch.zeros(B, T, 512, device=device)
    
    # Rama espectral 
    if model.use_spectral:
        dct_flat = dct_coeffs.view(B * T, -1)
        spectral_feat = model.spectral_branch(dct_flat).view(B, T, model.spectral_hidden_dim)
    else:
        spectral_feat = torch.zeros(B, T, model.spectral_hidden_dim, device=device)
    
    # Métricas temporales
    if model.use_metrics:
        ssim_feat, jitter_feat = model.extract_metrics(frames)
    else:
        ssim_feat = torch.zeros(B, T, 1, device=device)
        jitter_feat = torch.zeros(B, T, 1, device=device)
    
    combined = torch.cat([spatial_feat, spectral_feat, ssim_feat, jitter_feat], dim=2)
    lstm_out, (hn, cn) = model.lstm(combined)
    video_summary = hn[-1]  # última capa
    logits = model.classifier(video_summary)
    return logits

model.forward = custom_forward

# ============================================================
# 3. OPTIMIZADOR, LOSS (PESOS DE CLASE) Y MÉTRICAS
# ============================================================
with h5py.File(args.h5_path, 'r') as f:
    train_labels = f['Y'][:]  # Cargar todas las etiquetas
train_labels = train_labels[train_idx]  # Filtrar por índices de entrenamiento

n_real = np.sum(train_labels == 0)
n_fake = np.sum(train_labels == 1)
total = n_real + n_fake
weight_real = total / (2.0 * n_real) if n_real > 0 else 1.0
weight_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float).to(device)
print(f"Pesos de clase (real, fake): {class_weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# AMP seguro (se desactiva automáticamente en CPU)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Métricas de TorchMetrics
acc_metric = BinaryAccuracy().to(device)
recall_metric = BinaryRecall().to(device)
auc_metric = BinaryAUROC().to(device)

# ============================================================
# 4. BUCLE DE ENTRENAMIENTO CON EARLY STOPPING
# ============================================================
best_val_recall = 0.0
best_epoch = 0
patience_counter = 0

for epoch in range(args.epochs):
    start_time = time.time()

    # --- Entrenamiento ---
    model.train()
    running_loss = 0.0
    for frames, dct, labels in train_loader:
        # Nota: Si load_frames=False, `frames` será un tensor vacío. Solo enviamos dct y labels a GPU
        if frames.numel() > 0:
            frames = frames.to(device)
            
        dct, labels = dct.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(frames, dct)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * dct.size(0) # Usamos dct.size(0) para el tamaño del batch

    epoch_train_loss = running_loss / len(train_dataset)

    # --- Validación ---
    model.eval()
    val_loss = 0.0
    acc_metric.reset()
    recall_metric.reset()
    auc_metric.reset()

    with torch.no_grad():
        for frames, dct, labels in val_loader:
            if frames.numel() > 0:
                frames = frames.to(device)
                
            dct, labels = dct.to(device), labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(frames, dct)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * dct.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probabilidad de clase 1 (fake)
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

    # Early stopping basado en Recall
    if epoch_val_recall > best_val_recall:
        best_val_recall = epoch_val_recall
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), args.save_model)
        print(f"  >>> Mejor modelo guardado (Recall={best_val_recall:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping activado tras {args.patience} épocas sin mejora en Recall.")
            break

    print("-" * 50)

# ============================================================
# 5. EVALUACIÓN FINAL EN TEST (OPCIONAL)
# ============================================================
test_results = {}
if args.test_idx is not None:
    print("\nEvaluando en conjunto de test...")
    test_idx = np.load(args.test_idx)
    
    # ### MODIFICADO: Pasar la bandera load_frames también al test
    test_dataset = DeepfakeHDF5Dataset(args.h5_path, test_idx,
                                       num_frames=args.num_frames,
                                       num_dct=args.num_dct,
                                       load_frames=need_frames)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    
    # Cargar el mejor modelo (mapeo seguro al dispositivo)
    model.load_state_dict(torch.load(args.save_model, map_location=device))
    model.eval()
    
    test_loss = 0.0
    acc_metric.reset()
    recall_metric.reset()
    auc_metric.reset()
    
    with torch.no_grad():
        for frames, dct, labels in test_loader:
            if frames.numel() > 0:
                frames = frames.to(device)
            dct, labels = dct.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(frames, dct)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item() * dct.size(0)
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
# 6. REGISTRO EN CSV
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
           best_epoch, epoch_val_loss, epoch_val_acc, best_val_recall, epoch_val_auc]
    if args.test_idx is not None:
        row += [test_results['test_loss'], test_results['test_acc'], test_results['test_recall'], test_results['test_auc']]
    writer.writerow(row)

print(f"\nResultados añadidos a {csv_file}")
