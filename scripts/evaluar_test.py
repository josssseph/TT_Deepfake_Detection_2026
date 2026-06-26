#!/usr/bin/env python3
"""
evaluar_test.py
==================================================================
Script final para evaluar el modelo campeón en el conjunto de Test.
Genera las métricas definitivas (ACC, AUC, Recall, Precision, F1),
la Matriz de Confusión y la Curva ROC para el documento de tesis.
"""

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from dataset import DeepfakeHDF5Dataset
from model import PrecomputedDeepfakeDetector

parser = argparse.ArgumentParser(description="Evaluación final en el conjunto de Test")

# Rutas
parser.add_argument('--h5_path', type=str, required=True)
parser.add_argument('--test_idx', type=str, required=True, help='Índices del conjunto de Test inédito')
parser.add_argument('--model_weights', type=str, required=True, help='Pesos (.pth) del modelo campeón')
parser.add_argument('--output_dir', type=str, default='resultados_finales', help='Carpeta para guardar gráficas y métricas')

# Configuración del Campeón (Debes poner las del modelo ganador)
parser.add_argument('--num_frames', type=int, required=True)
parser.add_argument('--num_dct', type=int, required=True)
parser.add_argument('--spatial', dest='use_spatial', action='store_true')
parser.add_argument('--no_spatial', dest='use_spatial', action='store_false')
parser.add_argument('--spectral', dest='use_spectral', action='store_true')
parser.add_argument('--no_spectral', dest='use_spectral', action='store_false')
parser.add_argument('--metrics', dest='use_metrics', action='store_true')
parser.add_argument('--no_metrics', dest='use_metrics', action='store_false')
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

# Crear carpeta de salida
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Iniciando Evaluación Final en {device}")

# Cargar el dataset de Test
test_idx = np.load(args.test_idx)
print(f"Evaluando sobre {len(test_idx)} videos completamente nuevos...")

test_dataset = DeepfakeHDF5Dataset(
    args.h5_path, test_idx, num_frames=args.num_frames, num_dct=args.num_dct,
    load_spatial=args.use_spatial, load_spectral=args.use_spectral, load_metrics=args.use_metrics
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Inicializar y cargar el modelo campeón
model = PrecomputedDeepfakeDetector(
    dct_input_dim=args.num_dct, spectral_hidden_dim=128, lstm_hidden=256, lstm_layers=1
).to(device)

print(f"Cargando pesos de: {args.model_weights}")
model.load_state_dict(torch.load(args.model_weights, map_location=device))
model.eval()

# Función auxiliar para tensores
def prepare_batch(spatial, dct, ssim, jitter, labels, device):
    B, T = labels.size(0), args.num_frames
    spatial = spatial.to(device) if args.use_spatial else torch.zeros(B, T, 512, device=device)
    dct = dct.to(device) if args.use_spectral else torch.zeros(B, T, args.num_dct, device=device)
    if args.use_metrics:
        ssim, jitter = ssim.to(device), jitter.to(device)
    else:
        ssim, jitter = torch.zeros(B, T, 1, device=device), torch.zeros(B, T, 1, device=device)
    return spatial, dct, ssim, jitter, labels.to(device)

all_preds_probs = []
all_true_labels = []

# ============================================================
# BUCLE DE INFERENCIA
# ============================================================
with torch.no_grad():
    for batch in test_loader:
        spatial, dct, ssim, jitter, labels = batch
        spatial, dct, ssim, jitter, labels = prepare_batch(spatial, dct, ssim, jitter, labels, device)
        
        outputs = model(spatial, dct, ssim, jitter)
        probs = torch.softmax(outputs, dim=1)[:, 1] # Probabilidad de ser Fake
        
        all_preds_probs.extend(probs.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# Convertir a numpy arrays
y_true = np.array(all_true_labels)
y_probs = np.array(all_preds_probs)
y_pred = (y_probs >= 0.333).astype(int)

# ============================================================
# CÁLCULO DE MÉTRICAS Y GRÁFICAS
# ============================================================

# 1. Reporte de Clasificación (Texto)
report = classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'], digits=4)
print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN (TEST SET)")
print("="*50)
print(report)

# Guardar métricas en un TXT
with open(os.path.join(args.output_dir, 'metricas_test_final.txt'), 'w') as f:
    f.write("REPORTE DE CLASIFICACIÓN (TEST SET)\n")
    f.write("=========================================\n")
    f.write(report)
    f.write("\nConfiguración del modelo evaluado:\n")
    f.write(f"- Pesos: {os.path.basename(args.model_weights)}\n")
    f.write(f"- Frames: {args.num_frames}\n")
    f.write(f"- DCT: {args.num_dct}\n")

# 2. Matriz de Confusión (Gráfica)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicción Real', 'Predicción Fake'],
            yticklabels=['Real', 'Fake'], annot_kws={"size": 16})
plt.title('Matriz de Confusión en Conjunto de Test', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Etiqueta Verdadera', fontsize=14)
plt.xlabel('Predicción del Modelo', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'matriz_confusion.png'), dpi=300)
plt.close()

# 3. Curva ROC (Gráfica)
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Precisión)', fontsize=14)
plt.ylabel('Tasa de Verdaderos Positivos (Recall)', fontsize=14)
plt.title('Curva ROC - Rendimiento de Detección Final', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'curva_roc.png'), dpi=300)
plt.close()

print(f"Evaluación completada. Gráficas y métricas guardadas en: {args.output_dir}")
