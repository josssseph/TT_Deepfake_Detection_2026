import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# Importamos nuestros módulos personalizados
from dataset import DeepfakeHDF5Dataset
from model import DeepfakeDetector

# ==========================================
# 1. CONFIGURACIÓN E HIPERPARÁMETROS
# ==========================================
H5_PATH = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_dataset_30frames.h5'
MODEL_SAVE_PATH = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/best_deepfake_detector.pth'

BATCH_SIZE = 8        # Lote pequeño/mediano por la carga de videos (30 frames)
EPOCHS = 30           # Pasadas completas por el dataset
LEARNING_RATE = 1e-4  # Velocidad de aprendizaje (suave para no dañar ResNet)
NUM_WORKERS = 4        # Usamos 4 núcleos de CPU para leer rápido el HDF5

# Nos aseguramos de que la carpeta models exista
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ==========================================
# 2. PREPARACIÓN DE DATOS Y ENTORNO
# ==========================================
# Detectamos si la A100 está disponible (En SLURM dirá que sí)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Iniciando entrenamiento en: {device}")

# Cargamos los índices previamente divididos (Hold-out aislado)
train_idx = np.load('/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_indices.npy')
val_idx   = np.load('/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/val_indices.npy')

print(f"Videos para Entrenar: {len(train_idx)} | Validar: {len(val_idx)}")

# Creamos Datasets y DataLoaders
train_dataset = DeepfakeHDF5Dataset(H5_PATH, train_idx)
val_dataset   = DeepfakeHDF5Dataset(H5_PATH, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

# ==========================================
# 3. INICIALIZACIÓN DEL MODELO
# ==========================================
model = DeepfakeDetector().to(device)

# Función de pérdida (Binary Cross Entropy pero optimizada para 2 clases)
criterion = nn.CrossEntropyLoss()

# Optimizador AdamW (mejorado para evitar sobreajuste)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Acelerador de GPU para las A100 (Automatic Mixed Precision)
scaler = torch.amp.GradScaler('cuda')

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO (TRAINING LOOP)
# ==========================================
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.time()

    # ---- FASE DE ENTRENAMIENTO ----
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (frames, dct, labels) in enumerate(train_loader):
        frames, dct, labels = frames.to(device), dct.to(device), labels.to(device)

        optimizer.zero_grad()  # Limpiamos gradientes anteriores

        # Mixed Precision: Calcula matemática en 16-bits para máxima velocidad en la A100
        with torch.amp.autocast('cuda'):
            outputs = model(frames, dct)
            loss = criterion(outputs, labels)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Estadísticas
        running_loss += loss.item() * frames.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data).item()
        total_train += labels.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train

    # ---- FASE DE VALIDACIÓN ----
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # Apagamos el cálculo de gradientes (ahorra muchísima RAM)
        for frames, dct, labels in val_loader:
            frames, dct, labels = frames.to(device), dct.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(frames, dct)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * frames.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data).item()
            total_val += labels.size(0)

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val

    end_time = time.time()

    # Imprimimos resumen de la época
    print(f"Época [{epoch+1}/{EPOCHS}] - Tiempo: {end_time - start_time:.1f}s")
    print(f"  Train -> Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}")
    print(f"  Val   -> Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}")

    # Guardar el mejor modelo (Si el error de validación baja, guardamos esta versión)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  Nuevo mejor modelo guardado!")

    print("-" * 50)

print("Entrenamiento completado. Mejor modelo asegurado.")
