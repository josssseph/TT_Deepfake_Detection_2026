import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

#h5_path = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_dataset_30frames.h5'
h5_path = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_dataset_max60frames_4096dct.h5'
save_dir = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/'

print("Abriendo dataset HDF5...")
with h5py.File(h5_path, 'r') as f:
    video_ids = f['video_id'][:]

print("Agrupando por video_id original (aislamiento de identidades)...")
vid_to_indices = defaultdict(list)
for idx, vid in enumerate(video_ids):
    vid_to_indices[vid].append(idx)

unique_vids = list(vid_to_indices.keys())

print("Realizando división exacta: 70% Train / 15% Val / 15% Test...")

# 1. Separamos 15% para Test (queda 85% para Train+Val)
train_val_vids, test_vids = train_test_split(
    unique_vids,
    test_size=0.15,
    random_state=42
)

# 2. Del 85% restante, sacamos la proporción para el 15% de validación (0.15 / 0.85 = 0.17647)
val_ratio = 0.15 / 0.85
train_vids, val_vids = train_test_split(
    train_val_vids,
    test_size=val_ratio,
    random_state=42
)

# 3. Expansión: recuperar todos los frames (reales y fakes) de esos IDs
train_indices = [i for vid in train_vids for i in vid_to_indices[vid]]
val_indices   = [i for vid in val_vids for i in vid_to_indices[vid]]
test_indices  = [i for vid in test_vids for i in vid_to_indices[vid]]

print(f"Guardando archivos en {save_dir}...")

np.save(os.path.join(save_dir, 'train_indices.npy'), train_indices)
np.save(os.path.join(save_dir, 'val_indices.npy'), val_indices)
np.save(os.path.join(save_dir, 'test_indices.npy'), test_indices)

print(
    f"Archivos generados. Muestras -> "
    f"Train: {len(train_indices)}, "
    f"Val: {len(val_indices)}, "
    f"Test: {len(test_indices)}"
)
