#!/usr/bin/env python3
"""
precompute_features.py  (v2 – GPU, vectorizado, tolerante a fallos)
==================================================================
Precomputa una sola vez las características espaciales (ResNet18 congelada)
y las métricas temporales (SSIM, Jitter) para todos los vídeos, y las guarda
junto con los DCT en un HDF5 ligero (~3.5 GB).

Uso:
  python precompute_features.py                # procesa el dataset completo
  python precompute_features.py --limit 10     # solo 10 vídeos (prueba rápida)
"""

import h5py
import numpy as np
import torch
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm
import os
import argparse

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS Y ARGUMENTOS
# ============================================================
USER_HOME = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project'
ORIGINAL_H5 = f'{USER_HOME}/data/processed/ff_dataset_max60frames_4096dct.h5'
OUTPUT_H5   = f'{USER_HOME}/data/processed/ff_features_precomputed.h5'

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=None,
                    help='Procesar solo los primeros N vídeos (para pruebas)')
parser.add_argument('--workers', type=int, default=0,
                    help='Workers del DataLoader. Usar 0 para lectura HDF5 más conservadora')
args = parser.parse_args()

if args.limit is not None and args.limit <= 0:
    parser.error('--limit debe ser un entero positivo')
if args.workers < 0:
    parser.error('--workers no puede ser negativo')

# Dispositivo GPU (si no hay, CPU, pero será lentísimo)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

# ============================================================
# 2. CARGA DEL MODELO ESPACIAL CONGELADO (ResNet18)
# ============================================================
print("Cargando ResNet18 preentrenada y congelada...")
resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = torch.nn.Identity()
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False
resnet.to(DEVICE)

# ============================================================
# 3. MÉTRICAS TEMPORALES VECTORIZADAS (SSIM + Jitter)
# ============================================================
def compute_metrics_vectorized(frames_batch):
    """
    Calcula SSIM y Jitter para un batch de secuencias.
    Entrada:
        frames_batch : (B, T, C, H, W) en float [0,1], RGB.
    Retorna:
        ssim_seq   : (B, T, 1)
        jitter_seq : (B, T, 1)
    """
    B, T, C, H, W = frames_batch.shape
    device = frames_batch.device

    # SSIM: comparar frames[:, 1:] con frames[:, :-1]
    prev = frames_batch[:, :-1].reshape(-1, C, H, W)
    curr = frames_batch[:, 1:].reshape(-1, C, H, W)

    # TorchMetrics puede devolver un escalar por imagen o un mapa; soportamos ambos casos.
    pair_count = B * (T - 1)
    ssim_raw = ssim(curr, prev, data_range=1.0, reduction='none')
    if ssim_raw.numel() == pair_count:
        ssim_values = ssim_raw.reshape(pair_count)
    else:
        ssim_values = ssim_raw.reshape(pair_count, -1).mean(dim=1)
    ssim_values = ssim_values.view(B, T - 1)   # (B, T-1)

    # Jitter: diferencia absoluta media frame a frame
    jitter_values = torch.mean(torch.abs(curr - prev), dim=[1, 2, 3])
    jitter_values = jitter_values.view(B, T - 1)   # (B, T-1)

    # Añadir el primer frame con SSIM=1.0 y Jitter=0.0
    ssim_full = torch.cat([torch.ones(B, 1, device=device), ssim_values], dim=1)
    jitter_full = torch.cat([torch.zeros(B, 1, device=device), jitter_values], dim=1)

    return ssim_full.unsqueeze(-1), jitter_full.unsqueeze(-1)   # (B, T, 1)

# ============================================================
# 4. DATASET QUE LEE LOS FRAMES ORIGINALES (UNA SOLA PASADA)
# ============================================================
class OriginalFramesDataset(Dataset):
    """Lee los frames (X) del HDF5 original."""
    def __init__(self, h5_path, limit=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.total = f['X'].shape[0]
        if limit and limit < self.total:
            self.total = limit
            print(f"Modo prueba: solo {self.total} vídeos")
        self.h5_file = None

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', rdcc_nbytes=0)
        # Leer los 60 frames y convertir a RGB tensor (T, C, H, W)
        x = self.h5_file['X'][idx][..., ::-1].copy()   # BGR->RGB
        frames = torch.from_numpy(x).float().permute(0, 3, 1, 2).contiguous()
        return frames


def copy_dataset_slice(src, dst, name, total, chunk_rows=64):
    """Copia solo los primeros `total` elementos sin cargar todo el dataset en RAM."""
    src_dset = src[name]
    shape = (total,) + src_dset.shape[1:]

    if name == 'X_dct':
        chunks = (1,) + src_dset.shape[1:]
        compression = 'lzf'
    elif len(shape) == 1:
        chunks = (min(total, 1024),)
        compression = None
    else:
        chunks = True
        compression = 'lzf'

    dst_dset = dst.create_dataset(
        name,
        shape=shape,
        dtype=src_dset.dtype,
        chunks=chunks,
        compression=compression
    )

    for start in range(0, total, chunk_rows):
        end = min(start + chunk_rows, total)
        dst_dset[start:end] = src_dset[start:end]

# ============================================================
# 5. PROCESO PRINCIPAL
# ============================================================
def main():
    with h5py.File(ORIGINAL_H5, 'r') as src:
        print("Dataset original:")
        print(f"  X:      shape={src['X'].shape}, dtype={src['X'].dtype}, chunks={src['X'].chunks}")
        print(f"  X_dct:  shape={src['X_dct'].shape}, dtype={src['X_dct'].dtype}, chunks={src['X_dct'].chunks}")

        total_original = src['X'].shape[0]
        total = total_original if args.limit is None else min(args.limit, total_original)
        T = 60

        # Crear el dataset de salida
        with h5py.File(OUTPUT_H5, 'w') as dst:
            dst.create_dataset('spatial_feat', (total, T, 512), dtype='float32',
                               chunks=(1, T, 512), compression='lzf')
            dst.create_dataset('ssim_feat',    (total, T, 1), dtype='float32',
                               chunks=(1, T, 1), compression='lzf')
            dst.create_dataset('jitter_feat',  (total, T, 1), dtype='float32',
                               chunks=(1, T, 1), compression='lzf')

            # Copiar X_dct, Y, video_id eficientemente y respetando --limit.
            for name in ['X_dct', 'Y', 'video_id']:
                copy_dataset_slice(src, dst, name, total)

            # Dataset de frames y DataLoader (batch_size=1 para evitar OOM)
            frame_ds = OriginalFramesDataset(ORIGINAL_H5, limit=args.limit)
            loader = DataLoader(frame_ds, batch_size=1, shuffle=False,
                                num_workers=args.workers, pin_memory=False)

            print(f"Precomputando {total} vídeos...")
            for idx, batch in enumerate(tqdm(loader, total=total)):
                frames = batch.squeeze(0).to(DEVICE)   # (60, 3, 224, 224)

                # Espacial: pasar los 60 frames de golpe por ResNet
                with torch.inference_mode():
                    spatial_feat = resnet(frames).cpu().numpy().astype('float32')  # (60, 512)

                # Métricas (vectorizadas en GPU)
                ssim_feat, jitter_feat = compute_metrics_vectorized(frames.unsqueeze(0))
                ssim_feat = ssim_feat.squeeze(0).cpu().numpy().astype('float32')
                jitter_feat = jitter_feat.squeeze(0).cpu().numpy().astype('float32')

                # Guardar en el HDF5 de salida
                dst['spatial_feat'][idx] = spatial_feat
                dst['ssim_feat'][idx] = ssim_feat
                dst['jitter_feat'][idx] = jitter_feat

    size_gb = os.path.getsize(OUTPUT_H5) / 1024**3
    print(f"Precomputación finalizada. Archivo: {OUTPUT_H5} ({size_gb:.2f} GB)")

if __name__ == '__main__':
    main()
