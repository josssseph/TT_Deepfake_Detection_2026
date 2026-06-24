#!/usr/bin/env python3
"""
precompute_temporal_aligned.py
================================
Genera un HDF5 con metricas temporales alineadas al numero de frames usado.

El archivo conserva las caracteristicas ya precomputadas:
  - spatial_feat : (N, 60, 512)
  - X_dct        : (N, 60, 4096)
  - Y            : (N,)
  - video_id     : (N,)

Y agrega metricas temporales por configuracion:
  - temporal/f10/indices
  - temporal/f10/ssim_feat    : (N, 10, 1)
  - temporal/f10/jitter_feat  : (N, 10, 1)
  - temporal/f20/...
  - ...

Para cada f, primero se seleccionan f frames uniformemente entre los 60 y
luego se calculan SSIM/Jitter entre frames consecutivos de esa secuencia
seleccionada. Esto evita usar SSIM(frame t, frame t-1) cuando la LSTM en
realidad recibe saltos como frame 2 -> frame 4.
"""

import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm


USER_HOME = "/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project"
ORIGINAL_H5 = f"{USER_HOME}/data/processed/ff_dataset_max60frames_4096dct.h5"
FEATURES_H5 = f"{USER_HOME}/data/processed/ff_features_precomputed.h5"
OUTPUT_H5 = f"{USER_HOME}/data/processed/ff_features_aligned_temporal.h5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precomputa SSIM/Jitter alineados al submuestreo temporal"
    )
    parser.add_argument("--original_h5", type=str, default=ORIGINAL_H5,
                        help="HDF5 original con X=(N,60,224,224,3)")
    parser.add_argument("--features_h5", type=str, default=FEATURES_H5,
                        help="HDF5 precomputado con spatial_feat y X_dct")
    parser.add_argument("--output_h5", type=str, default=OUTPUT_H5,
                        help="HDF5 de salida con metricas temporales alineadas")
    parser.add_argument("--frames", type=int, nargs="+",
                        default=[10, 20, 30, 40, 50, 60],
                        help="Valores de num_frames para generar")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N videos para prueba")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Videos por batch. 1 es lo mas conservador")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers del DataLoader. 0 es lo mas estable con HDF5")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribir output_h5 si ya existe")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser positivo")
    if args.batch_size <= 0:
        parser.error("--batch_size debe ser positivo")
    if args.workers < 0:
        parser.error("--workers no puede ser negativo")
    if len(set(args.frames)) != len(args.frames):
        parser.error("--frames contiene valores repetidos")
    if any(f <= 1 for f in args.frames):
        parser.error("Cada valor en --frames debe ser mayor que 1")

    args.frames = sorted(args.frames)
    return args


class OriginalFramesDataset(Dataset):
    """Lee los frames RGB normalizados desde el HDF5 original."""

    def __init__(self, h5_path, total):
        self.h5_path = h5_path
        self.total = total
        self.h5_file = None

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", rdcc_nbytes=0)

        # X fue guardado como BGR float32 [0,1]: (60, 224, 224, 3)
        x = self.h5_file["X"][idx][..., ::-1].copy()
        frames = torch.from_numpy(x).float().permute(0, 3, 1, 2).contiguous()
        return torch.tensor(idx, dtype=torch.long), frames


def copy_dataset_slice(src, dst, name, total, chunk_rows=64):
    """Copia los primeros total elementos sin cargar todo en RAM."""
    src_dset = src[name]
    shape = (total,) + src_dset.shape[1:]

    if name == "spatial_feat":
        chunks = (1,) + src_dset.shape[1:]
        compression = "lzf"
    elif name == "X_dct":
        chunks = (1,) + src_dset.shape[1:]
        compression = "lzf"
    elif len(shape) == 1:
        chunks = (min(total, 1024),)
        compression = None
    else:
        chunks = True
        compression = "lzf"

    dst_dset = dst.create_dataset(
        name,
        shape=shape,
        dtype=src_dset.dtype,
        chunks=chunks,
        compression=compression,
    )

    for start in tqdm(range(0, total, chunk_rows), desc=f"Copiando {name}"):
        end = min(start + chunk_rows, total)
        dst_dset[start:end] = src_dset[start:end]


def build_frame_indices(total_frames, frame_counts):
    frame_indices = {}
    for count in frame_counts:
        if count > total_frames:
            raise ValueError(
                f"num_frames={count} excede los frames disponibles ({total_frames})"
            )
        frame_indices[count] = np.linspace(
            0, total_frames - 1, count, dtype=np.int64
        )
    return frame_indices


def compute_temporal_metrics(frames_batch):
    """
    frames_batch: (B, T, C, H, W), RGB float [0,1].
    Retorna:
      ssim_full:   (B, T, 1)
      jitter_full: (B, T, 1)
    """
    bsz, steps, channels, height, width = frames_batch.shape
    device = frames_batch.device

    prev = frames_batch[:, :-1].reshape(-1, channels, height, width)
    curr = frames_batch[:, 1:].reshape(-1, channels, height, width)
    pair_count = bsz * (steps - 1)

    ssim_raw = ssim(curr, prev, data_range=1.0, reduction="none")
    if ssim_raw.numel() == pair_count:
        ssim_values = ssim_raw.reshape(pair_count)
    else:
        ssim_values = ssim_raw.reshape(pair_count, -1).mean(dim=1)
    ssim_values = ssim_values.view(bsz, steps - 1)

    jitter_values = torch.mean(torch.abs(curr - prev), dim=[1, 2, 3])
    jitter_values = jitter_values.view(bsz, steps - 1)

    ssim_full = torch.cat(
        [torch.ones(bsz, 1, device=device), ssim_values],
        dim=1,
    ).unsqueeze(-1)
    jitter_full = torch.cat(
        [torch.zeros(bsz, 1, device=device), jitter_values],
        dim=1,
    ).unsqueeze(-1)

    return ssim_full, jitter_full


def contiguous_slice_from_rows(rows):
    rows = rows.detach().cpu().numpy()
    start = int(rows[0])
    end = int(rows[-1]) + 1
    expected = np.arange(start, end)
    if not np.array_equal(rows, expected):
        raise RuntimeError("El DataLoader debe usar shuffle=False para escritura contigua")
    return slice(start, end)


def main():
    args = parse_args()

    if os.path.exists(args.output_h5) and not args.overwrite:
        raise FileExistsError(
            f"Ya existe {args.output_h5}. Usa --overwrite para reemplazarlo."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Original H5: {args.original_h5}")
    print(f"Features H5: {args.features_h5}")
    print(f"Output H5:   {args.output_h5}")
    print(f"Frames:      {args.frames}")

    with h5py.File(args.original_h5, "r") as raw, h5py.File(args.features_h5, "r") as feat:
        total_raw = raw["X"].shape[0]
        total_feat = feat["spatial_feat"].shape[0]
        if total_raw != total_feat:
            raise ValueError(f"N no coincide: raw={total_raw}, features={total_feat}")

        total_frames = raw["X"].shape[1]
        total = total_raw if args.limit is None else min(args.limit, total_raw)
        frame_indices = build_frame_indices(total_frames, args.frames)

        print("Shapes de entrada:")
        print(f"  X:            shape={raw['X'].shape}, dtype={raw['X'].dtype}")
        print(f"  spatial_feat: shape={feat['spatial_feat'].shape}, dtype={feat['spatial_feat'].dtype}")
        print(f"  X_dct:        shape={feat['X_dct'].shape}, dtype={feat['X_dct'].dtype}")
        print(f"Videos a procesar: {total}")

        with h5py.File(args.output_h5, "w") as dst:
            dst.attrs["source_original_h5"] = args.original_h5
            dst.attrs["source_features_h5"] = args.features_h5
            dst.attrs["temporal_alignment"] = (
                "SSIM/Jitter recalculados despues del submuestreo uniforme"
            )

            for name in ["spatial_feat", "X_dct", "Y", "video_id"]:
                copy_dataset_slice(feat, dst, name, total)

            temporal_group = dst.create_group("temporal")
            for count, indices in frame_indices.items():
                group = temporal_group.create_group(f"f{count}")
                group.create_dataset("indices", data=indices.astype("int16"))
                group.create_dataset(
                    "ssim_feat",
                    shape=(total, count, 1),
                    dtype="float32",
                    chunks=(1, count, 1),
                    compression="lzf",
                )
                group.create_dataset(
                    "jitter_feat",
                    shape=(total, count, 1),
                    dtype="float32",
                    chunks=(1, count, 1),
                    compression="lzf",
                )

            dataset = OriginalFramesDataset(args.original_h5, total)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
            )

            print("Calculando metricas temporales alineadas...")
            for rows, frames in tqdm(loader, total=len(loader)):
                row_slice = contiguous_slice_from_rows(rows)
                frames = frames.to(device, non_blocking=False)

                with torch.inference_mode():
                    for count, indices in frame_indices.items():
                        index_tensor = torch.as_tensor(indices, device=device)
                        selected = frames.index_select(dim=1, index=index_tensor)
                        ssim_feat, jitter_feat = compute_temporal_metrics(selected)

                        group = temporal_group[f"f{count}"]
                        group["ssim_feat"][row_slice] = (
                            ssim_feat.detach().cpu().numpy().astype("float32")
                        )
                        group["jitter_feat"][row_slice] = (
                            jitter_feat.detach().cpu().numpy().astype("float32")
                        )

    size_gb = os.path.getsize(args.output_h5) / 1024 ** 3
    print(f"Archivo generado: {args.output_h5}")
    print(f"Tamanio final: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
