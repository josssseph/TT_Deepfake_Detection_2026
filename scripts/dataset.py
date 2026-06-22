import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class DeepfakeHDF5Dataset(Dataset):
    def __init__(self, h5_path, indices, num_frames=30, num_dct=1024, load_frames=True):
        self.h5_path = h5_path
        self.indices = indices
        self.num_frames = num_frames
        self.num_dct = num_dct
        self.load_frames = load_frames
        self.h5_file = None  

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            # swmr=True ayuda a múltiples workers a leer sin bloquearse
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=0)
            
        real_idx = self.indices[idx]
        
        # 1. Leer DCT y label
        dct_all = self.h5_file['X_dct'][real_idx]
        y_label = self.h5_file['Y'][real_idx]
        
        total_available = dct_all.shape[0]
        
        if self.num_frames < total_available:
            indices_frames = np.linspace(0, total_available - 1, self.num_frames, dtype=int)
            dct_selected = dct_all[indices_frames, :self.num_dct]
        else:
            indices_frames = np.arange(total_available)
            dct_selected = dct_all[:, :self.num_dct]
            
        dct = torch.from_numpy(dct_selected).float()
        label = torch.tensor(y_label, dtype=torch.long)
        
        # 2. Lectura Eficiente de Imágenes (Evitando OOM)
        if self.load_frames:
            # En lugar de cargar TODO el video a la RAM (x_all = h5_file['X'][real_idx])
            # Le pedimos a h5py que nos traiga SOLO los frames que necesitamos.
            # h5py requiere que los índices estén en una lista ordenada
            indices_ordenados = np.sort(indices_frames).tolist()
            
            # Leemos solo los frames necesarios directamente del disco
            x_selected = self.h5_file['X'][real_idx, indices_ordenados]
            
            # x_selected ahora tiene forma (num_frames_necesarios, 224, 224, 3)
            # Reordenamos si los índices originales no estaban ordenados (poco probable, pero seguro)
            if not np.array_equal(indices_frames, indices_ordenados):
                # Encontramos la posición original de los índices ordenados
                mapping = {val: i for i, val in enumerate(indices_ordenados)}
                reorder_idx = [mapping[val] for val in indices_frames]
                x_selected = x_selected[reorder_idx]

            x_rgb = x_selected[..., ::-1].copy()
            frames = torch.from_numpy(x_rgb).float()
            frames = frames.permute(0, 3, 1, 2)
        else:
            frames = torch.empty(0)            
        return frames, dct, label
