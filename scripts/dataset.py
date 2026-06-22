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
        
        # 2. Lectura Contigua de Imágenes (Solución al 0% de GPU)
        if self.load_frames:
            # Leemos TODOS los frames de un solo golpe (1 sola operación de disco)
            # Esto es muchísimo más rápido que pedir índices salteados al HDF5
            x_all = self.h5_file['X'][real_idx]
            
            # Submuestreamos usando numpy en la memoria ultra-rápida
            x_selected = x_all[indices_frames]
            
            x_rgb = x_selected[..., ::-1].copy()
            frames = torch.from_numpy(x_rgb).float()
            frames = frames.permute(0, 3, 1, 2)
        else:
            frames = torch.empty(0)
            
        return frames, dct, label
