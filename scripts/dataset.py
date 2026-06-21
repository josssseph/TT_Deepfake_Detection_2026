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
        # rdcc_nbytes=0 apaga el caché de HDF5, previniendo fugas de RAM en HPC
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', rdcc_nbytes=0)
            
        real_idx = self.indices[idx]
        
        # 1. Leer SIEMPRE los coeficientes DCT (Son minúsculos en RAM)
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
        
        # 2. Leer las imágenes SOLO si la arquitectura actual las necesita
        if self.load_frames:
            # h5py lee mucho más rápido si le pasamos una lista ordenada de índices
            idx_sorted = np.sort(indices_frames).tolist()
            x_selected = self.h5_file['X'][real_idx, idx_sorted]
            
            x_rgb = x_selected[..., ::-1].copy()
            frames = torch.from_numpy(x_rgb).float()
            frames = frames.permute(0, 3, 1, 2)
        else:
            # Si es spectral_only, devolvemos un tensor vacío para no romper el unpacking
            frames = torch.empty(0)
            
        return frames, dct, label
