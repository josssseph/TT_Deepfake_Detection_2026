import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class DeepfakeHDF5Dataset(Dataset):
    def __init__(self, h5_path, indices, num_frames=30, num_dct=1024):
        """
        Args:
            h5_path: Ruta al archivo .h5 masivo.
            indices: Lista o array de numpy con los índices específicos 
                     (de train, val o test) que este dataset debe leer.
            num_frames: Número de frames a utilizar por video (submuestreo uniforme).
            num_dct: Número de coeficientes DCT a conservar por frame.
        """
        self.h5_path = h5_path
        self.indices = indices
        self.num_frames = num_frames
        self.num_dct = num_dct
        self.h5_file = None  # Lazy loading: se abrirá en el primer __getitem__

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        real_idx = self.indices[idx]
        
        # Leer los datos completos (60 frames y 4096 DCT)
        x_all = self.h5_file['X'][real_idx]          # (60, 224, 224, 3) BGR
        dct_all = self.h5_file['X_dct'][real_idx]    # (60, 4096)
        y_label = self.h5_file['Y'][real_idx]        # ()
        
        # Submuestreo uniforme de frames (índices equidistantes)
        total_available = x_all.shape[0]  # siempre 60
        if self.num_frames < total_available:
            indices_frames = np.linspace(0, total_available - 1, self.num_frames, dtype=int)
            x_selected = x_all[indices_frames]
            dct_selected = dct_all[indices_frames, :self.num_dct]
        else:
            # Si pedimos exactamente 60 frames, usamos todos
            x_selected = x_all
            dct_selected = dct_all[:, :self.num_dct]
        
        # Corregir el color (BGR a RGB) y hacer copia en memoria
        x_rgb = x_selected[..., ::-1].copy()
        
        # Convertir a tensores PyTorch
        frames = torch.from_numpy(x_rgb).float()
        dct = torch.from_numpy(dct_selected).float()
        label = torch.tensor(y_label, dtype=torch.long)
        
        # Reordenar dimensiones de frames: (T, H, W, C) -> (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
        
        return frames, dct, label
