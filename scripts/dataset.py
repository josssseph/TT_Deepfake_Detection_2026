import torch
from torch.utils.data import Dataset
import h5py

class DeepfakeHDF5Dataset(Dataset):
    def __init__(self, h5_path, indices):
        """
        Args:
            h5_path: Ruta al archivo .h5 masivo.
            indices: Lista o array de numpy con los índices específicos 
                     (de train, val o test) que este dataset debe leer.
        """
        self.h5_path = h5_path
        self.indices = indices
        self.h5_file = None  # Lazy loading: se abrirá en el primer __getitem__

    def __len__(self):
        # El tamaño ya no es el total del HDF5, sino el tamaño del subset (train/val/test)
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. Apertura segura para evitar bloqueos de HDF5 con multiprocesamiento
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        # 2. Obtener el índice real dentro del HDF5 basado en la división
        real_idx = self.indices[idx]
        
        # 3. Leer los datos desde el disco a NumPy
        x_numpy = self.h5_file['X'][real_idx]        # Shape: (30, 224, 224, 3) en BGR
        dct_numpy = self.h5_file['X_dct'][real_idx]  # Shape: (30, 1024)
        y_label = self.h5_file['Y'][real_idx]        # Shape: ()
        
        # 4. Corregir el color (OpenCV BGR -> RGB) y crear copia en memoria
        x_rgb = x_numpy[..., ::-1].copy()
        
        # 5. Convertir a tensores de PyTorch
        frames = torch.from_numpy(x_rgb).float()
        dct = torch.from_numpy(dct_numpy).float()
        label = torch.tensor(y_label, dtype=torch.long)
        
        # 6. Reordenar dimensiones de frames para PyTorch: (30, H, W, C) -> (30, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
        
        return frames, dct, label
