import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class DeepfakeHDF5Dataset(Dataset):
    """
    Dataset para el archivo de características precomputadas.
    Cada vídeo tiene 60 frames de características:
      - spatial_feat : (60, 512)   si load_spatial=True
      - X_dct        : (60, 4096)  si load_spectral=True
      - ssim_feat    : (60, 1)     si load_metrics=True
      - jitter_feat  : (60, 1)     si load_metrics=True
    Devuelve siempre cinco elementos:
      spatial, dct, ssim, jitter, label
    Si una modalidad esta desactivada, su tensor se devuelve vacio con forma
    (T, 0). train_tuning.py lo reemplaza por ceros de la dimension esperada.
    """
    def __init__(self, h5_path, indices, num_frames=30, num_dct=1024,
                 load_spatial=True, load_spectral=True, load_metrics=True):
        self.h5_path = h5_path
        self.indices = indices
        self.num_frames = num_frames
        self.num_dct = num_dct
        self.load_spatial = load_spatial
        self.load_spectral = load_spectral
        self.load_metrics = load_metrics
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            # Abrir sin caché para no repetir errores de memoria
            self.h5_file = h5py.File(self.h5_path, 'r', rdcc_nbytes=0)

        real_idx = self.indices[idx]

        # 1. Cargar label (siempre)
        y_label = self.h5_file['Y'][real_idx]

        # 2. Submuestreo uniforme de frames (igual que antes)
        # Usamos X_dct como referencia para la longitud temporal (60)
        total_frames = self.h5_file['X_dct'].shape[1]  # siempre 60
        if self.num_frames < total_frames:
            indices_frames = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices_frames = np.arange(total_frames)

        # 3. Cargar las características solicitadas. Si una modalidad esta
        # desactivada devolvemos un tensor vacio para mantener la interfaz fija.
        T = len(indices_frames)

        if self.load_spatial:
            spatial = self.h5_file['spatial_feat'][real_idx, indices_frames]  # (T, 512)
            spatial = torch.from_numpy(spatial).float()
        else:
            spatial = torch.empty(T, 0)

        if self.load_spectral:
            dct = self.h5_file['X_dct'][real_idx, indices_frames, :self.num_dct]  # (T, num_dct)
            dct = torch.from_numpy(dct).float()
        else:
            dct = torch.empty(T, 0)

        if self.load_metrics:
            ssim = self.h5_file['ssim_feat'][real_idx, indices_frames]      # (T, 1)
            jitter = self.h5_file['jitter_feat'][real_idx, indices_frames]  # (T, 1)
            ssim = torch.from_numpy(ssim).float()
            jitter = torch.from_numpy(jitter).float()
        else:
            ssim = torch.empty(T, 0)
            jitter = torch.empty(T, 0)

        label = torch.tensor(y_label, dtype=torch.long)

        return spatial, dct, ssim, jitter, label
