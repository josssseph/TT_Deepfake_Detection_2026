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
      - temporal/f{T}/ssim_feat    : (T, 1) si load_metrics=True
      - temporal/f{T}/jitter_feat  : (T, 1) si load_metrics=True
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
        self.indices_frames = None
        self.temporal_group_name = f"temporal/f{num_frames}"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            # Abrir sin caché para no repetir errores de memoria
            self.h5_file = h5py.File(self.h5_path, 'r', rdcc_nbytes=0)
            self.indices_frames = self._load_frame_indices()

        real_idx = self.indices[idx]

        # 1. Cargar label (siempre)
        y_label = self.h5_file['Y'][real_idx]

        # 2. Submuestreo uniforme de frames. Si existe el grupo temporal/f{T},
        # usamos sus indices para asegurar coherencia con SSIM/Jitter alineados.
        indices_frames = self.indices_frames

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
            if self.temporal_group_name in self.h5_file:
                group = self.h5_file[self.temporal_group_name]
                ssim = group['ssim_feat'][real_idx]      # (T, 1)
                jitter = group['jitter_feat'][real_idx]  # (T, 1)
            else:
                # Compatibilidad con el HDF5 precomputado anterior.
                ssim = self.h5_file['ssim_feat'][real_idx, indices_frames]
                jitter = self.h5_file['jitter_feat'][real_idx, indices_frames]
            ssim = torch.from_numpy(ssim).float()
            jitter = torch.from_numpy(jitter).float()
        else:
            ssim = torch.empty(T, 0)
            jitter = torch.empty(T, 0)

        label = torch.tensor(y_label, dtype=torch.long)

        return spatial, dct, ssim, jitter, label

    def _load_frame_indices(self):
        if self.temporal_group_name in self.h5_file:
            return self.h5_file[self.temporal_group_name]['indices'][:].astype(int)

        total_frames = self.h5_file['X_dct'].shape[1]
        if self.num_frames < total_frames:
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return np.arange(total_frames)
