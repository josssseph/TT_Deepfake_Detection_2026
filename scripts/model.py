import torch
import torch.nn as nn

class PrecomputedDeepfakeDetector(nn.Module):
    """
    Modelo ligero para características precomputadas.

    Entradas (todas con forma (B, T, ...)):
      - spatial_feat : (B, T, 512)   ← ya extraído por la ResNet congelada
      - dct_coeffs   : (B, T, num_dct)
      - ssim_feat    : (B, T, 1)
      - jitter_feat  : (B, T, 1)

    Las ramas espacial y de métricas temporales ya vienen dadas.
    Solo la rama espectral (MLP) y la LSTM + clasificador son entrenables.
    """
    def __init__(
        self,
        dct_input_dim=1024,
        spectral_hidden_dim=128,
        lstm_hidden=256,
        lstm_layers=1,
        use_spatial=True,
        use_spectral=True,
        use_metrics=True
    ):
        super().__init__()

        # Guardamos la dimensión de salida espectral para uso externo (si se necesita)
        self.spectral_hidden_dim = spectral_hidden_dim
        self.use_spatial = use_spatial
        self.use_spectral = use_spectral
        self.use_metrics = use_metrics

        # ==========================================
        # Única rama entrenable: MLP para DCT
        # ==========================================
        self.spectral_branch = nn.Sequential(
            nn.Linear(dct_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, spectral_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ==========================================
        # Entrada a la LSTM:
        # 512 (espacial) + spectral_hidden_dim + 1 (SSIM) + 1 (Jitter)
        # ==========================================
        input_features = 512 + spectral_hidden_dim + 2

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, spatial_feat, dct_coeffs, ssim_feat, jitter_feat):
        # spatial_feat, ssim_feat, jitter_feat ya vienen con la forma correcta
        if self.use_spatial:
            B, T, _ = spatial_feat.shape
        elif self.use_spectral:
            B, T, _ = dct_coeffs.shape
        elif self.use_metrics:
            B, T, _ = ssim_feat.shape
        else:
            raise RuntimeError("Al menos una modalidad debe estar activa.")

        if self.use_spatial:
            device = spatial_feat.device
        elif self.use_spectral:
            device = dct_coeffs.device
        else:
            device = ssim_feat.device

        if not self.use_spatial:
            spatial_feat = torch.zeros(B, T, 512, device=device)

        # Rama espectral: solo se ejecuta cuando la modalidad DCT esta activa.
        if self.use_spectral:
            dct_flat = dct_coeffs.view(B * T, -1)
            spectral_feat = self.spectral_branch(dct_flat).view(B, T, self.spectral_hidden_dim)
        else:
            spectral_feat = torch.zeros(B, T, self.spectral_hidden_dim, device=device)

        if not self.use_metrics:
            ssim_feat = torch.zeros(B, T, 1, device=device)
            jitter_feat = torch.zeros(B, T, 1, device=device)

        # Concatenamos todo
        combined = torch.cat([spatial_feat, spectral_feat, ssim_feat, jitter_feat], dim=2)

        lstm_out, (hn, _) = self.lstm(combined)
        video_summary = hn[-1]   # última capa de la LSTM

        logits = self.classifier(video_summary)
        return logits
