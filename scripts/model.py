import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        dct_input_dim=1024,
        spectral_hidden_dim=128,
        lstm_hidden=256,
        lstm_layers=1
    ):
        super(DeepfakeDetector, self).__init__()

        # Guardamos explícitamente la dimensión de salida de la rama espectral
        # (necesario para train_tuning.py cuando se usa --no_spectral)
        self.spectral_hidden_dim = spectral_hidden_dim

        # ==========================================
        # 1. RAMA ESPACIAL (ResNet18 Moderno)
        # ==========================================
        self.spatial_branch = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )
        self.spatial_branch.fc = nn.Identity()

        # Congelar pesos de ResNet
        for param in self.spatial_branch.parameters():
            param.requires_grad = False

        # ==========================================
        # 2. RAMA ESPECTRAL (MLP)
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
        # 3. RAMA TEMPORAL (LSTM + Clasificador)
        # ==========================================
        # 512 (Espacial) + spectral_hidden_dim + 1 (SSIM) + 1 (Jitter)
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

    def extract_metrics(self, frames):
        """
        Calcula SSIM y Jitter para cada frame respecto al anterior.
        """
        B, T, C, H, W = frames.shape
        device = frames.device

        ssim_seq = [torch.ones(B, 1, device=device)]
        jitter_seq = [torch.zeros(B, 1, device=device)]

        # ¡ESTO SALVARÁ TU RAM Y VRAM!
        with torch.no_grad():
            for t in range(1, T):
                frame_actual = frames[:, t]
                frame_previo = frames[:, t - 1]

                ssim_val = ssim(
                    frame_actual,
                    frame_previo,
                    data_range=1.0,
                    reduction="none"
                )

                ssim_seq.append(ssim_val.unsqueeze(1))

                jitter_val = torch.mean(
                    torch.abs(frame_actual - frame_previo),
                    dim=[1, 2, 3]
                )

                jitter_seq.append(jitter_val.unsqueeze(1))

        ssim_feat = torch.stack(ssim_seq, dim=1)
        jitter_feat = torch.stack(jitter_seq, dim=1)

        return ssim_feat, jitter_feat

    def forward(self, frames, dct_coeffs):
        B, T, C, H, W = frames.shape

        # ==========================================
        # Extracción espacial
        # ==========================================
        frames_reshaped = frames.view(B * T, C, H, W)

        spatial_feat = self.spatial_branch(frames_reshaped)
        spatial_feat = spatial_feat.view(B, T, 512)

        # ==========================================
        # Extracción espectral
        # ==========================================
        spectral_feat = self.spectral_branch(dct_coeffs)

        # ==========================================
        # Métricas temporales
        # ==========================================
        ssim_feat, jitter_feat = self.extract_metrics(frames)

        # ==========================================
        # Fusión multimodal
        # ==========================================
        combined_features = torch.cat(
            [
                spatial_feat,
                spectral_feat,
                ssim_feat,
                jitter_feat
            ],
            dim=2
        )

        # ==========================================
        # LSTM
        # ==========================================
        lstm_out, (hn, cn) = self.lstm(combined_features)

        # Tomar la última capa de la LSTM
        video_summary = hn[-1]

        # ==========================================
        # Clasificación final
        # ==========================================
        logits = self.classifier(video_summary)

        return logits
