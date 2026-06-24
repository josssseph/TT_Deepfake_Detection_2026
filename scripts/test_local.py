import torch
from model import DeepfakeDetector

print("Cargando el modelo definitivo (PyTorch 2.5+)...")
model = DeepfakeDetector()

print("Generando tensores de prueba...")
dummy_videos = torch.rand(2, 30, 3, 224, 224) 
dummy_dct = torch.randn(2, 30, 1024)

print("Ejecutando Forward Pass Completo (Espacial + Espectral + SSIM + Jitter + LSTM)...")
predicciones = model(dummy_videos, dummy_dct)

print(f"\n✅ FORMA FINAL: {predicciones.shape} (Esperado: [2, 2])")
print(predicciones)
