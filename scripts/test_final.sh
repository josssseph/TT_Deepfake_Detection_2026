#!/bin/bash
#SBATCH -J test_final
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/test_final_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/test_final_%j.err

set -euo pipefail

echo "========================================="
echo "INICIANDO EVALUACIÓN FINAL EN TEST"
echo "Fecha: $(date)"
echo "========================================="

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

H5_PATH="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_features_aligned_temporal.h5"
TEST_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/test_indices.npy"
OUTPUT_DIR="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/resultados_finales"

# =========================================================================
# CONFIGURACIÓN DEL MODELO CAMPEÓN (spatial_spectral)
# =========================================================================
#CAMPEON_WEIGHTS="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/finetuned/ft_full_model_bs64_lr5e-5_wd1e-4.pth"
CAMPEON_WEIGHTS="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/finetuned/ft_spatial_spectral_bs64_lr5e-5_wd1e-3.pth"

# Estos valores debes confirmarlos con el CSV del mejor modelo.
# Ajusta NUM_FRAMES y NUM_DCT según los usados en el fine-tuning ganador.
NUM_FRAMES=50
NUM_DCT=256

# Arquitectura: solo espacial y espectral, sin métricas temporales
ARCH_FLAGS="--spatial --spectral --no_metrics"

# =========================================================================

python -u /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/evaluar_test.py \
    --h5_path "$H5_PATH" \
    --test_idx "$TEST_IDX" \
    --model_weights "$CAMPEON_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    --num_dct "$NUM_DCT" \
    $ARCH_FLAGS \
    --batch_size 64

echo "========================================="
echo "EVALUACIÓN FINAL COMPLETADA"
echo "Revisa la carpeta: $OUTPUT_DIR"
echo "========================================="
