#!/bin/bash
#SBATCH -J tuning_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/tuning_v2_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/tuning_v2_%j.err

set -euo pipefail

echo "========================================="
echo "INICIANDO TUNING V2: REPRODUCIBLE Y DUAL"
echo "Fecha: $(date)"
echo "========================================="

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

H5_PATH="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_features_aligned_temporal.h5"
TRAIN_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_indices.npy"
VAL_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/val_indices.npy"
CSV_FILE="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/resultados_tuning_v2.csv"
MODEL_DIR="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/aligned_v2"
mkdir -p "$MODEL_DIR"

BASE_SCRIPT="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_tuning_v2.py"

# Redujimos el grid a los valores que sabemos matemáticamente que funcionan
# (DCT bajos/medios y Frames medios/altos) para ahorrar tiempo y recursos.
FRAMES_LIST=(20 30 40 50 60)
DCT_LIST=(256 512 1024)
ARCH_LIST=(full_model spatial_spectral)

for arch in "${ARCH_LIST[@]}"; do
    case "$arch" in
        full_model)
            SPATIAL="--spatial"; SPECTRAL="--spectral"; METRICS="--metrics" ;;
        spatial_spectral)
            SPATIAL="--spatial"; SPECTRAL="--spectral"; METRICS="--no_metrics" ;;
    esac

    for frames in "${FRAMES_LIST[@]}"; do
        for dct in "${DCT_LIST[@]}"; do
            exp_name="v2_${arch}_f${frames}_dct${dct}"
            
            # Verificación para evitar correr si ya está en el CSV (Si se pausa el HPC)
            if grep -q "^$exp_name," "$CSV_FILE" 2>/dev/null; then
                echo "⏭️  Saltando $exp_name (Ya existe en CSV)"
                continue
            fi

            echo "🚀 Entrenando $exp_name"
            
            python -u "$BASE_SCRIPT" \
                --h5_path "$H5_PATH" \
                --train_idx "$TRAIN_IDX" \
                --val_idx "$VAL_IDX" \
                --results_csv "$CSV_FILE" \
                --exp_name "$exp_name" \
                --save_model_dir "$MODEL_DIR" \
                --num_frames "$frames" \
                --num_dct "$dct" \
                --lr 1e-4 \
                --batch_size 64 \
                --epochs 50 \
                --patience 10 \
                $SPATIAL $SPECTRAL $METRICS
        done
    done
done

echo "========================================="
echo "TUNING V2 FINALIZADO CON ÉXITO"
echo "========================================="
