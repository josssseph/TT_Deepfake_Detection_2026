#!/bin/bash
#SBATCH -J tuning_aligned
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/tuning_aligned_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/tuning_aligned_%j.err

set -euo pipefail

echo "========================================="
echo "INICIANDO TUNING CON TEMPORAL ALINEADO"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "========================================="

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

H5_PATH="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_features_aligned_temporal.h5"
TRAIN_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_indices.npy"
VAL_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/val_indices.npy"
CSV_FILE="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/resultados_tuning_aligned.csv"
MODEL_DIR="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/aligned"
mkdir -p "$MODEL_DIR"

BASE_SCRIPT="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_tuning.py"
BASE_LR=1e-4
BASE_BATCH=64
BASE_WORKERS=4
BASE_EPOCHS=80
BASE_PATIENCE=15
BASE_LSTM_HIDDEN=256
BASE_LSTM_LAYERS=1
BASE_SPECTRAL_HIDDEN=128
BASE_EARLY_METRIC="auc"

FRAMES_LIST=(10 20 30 40 50 60)
DCT_LIST=(256 512 1024 2048 4096)
ARCH_LIST=(full_model spatial_spectral)

total_experiments=$((${#FRAMES_LIST[@]} * ${#DCT_LIST[@]} * ${#ARCH_LIST[@]}))
current_experiment=0

for arch in "${ARCH_LIST[@]}"; do
    case "$arch" in
        full_model)
            SPATIAL="--spatial"; SPECTRAL="--spectral"; METRICS="--metrics" ;;
        spatial_spectral)
            SPATIAL="--spatial"; SPECTRAL="--spectral"; METRICS="--no_metrics" ;;
        *)
            echo "Arquitectura no soportada: $arch" >&2
            exit 1 ;;
    esac

    for frames in "${FRAMES_LIST[@]}"; do
        for dct in "${DCT_LIST[@]}"; do
            current_experiment=$((current_experiment + 1))
            exp_name="aligned_${arch}_f${frames}_dct${dct}"

            echo "========================================="
            echo "[$current_experiment/$total_experiments] Entrenando $exp_name"
            echo "Arquitectura: $arch | frames=$frames | dct=$dct"
            echo "========================================="

            python -u "$BASE_SCRIPT" \
                --h5_path "$H5_PATH" \
                --train_idx "$TRAIN_IDX" \
                --val_idx "$VAL_IDX" \
                --results_csv "$CSV_FILE" \
                --exp_name "$exp_name" \
                --save_model "$MODEL_DIR/${exp_name}.pth" \
                --num_frames "$frames" \
                --num_dct "$dct" \
                --lr "$BASE_LR" \
                --batch_size "$BASE_BATCH" \
                --num_workers "$BASE_WORKERS" \
                --epochs "$BASE_EPOCHS" \
                --patience "$BASE_PATIENCE" \
                --lstm_hidden "$BASE_LSTM_HIDDEN" \
                --lstm_layers "$BASE_LSTM_LAYERS" \
                --spectral_hidden_dim "$BASE_SPECTRAL_HIDDEN" \
                --early_stop_metric "$BASE_EARLY_METRIC" \
                $SPATIAL $SPECTRAL $METRICS
        done
    done
done

echo "========================================="
echo "TUNING ALINEADO FINALIZADO"
echo "Fecha: $(date)"
echo "Resultados: $CSV_FILE"
echo "Modelos: $MODEL_DIR"
echo "========================================="
