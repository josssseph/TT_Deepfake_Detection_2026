#!/bin/bash
#SBATCH -J temporal_aligned
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/temporal_aligned_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/temporal_aligned_%j.err

set -euo pipefail

echo "========================================="
echo "INICIO PRECOMPUTE TEMPORAL ALINEADO"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "========================================="

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

SCRIPT="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/precompute_temporal_aligned.py"
ORIGINAL_H5="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_dataset_max60frames_4096dct.h5"
FEATURES_H5="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_features_precomputed.h5"
OUTPUT_H5="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_features_aligned_temporal.h5"

LIMIT_ARGS=()
if [[ -n "${TEMPORAL_LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$TEMPORAL_LIMIT")
    echo "Modo prueba: TEMPORAL_LIMIT=$TEMPORAL_LIMIT"
fi

BATCH_SIZE="${TEMPORAL_BATCH_SIZE:-1}"
WORKERS="${TEMPORAL_WORKERS:-0}"

echo "Batch size: $BATCH_SIZE"
echo "Workers: $WORKERS"
echo "Output: $OUTPUT_H5"

python -u "$SCRIPT" \
    --original_h5 "$ORIGINAL_H5" \
    --features_h5 "$FEATURES_H5" \
    --output_h5 "$OUTPUT_H5" \
    --frames 10 20 30 40 50 60 \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --overwrite \
    "${LIMIT_ARGS[@]}"

echo "========================================="
echo "FIN PRECOMPUTE TEMPORAL ALINEADO"
echo "Fecha: $(date)"
echo "========================================="
