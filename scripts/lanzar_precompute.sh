#!/bin/bash
#SBATCH -J precompute_features
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/precompute_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/precompute_%j.err

set -euo pipefail

echo "========================================="
echo "Inicio de precomputación de características"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "========================================="

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

# Para prueba corta:
#   sbatch --export=ALL,PRECOMPUTE_LIMIT=10 lanzar_precompute.sh
LIMIT_ARGS=()
if [[ -n "${PRECOMPUTE_LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$PRECOMPUTE_LIMIT")
    echo "Modo prueba activado: PRECOMPUTE_LIMIT=$PRECOMPUTE_LIMIT"
fi

WORKERS="${PRECOMPUTE_WORKERS:-0}"
echo "Workers DataLoader: $WORKERS"

# Ejecutar sin buffering para ver el progreso en tiempo real
python -u ~/deepfake_project/scripts/precompute_features.py \
    --workers "$WORKERS" \
    "${LIMIT_ARGS[@]}"

echo "========================================="
echo "Precomputación finalizada"
echo "Fecha: $(date)"
echo "========================================="
