#!/bin/bash
#SBATCH -J ft_grid_search
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/ft_grid_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/ft_grid_%j.err

set -euo pipefail

echo "========================================="
echo "INICIANDO FINE-TUNING QUIRÚRGICO (MICRO-GRID)"
echo "Rigor Científico: Semilla Fija (42) y Base de Mínima Pérdida"
echo "Fecha: $(date)"
echo "========================================="

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

H5_PATH="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_features_aligned_temporal.h5"
TRAIN_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_indices.npy"
VAL_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/val_indices.npy"
CSV_FILE="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/resultados_finetuning_expert.csv"

# Carpeta donde se guardarán los resultados del Fine-Tuning
OUTPUT_DIR="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/finetuned"
mkdir -p "$OUTPUT_DIR"

BASE_SCRIPT="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/finetuning_expert.py"

# Los pesos pre-entrenados que ganaron en la V2 (los de MÍNIMA PÉRDIDA)
# Justificación: Partimos de la estabilidad matemática, no del sobreajuste.
WEIGHTS_DIR="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models/aligned_v2"
MODEL_1_WEIGHTS="$WEIGHTS_DIR/v2_full_model_f50_dct256_best_loss.pth"
MODEL_2_WEIGHTS="$WEIGHTS_DIR/v2_spatial_spectral_f50_dct256_best_loss.pth"

# Hiperparámetros a explorar
BATCH_SIZES=(32 64)
LRS=(5e-5 1e-5)
WDS=(1e-3 1e-4)
SEED=42 # Garantiza reproducibilidad absoluta en las comparaciones

# Función para ejecutar el script de python
run_finetuning() {
    local base_name=$1
    local weights=$2
    local arch_flags=$3
    
    for bs in "${BATCH_SIZES[@]}"; do
        for lr in "${LRS[@]}"; do
            for wd in "${WDS[@]}"; do
                
                exp_name="ft_${base_name}_bs${bs}_lr${lr}_wd${wd}"
                
                # Evitar re-ejecución si se cancela el job
                if grep -q "^$exp_name," "$CSV_FILE" 2>/dev/null; then
                    echo "Saltando $exp_name (Ya existe)"
                    continue
                fi

                echo "--------------------------------------------------------"
                echo "Afinamiento: $exp_name | BS: $bs | LR: $lr | WD: $wd"
                
                python -u "$BASE_SCRIPT" \
                    --h5_path "$H5_PATH" \
                    --train_idx "$TRAIN_IDX" \
                    --val_idx "$VAL_IDX" \
                    --pretrained_weights "$weights" \
                    --num_frames 50 \
                    --num_dct 256 \
                    $arch_flags \
                    --batch_size "$bs" \
                    --lr "$lr" \
                    --weight_decay "$wd" \
                    --seed "$SEED" \
                    --epochs 30 \
                    --patience 8 \
                    --exp_name "$exp_name" \
                    --results_csv "$CSV_FILE" \
                    --save_model "$OUTPUT_DIR/${exp_name}.pth"
            done
        done
    done
}

echo "=== INICIANDO CAMPEÓN 1: FULL MODEL ==="
run_finetuning "full_model" "$MODEL_1_WEIGHTS" "--spatial --spectral --metrics"

echo "=== INICIANDO CAMPEÓN 2: SPATIAL SPECTRAL ==="
run_finetuning "spatial_spectral" "$MODEL_2_WEIGHTS" "--spatial --spectral --no_metrics"

echo "========================================="
echo "FINE-TUNING FINALIZADO CON ÉXITO"
echo "========================================="
