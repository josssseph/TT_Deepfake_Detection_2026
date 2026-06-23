#!/bin/bash
#SBATCH -J tuning
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/tuning_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/tuning_%j.err

echo "========================================="
echo "INICIANDO EXPERIMENTOS DE TUNING"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "========================================="

# 1. Cargar entorno
module load python/3.11
source ~/deepfake_project/env_df/bin/activate

# ============================================================
# 2. COPIAR HDF5 AL DISCO LOCAL DEL NODO (evita atasco de red)
# ============================================================
LOCAL_DIR="/tmp/joseph_deepfake_$SLURM_JOB_ID"
mkdir -p "$LOCAL_DIR"

echo "Copiando dataset (111 GB) a disco local del nodo ($LOCAL_DIR)..."
cp /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/data/processed/ff_dataset_max60frames_4096dct.h5 "$LOCAL_DIR/"
echo "Copia finalizada. Entrenando desde disco local."

H5_PATH="$LOCAL_DIR/ff_dataset_max60frames_4096dct.h5"
TRAIN_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_indices.npy"
VAL_IDX="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/val_indices.npy"
CSV_FILE="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/resultados_tuning.csv"
MODEL_DIR="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/models"
mkdir -p "$MODEL_DIR"

# ============================================================
# 3. PARÁMETROS BASE (sin workers, batch pequeño)
# ============================================================
BASE_SCRIPT="/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts/train_tuning.py"
BASE_LR=1e-4
BASE_BATCH=4
BASE_WORKERS=0
BASE_EPOCHS=30
BASE_PATIENCE=10
BASE_LSTM_HIDDEN=256
BASE_LSTM_LAYERS=1
BASE_SPECTRAL_HIDDEN=128

# ============================================================
# FASE 1: ABLACIÓN DE RAMAS
# ============================================================
echo "========================================="
echo "FASE 1: ABLACIÓN DE RAMAS"
echo "========================================="

for arch in spatial_only spectral_only structural_only spatial_spectral spatial_structural spectral_structural full_model; do
    echo "----- Entrenando $arch -----"
    case $arch in
        spatial_only)
            SPATIAL="--spatial"; SPECTRAL="--no_spectral"; METRICS="--no_metrics" ;;
        spectral_only)
            SPATIAL="--no_spatial"; SPECTRAL="--spectral"; METRICS="--no_metrics" ;;
        structural_only)
            SPATIAL="--no_spatial"; SPECTRAL="--no_spectral"; METRICS="--metrics" ;;
        spatial_spectral)
            SPATIAL="--spatial"; SPECTRAL="--spectral"; METRICS="--no_metrics" ;;
        spatial_structural)
            SPATIAL="--spatial"; SPECTRAL="--no_spectral"; METRICS="--metrics" ;;
        spectral_structural)
            SPATIAL="--no_spatial"; SPECTRAL="--spectral"; METRICS="--metrics" ;;
        full_model)
            SPATIAL="--spatial"; SPECTRAL="--spectral"; METRICS="--metrics" ;;
    esac

    python "$BASE_SCRIPT" \
        --h5_path "$H5_PATH" \
        --train_idx "$TRAIN_IDX" \
        --val_idx "$VAL_IDX" \
        --results_csv "$CSV_FILE" \
        --exp_name "fase1_$arch" \
        --save_model "$MODEL_DIR/fase1_${arch}.pth" \
        --num_frames 30 --num_dct 1024 \
        --lr $BASE_LR --batch_size $BASE_BATCH --num_workers $BASE_WORKERS \
        --epochs $BASE_EPOCHS --patience $BASE_PATIENCE \
        --lstm_hidden $BASE_LSTM_HIDDEN --lstm_layers $BASE_LSTM_LAYERS \
        --spectral_hidden_dim $BASE_SPECTRAL_HIDDEN \
        $SPATIAL $SPECTRAL $METRICS
done

# ============================================================
# FASE 2: TUNING DE FRAMES
# ============================================================
echo "========================================="
echo "FASE 2: TUNING DE FRAMES"
echo "========================================="

for frames in 10 20 30 40 50 60; do
    echo "----- num_frames = $frames -----"
    python "$BASE_SCRIPT" \
        --h5_path "$H5_PATH" \
        --train_idx "$TRAIN_IDX" --val_idx "$VAL_IDX" \
        --results_csv "$CSV_FILE" \
        --exp_name "fase2_frames_${frames}" \
        --save_model "$MODEL_DIR/fase2_frames_${frames}.pth" \
        --num_frames $frames --num_dct 1024 \
        --lr $BASE_LR --batch_size $BASE_BATCH --num_workers $BASE_WORKERS \
        --epochs $BASE_EPOCHS --patience $BASE_PATIENCE \
        --lstm_hidden $BASE_LSTM_HIDDEN --lstm_layers $BASE_LSTM_LAYERS \
        --spectral_hidden_dim $BASE_SPECTRAL_HIDDEN \
        --spatial --spectral --metrics
done

# ============================================================
# FASE 3: TUNING DE COEFICIENTES DCT
# ============================================================
echo "========================================="
echo "FASE 3: TUNING DE COEFICIENTES DCT"
echo "========================================="

for dct in 256 512 1024 2048 4096; do
    echo "----- num_dct = $dct -----"
    python "$BASE_SCRIPT" \
        --h5_path "$H5_PATH" \
        --train_idx "$TRAIN_IDX" --val_idx "$VAL_IDX" \
        --results_csv "$CSV_FILE" \
        --exp_name "fase3_dct_${dct}" \
        --save_model "$MODEL_DIR/fase3_dct_${dct}.pth" \
        --num_frames 30 --num_dct $dct \
        --lr $BASE_LR --batch_size $BASE_BATCH --num_workers $BASE_WORKERS \
        --epochs $BASE_EPOCHS --patience $BASE_PATIENCE \
        --lstm_hidden $BASE_LSTM_HIDDEN --lstm_layers $BASE_LSTM_LAYERS \
        --spectral_hidden_dim $BASE_SPECTRAL_HIDDEN \
        --spatial --spectral --metrics
done

# Limpiar disco local
rm -rf "$LOCAL_DIR"
echo "========================================="
echo "TODOS LOS EXPERIMENTOS FINALIZADOS"
echo "Fecha: $(date)"
echo "========================================="
