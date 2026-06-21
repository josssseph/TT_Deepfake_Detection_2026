#!/bin/bash
#SBATCH -J df_preproc_cpu
#SBATCH -p cpu                  # Partición solo CPU (hasta 64 cores / 128 GB)
#SBATCH -n 1
#SBATCH -c 16                   # Usa 16 CPUs para acelerar la detección de rostros
#SBATCH --mem=64G               # Memoria suficiente para procesar videos en paralelo
#SBATCH --time=08:00:00         # Tiempo holgado
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/logs_preproc_cpu_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/logs_preproc_cpu_%j.err

echo "==============================="
echo "Inicio del preprocesamiento (SOLO CPU)"
echo "Fecha inicio: $(date)"
echo "Nodo: $(hostname)"
echo "==============================="

# Cargar solo el módulo de Python (sin CUDA ni OpenCV del sistema)
module load python/3.11

# Activar el entorno virtual donde ya tienes opencv-python-headless
source ~/deepfake_project/env_df/bin/activate

# Ejecutar el script de procesamiento (versión CPU)
python ~/deepfake_project/scripts/procesamiento_masivo.py

echo "==============================="
echo "Fin del preprocesamiento"
echo "Fecha fin: $(date)"
echo "==============================="
