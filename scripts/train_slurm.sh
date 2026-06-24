#!/bin/bash
#SBATCH -J df_train                  # Nombre del trabajo
#SBATCH -p gpu                       # IMPORTANTE: Solicitamos la partición con Tarjetas Gráficas
#SBATCH --gres=gpu:1                 # Solicitamos exactamente 1 GPU
#SBATCH -n 1
#SBATCH -c 8                         # 8 CPUs para alimentar los datos a la GPU sin cuello de botella
#SBATCH --mem=100G                    # Memoria RAM general
#SBATCH --time=48:00:00              # Le damos un máximo de 48 horas de paciencia
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/train_ia_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/train_ia_%j.err

echo "========================================="
echo "INICIANDO ENTRENAMIENTO DE IA (NODO GPU)"
echo "Fecha de inicio: $(date)"
echo "Nodo asignado: $(hostname)"
echo "========================================="

# 1. Cargar el entorno
module load python/3.11
source ~/deepfake_project/env_df/bin/activate

# 2. Ejecutar Entrenamiento
python ~/deepfake_project/scripts/train.py

echo "========================================="
echo "ENTRENAMIENTO FINALIZADO"
echo "Fecha de fin: $(date)"
echo "========================================="
