#!/bin/bash
#SBATCH -J data_split                # Nombre del trabajo
#SBATCH -p cpu-dev                   # Partición de desarrollo (ideal para pruebas cortas)
#SBATCH -n 1                         # Número de tareas
#SBATCH -c 4                         # CPUs por tarea
#SBATCH --mem=8G                     # Memoria RAM
#SBATCH --time=00:30:00              # Tiempo máximo
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/test_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/test_%j.err

echo "========================================="
echo "Iniciando prueba de arquitectura en SLURM"
echo "Fecha: $(date)"
echo "Nodo asignado: $(hostname)"
echo "========================================="

# 1. Cargar el entorno
module load python/3.11
source ~/deepfake_project/env_df/bin/activate

# 2. Ejecutar el script de prueba
python ~/deepfake_project/scripts/split_data.py

echo "========================================="
echo "Prueba finalizada"
echo "Fecha: $(date)"
echo "========================================="
