#!/bin/bash
#SBATCH -J count_params
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/count_params_%j.out
#SBATCH -e /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/logs/count_params_%j.err

set -euo pipefail

module load python/3.11
source ~/deepfake_project/env_df/bin/activate

cd /home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project/scripts

python -u count_trainable_params.py \
  --num_dct 256 \
  --spatial \
  --spectral \
  --no_metrics
