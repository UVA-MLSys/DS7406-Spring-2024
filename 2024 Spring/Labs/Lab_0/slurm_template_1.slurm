#!/bin/bash -l
# --- this job will be run on any available node
#SBATCH --job-name="TF1_2+4_Smoothed_3142"
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00
#SBATCH --mem=120GB
#SBATCH --account=ds6011-sp22-002
#SBATCH --export=NONE

# Load CUDA and CUDNN for training deep learning models
module load cuda cudnn anaconda

# Activate Conda
conda deactivate
conda activate ~/anaconda3/envs/ml

# Automation Code
python tft1_v1_full.py --mode 0 --features 0 4 9 10 12 14 --runname "TF1_2+4_Smoothed_3142" --county_size 3142
