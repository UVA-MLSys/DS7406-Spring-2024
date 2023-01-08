#!/usr/bin/env bash
#SBATCH --job-name="test"
#SBATCH --output=a100.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=ds6011-sp22-002
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn singularity
singularity exec crypten_latest.sif python mnist_test.py

# NOTE: this is for UVA Rivanna server