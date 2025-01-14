#!/usr/bin/env bash
#SBATCH --job-name="test"
#SBATCH --output=output.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx05
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load anaconda3

conda deactivate
conda activate crypten

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/crypten/lib
python mnist_test.py

# NOTE: this is for UVA CS server