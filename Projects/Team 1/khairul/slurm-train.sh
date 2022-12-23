#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=train.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=cheetah02
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load anaconda3

conda deactivate
conda activate crypten

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/crypten/lib
python mnist_train.py