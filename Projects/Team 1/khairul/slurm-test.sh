#!/usr/bin/env bash
#SBATCH --job-name="test"
#SBATCH --output=outputs/lynx05-test.out
#---SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=lynx05
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load anaconda3

conda deactivate
conda activate crypten

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/crypten/lib
python mnist_test.py