#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=k80_crypten.out
#SBATCH -A ds7003-fall22
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1

module load singularity pytorch/1.10.0
singularity run --nv $CONTAINERDIR/pytorch-1.10.0.sif train_crypten.py