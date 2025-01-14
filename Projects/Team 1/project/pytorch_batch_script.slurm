#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=p100.out
#SBATCH -A ds7003-fall22
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1

module load singularity pytorch/1.10.0
singularity run --nv $CONTAINERDIR/pytorch-1.10.0.sif train_mnist.py