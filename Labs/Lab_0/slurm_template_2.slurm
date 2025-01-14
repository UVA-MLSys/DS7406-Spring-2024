#!/bin/bash
# This slurm script file runs


#SBATCH -p standard
#SBATCH --mem=40000
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=1536137580@qq.com
#SBATCH --ntasks=1
#SBATCH -e mapping01.err
#SBATCH -o mapping01.out



docn="Path to your file"

for k in $(cat $docn)
do
  # Your automation code code goes here...
done
