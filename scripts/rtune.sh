#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=rtune
#SBATCH --output=rates.out

mkdir /scratch/jt2565/SCOTUS/inf_labelled/
python ./RDSV/embed_rates.py