#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=LR
#SBATCH --output=lr.out

python ./RDSV/run_LR.py