#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=A
#SBATCH --output=save.out

python ./RDSV/analyze_tune.py