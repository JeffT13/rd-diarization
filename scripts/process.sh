#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=D
#SBATCH --output=result.out

python ./RDSV/diar.py
python ./RDSV/eval.py