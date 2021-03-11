#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=test
#SBATCH --output=test.out


DIR="/scratch/jt2565/data/inference/" #inf_path
echo "$DIR"
if [-d "$DIR" ]; then 
	rm -rf "$DIR"
fi

mkdir "$DIR"
python ./RDSV/final_test.py