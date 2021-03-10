#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=rtune
#SBATCH --output=rates.out


DIR="/scratch/jt2565/SCOTUS/inf_labelled/" 
echo "$DIR"
if [-d "$DIR" ]; then 
	rm -rf "$DIR"
fi

mkdir "$DIR"
python ./RDSV/embed_rates.py