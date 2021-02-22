#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=Drun
#SBATCH --output=d_set1.out

DIR="/scratch/jt2565/SCOTUS/inf_labelled/"
if [-d "$DIR" ]; then 
	rm -rf /scratch/jt2565/SCOTUS/inf_labelled/

mkdir /scratch/jt2565/SCOTUS/inf_labelled/
python ./RDSV/d_set.py