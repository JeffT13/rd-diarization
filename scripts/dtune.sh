#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=dtune
#SBATCH --output=diar_tune.out

DIR="/scratch/jt2565/data/diarization/" #di_path
echo "$DIR"
if [-d "$DIR" ]; then 
	rm -rf "$DIR"
fi

mkdir "$DIR"
python ./RDSV/tune_rdsv.py