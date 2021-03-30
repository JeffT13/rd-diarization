# Reference Dependent Speaker Verification

This folder contains our implementation of **Reference-Dependent** Speaker Verification, as outlined in [our paper](). For a breakdown of the contents of this repo, we suggest sections 4 and 5 of the paper. This README assumes you have followed the procedure outlined in the `SCOTUS` folder for pulling the SCOTUS oral argument audio and transcripts, saving the transcripts in RTTM and building our speaker label dictionary. The RDSV process can be broken down into 5 steps:

    - Allocate cases to R, D, T set
    - Embed R & D set for each rate (saved seperately)
    - Run Logistic Regression Experiment 
    - Analyze LR performance to set Voice Encoder `rate`
    - Tune RDSV pipeline on D set
    - Analyze tuning performance to set RDSV `sim_thresh` and `score_thresh`
    - Diarize Heldout test set

## Experiment Reproduction

This process was ran as SLURM jobs, so the `rd-diarization/scripts` folder holds shell files which create the directory each process saves it's results down in and runs the respective python script. Filepaths are managed in the `param.py`, but these scripts have hardcoded filepaths that should be changed/removed if reran.

The order of scripts (with name of associated shell script) ran are:

    - `build_sets.py` (no shell script)
    - `embed_rates.py` (rtune.sh)
    - `run_LR.py` (lr_exp.sh)
    - `env_check.py`
    - `tune_rdsv.py` (dtune.sh)
    - `analyze_tune.py` (no shell script)
    - `final_test.py` (test)
    
In between any steps (or once you have ran them all) you can run the `env_check.py` script and it will print out some of the most relevant information up to the step you have completed. This was used to view the logistic regression experiments results, and could be used in place of the `analyze_tune.py` step as well for a more concise output.

## Visualizations

The `plots` folder contains an `rclone.sh` script used to save the outputs at each stage from the HPC cluster to a local machine, and visualizations are made in the `Visuals.ipynb` notebook locally. Not seen in the repo are 3 folders `info`, which is the home of the experiment outputs (dictionaries/csvs of performance) and two folders `r1` and `r5` which each house the R and D set embeddings for rate 1 and 5. 