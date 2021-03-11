#!/bin/bash

module load rclone/1.53.3


rclone copy /scratch/jt2565/data/diarization/17-571_ral.rttm dgr:greene/info
rclone copy /scratch/jt2565/data/diarization/17-571_rdsv.rttm dgr:greene/info
rclone copy /scratch/jt2565/data/rttm/17-571.rttm dgr:greene/info


rclone copy /scratch/jt2565/data/out/ral_spkr_dict.json dgr:greene/info
rclone copy /scratch/jt2565/data/out/set_dict.json dgr:greene/info
rclone copy /scratch/jt2565/data/out/lr_dict.json dgr:greene/info
rclone copy /scratch/jt2565/data/out/test_eval.csv dgr:greene/info




rclone copy /scratch/jt2565/data/out/r1/X.npy dgr:greene/r1
rclone copy /scratch/jt2565/data/out/r1/Xd.npy dgr:greene/r1
rclone copy /scratch/jt2565/data/out/r1/Y.npy dgr:greene/r1
rclone copy /scratch/jt2565/data/out/r1/Yd.npy dgr:greene/r1


rclone copy /scratch/jt2565/data/out/r5/X.npy dgr:greene/r5
rclone copy /scratch/jt2565/data/out/r5/Xd.npy dgr:greene/r5
rclone copy /scratch/jt2565/data/out/r5/Y.npy dgr:greene/r5
rclone copy /scratch/jt2565/data/out/r5/Yd.npy dgr:greene/r5