#!/bin/bash

module load rclone/1.53.3


rclone copy /scratch/jt2565/data/diarization/17-571_ral.rttm gdr:greene/info
rclone copy /scratch/jt2565/data/diarization/17-571_rdsv.rttm gdr:greene/info
rclone copy /scratch/jt2565/data/rttm/17-571.rttm gdr:greene/info


rclone copy /scratch/jt2565/data/out/ral_spkr_dict.json gdr:greene/info
rclone copy /scratch/jt2565/data/out/set_dict.json gdr:greene/info
rclone copy /scratch/jt2565/data/out/lr_dict.json gdr:greene/info
rclone copy /scratch/jt2565/data/out/test_eval.csv gdr:greene/info




rclone copy /scratch/jt2565/data/out/r1/X.npy gdr:greene/r1
rclone copy /scratch/jt2565/data/out/r1/Xd.npy gdr:greene/r1
rclone copy /scratch/jt2565/data/out/r1/Y.npy gdr:greene/r1
rclone copy /scratch/jt2565/data/out/r1/Yd.npy gdr:greene/r1


rclone copy /scratch/jt2565/data/out/r5/X.npy gdr:greene/r5
rclone copy /scratch/jt2565/data/out/r5/Xd.npy gdr:greene/r5
rclone copy /scratch/jt2565/data/out/r5/Y.npy gdr:greene/r5
rclone copy /scratch/jt2565/data/out/r5/Yd.npy gdr:greene/r5