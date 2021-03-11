import os, csv, sys, json
import torch
import webrtcvad
import librosa
import struct
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Union, List
from time import perf_counter as timer

from VoiceEncoder.hparams import *
from param import *

print(mel_window_length, audio_norm_target_dBFS)
print('---')
print(audio_path)
print('----')
print('Check passed')

if os.path.exists(sd_path):
    print('Spkr Dict Saved')
    
if os.path.exists(set_path):
    with open(set_path) as f:
        set_dict = json.load(f)
    print('R set:', set_dict['r'])
    print('D set:', set_dict['d'])

if os.path.exists(lr_path):
    with open(lr_path) as f:
        lr_dict = json.load(f)
    print('LR Results:', lr_dict[str(encoder_rate)])
    
    
if os.path.exists(tune_eval_path):
    hold = 100
    with open(tune_eval_path) as jt: 
        tune = json.load(jt)
    for key in tune.keys():
        print('diar settings:', key)
        temp = stats.describe(tune[key])
        if temp[2]<avgder_thresh:
            #Mean, SD, Max
            print(round(temp[2],3), round(np.sqrt(temp[3]),3), round(temp[1][1], 3))
            if temp[2]<hold:
                hold=temp[2]
                id = key
                perm = temp
    print('Best Param:', id, ' === ', perm)
    
    cases = [item.split('.')[0] for item in set_dict['r']] 
    scotus_ral = RefAudioLibrary(cases, inf_lab_path+'r'+str(encoder_rate)+'/', rttm_path, sd_path, min_audio_len=mal)
    print('\n', scotus_ral.RAL.keys())