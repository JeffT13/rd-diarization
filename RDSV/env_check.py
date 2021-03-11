import os, csv, sys, json
import torch
import webrtcvad
import librosa
import struct
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_dilation
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
        s = json.load(f)
    print('R set:', s['r'])
    print('D set:', s['d'])

if os.path.exists(tune_eval_path):
    hold = 100
    with open(tune_eval_path) as jt: 
        tune = json.load(jt)
    for key in tune:
        for k in tune[key]:
            temp = stats.describe(tune[key][k])
            if temp[2]<.2:
                #Mean, SD, Max
                if temp[2]<hold:
                    hold=temp[2]
                    id = key+'|'+k
                    perm = temp

    print('Best Param:', id, ' === ', perm)
        