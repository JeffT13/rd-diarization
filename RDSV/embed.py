import os, json
import numpy as np

from VoiceEncoder.util import casewrttm_to_dvec, case_to_dvec
from param import *

# process annotated cases
print('C & D Set Encoding (w/ labels)')
for wav in c_set+d_set:
    case = wav.split('.')[0]
    print('Encoding Case:', case)
    embed, splits, info, mask = casewrttm_to_dvec(audio_path+wav, rttm_path+case+'.rttm', sd_path, device=device, verbose=verbose)
    np.save(inf_lab_path+'{}_embeds.npy'.format(case),embed[0])
    np.save(inf_lab_path+'{}_embeds_labels.npy'.format(case),embed[1])
    np.save(inf_lab_path+'{}_embeds_times.npy'.format(case),embed[2])
      
# process unannotated cases
print('T Set Encoding (no labels)')
for wav in t_set:
    case = wav.split('.')[0]
    print('Encoding Case:', case)
    embed, info = case_to_dvec(audio_path+wav, device=device, verbose=verbose)
    np.save(inf_path+'{}_embeds.npy'.format(case),embed)
    np.save(inf_path+'{}_embeds_times.npy'.format(case),info[0])
