import os, json
import numpy as np

from VoiceEncoder.util import casewrttm_to_dvec, case_to_dvec
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
# process annotated cases
print('C & D Set Encoding (w/ labels)')
for wav in set_dict['c']+set_dict['d']:
    case = wav.split('.')[0]
    print('Encoding Case:', case)
    embed, splits, info, mask = casewrttm_to_dvec(audio_path+wav, rttm_path+case+'.rttm', sd_path, device=device, verbose=verbose, rate=encoder_rate)
    np.save(inf_lab_path+'{}_embeds.npy'.format(case),embed[0])
    np.save(inf_lab_path+'{}_embeds_labels.npy'.format(case),embed[1])
    np.save(inf_lab_path+'{}_embeds_times.npy'.format(case),embed[2])
      
# process unannotated cases
print('T Set Encoding (no labels)')
for wav in set_dict['t']:
    case = wav.split('.')[0]
    print('Encoding Case:', case)
    embed, info = case_to_dvec(audio_path+wav, device=device, verbose=verbose, rate=encoder_rate)
    np.save(inf_path+'{}_embeds.npy'.format(case),embed)
    np.save(inf_path+'{}_embeds_times.npy'.format(case),info[0])
