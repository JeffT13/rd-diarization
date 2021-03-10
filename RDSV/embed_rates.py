import os, json
import numpy as np

from VoiceEncoder.util import casewrttm_to_dvec
from param import *


with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
print('Beginning Rate Tuning')
for r in tune_rate:
    print('Processing for rate=', r)
    path_out = inf_lab_path+'r'+str(r)+'/'
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    for wav in set_dict['r']+set_dict['d']:
        case = wav.split('.')[0]
        embed, splits, info, mask = casewrttm_to_dvec(audio_path+wav, rttm_path+case+'.rttm', sd_path, device=device, verbose=verbose, rate=r)
        np.save(path_out+'{}_embeds.npy'.format(case),embed[0])
        np.save(path_out+'{}_embeds_labels.npy'.format(case),embed[1])
        np.save(path_out+'{}_embeds_times.npy'.format(case),embed[2])

if verbose:
    print('Rate Tuning complete and dict saved')