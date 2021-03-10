import os, json, timeit
import numpy as np

from VoiceEncoder.util import casewrttm_to_dvec
from rdsv import RefAudioLibrary, Diarize, diar_to_rttm, rttmto_RALrttm
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from param import *


with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
embed_dict = dict()

print('Beginning Rate Tuning')
for r in tune_rate:
    print('Processing for rate=', r)
    path_out = inf_lab_path+'r'+str(r)+'/'
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    embed_dict[r] = []
    for wav in set_dict['r']+set_dict['d']:
        case = wav.split('.')[0]
        start = timeit.default_timer()
        embed, splits, info, mask = casewrttm_to_dvec(audio_path+wav, rttm_path+case+'.rttm', sd_path, device=device, verbose=verbose, rate=r)
        end = timeit.default_timer() - start
        embed_dict[r].append([case,end, len(embed[0])])
        np.save(path_out+'{}_embeds.npy'.format(case),embed[0])
        np.save(path_out+'{}_embeds_labels.npy'.format(case),embed[1])
        np.save(path_out+'{}_embeds_times.npy'.format(case),embed[2])

with open(embed_path,'w') as out:
    json.dump(embed_dict, out)
    if verbose:
        print('Rate Tuning complete and dict saved')