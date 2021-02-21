
import os, json
import numpy as np
from rdsv import RefAudioLibrary, Diarize, diar_to_rttm, rttmto_RALrttm
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
# Build RAL
temp = [item.split('.')[0] for item in set_dict['c']]
scotus_ral = RefAudioLibrary(temp, inf_lab_path, rttm_path, sd_path)

for wav in set_dict['d']:
    case = wav.split('.')[0]
    print('Diarizing Case:', case)
    embed = np.load(inf_lab_path+case+'_embeds.npy')
    time = np.load(inf_lab_path+case+'_embeds_times.npy')
    timelst = Diarize(scotus_ral, embed, time, thresh=diar_thresh)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    

print('t set')
for wav in set_dict['t']:
    case = wav.split('.')[0]
    print('Diarizing Case:', case)
    embed = np.load(inf_path+case+'_embeds.npy')
    time = np.load(inf_path+case+'_embeds_times.npy')
    timelst = Diarize(scotus_ral, embed, time, thresh=diar_thresh)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    print('Case', case, 'RALrttm saved')