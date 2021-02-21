
import os, json
import numpy as np
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from VoiceEncoder.util import diar_to_rttm, rttmto_RALrttm
from rdsv import RefAudioLibrary, Diarize
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
# Build RAL
scotus_ral = RefAudioLibrary(set_dict['c'], inf_lab_path, rttm_path, sd_path)

for wav in set_dict['d']:
    case = wav.split('.')[0]
    print('Diarizing Case:', case)
    embed = np.load(inf_lab_path+case+'_embeds.npy')
    time = np.load(inf_lab_path+case+'_embeds_times.npy')
    timelst = Diarize(scotus_ral, embed, time, thresh=diar_thresh)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    

for wav in set_dict['t']:
    case = wav.split('.')[0]
    print('Diarizing Case:', case)
    embed = np.load(inf_path+case+'_embeds.npy')
    time = np.load(inf_path+case+'_embeds_times.npy')
    timelst = diarize(scotus_ral, embed, time, thresh=diar_thresh)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    print('Case', case, 'RALrttm saved')