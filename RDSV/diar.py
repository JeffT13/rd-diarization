
import os, json
import numpy as np
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from VoiceEncoder.util import diar_to_rttm, rttmto_RALrttm
from utils import RefAudioLibrary, Diarize
from param import *

# Build RAL
scotus_ral = RefAudioLibrary(c_set, inf_lab_path, rttm_path, sd_path)

for case in d_set:
    case = wav.split('.')[0]
    print('Diarizing Case:', case)
    embed = np.load(inf_lab_path+case+'_embeds.npy')
    time = np.load(inf_lab_path+case+'_embeds_times.npy')
    timelst = Diarize(scotus_ral, embed, time)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    

for wav in t_set:
    case = wav.split('.')[0]
    print('Diarizing Case:', case)
    embed = np.load(inf_path+case+'_embeds.npy')
    time = np.load(inf_path+case+'_embeds_times.npy')
    timelst = diarize(scotus_ral, embed, time)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    print('Case', case, 'RALrttm saved')