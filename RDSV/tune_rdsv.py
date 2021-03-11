import os, json
import numpy as np

from VoiceEncoder.util import casewrttm_to_dvec
from rdsv import RefAudioLibrary, Diarize, diar_to_rttm, rttmto_RALrttm
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
eval_dict = {}
cases = [item.split('.')[0] for item in set_dict['r']] #RAL needs case name only 
metric = DiarizationErrorRate(collar=der_collar, skip_overlap=True)

path_out = inf_lab_path+'r'+str(encoder_rate)+'/'
di_path_out = di_path+'r'+str(encoder_rate)+'/'
if not os.path.exists(di_path_out):
    os.mkdir(di_path_out)


# Diarization Tuning
for a in tune_mal:
    scotus_ral = RefAudioLibrary(cases, path_out, rttm_path, sd_path, min_audio_len=a)
    eval_dict[str(a)] = {}
    for m in tune_ms:
        for d in tune_dt:
            for s in tune_sct:
                # diarize development cases (loaded in)
                der = []
                for wav in set_dict['d']:
                    case = wav.split('.')[0]
                    embed = np.load(path_out+'{}_embeds.npy'.format(case))
                    info = np.load(path_out+'{}_embeds_times.npy'.format(case))
                    timelst = Diarize(scotus_ral, embed, info, sim_thresh=d, score_thresh=s, min_seg = m)
                    diar_to_rttm(timelst, case, di_path_out)
                    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path_out)
                    predict = case+'_rdsv.rttm'
                    ral_label = case+'_ral.rttm'
                    predictions = load_rttm(di_path_out+predict)[case]
                    groundtruths = load_rttm(di_path_out+ral_label)[case]
                    der.append(metric(groundtruths, predictions, detailed=True)['diarization error rate'])
                #Hyperparam tuning results
                eval_dict[str(a)][str(m)+str(d)+str(s)] = der

with open(tune_eval_path,'w') as out:
    json.dump(eval_dict, out)
    print('Diarization Tuning complete and dict saved')