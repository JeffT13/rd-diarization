import os, json, csv
import numpy as np
from scipy import stats

from VoiceEncoder.util import case_to_dvec
from rdsv import RefAudioLibrary, Diarize, diar_to_rttm, rttmto_RALrttm
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
# Build RAL (assumes C set was processed)
cases = [item.split('.')[0] for item in set_dict['r']] #RAL needs case name only 
scotus_ral = RefAudioLibrary(cases, inf_lab_path+'r'+str(encoder_rate)+'/', rttm_path, sd_path, min_audio_len=mal)
metric = DiarizationErrorRate(collar=der_collar, skip_overlap=True)

print('T Set Encoding (no labels)')
der = []
size = []
for wav in set_dict['t']:
    case = wav.split('.')[0]
    print('Encoding Case:', case)
    embed, info, sz = case_to_dvec(audio_path+wav, device=device, verbose=verbose, rate=encoder_rate)
    if save_test_emb:
        np.save(inf_path+'{}_embeds.npy'.format(case),embed)
        np.save(inf_path+'{}_embeds_times.npy'.format(case),info[0])
    timelst = Diarize(scotus_ral, embed, info[0], thresh=diar_thresh)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    predict = case+'_rdsv.rttm'
    ral_label = case+'_ral.rttm'
    predictions = load_rttm(di_path+predict)[case]
    groundtruths = load_rttm(di_path+ral_label)[case]
    der.append(metric(groundtruths, predictions, detailed=True)['diarization error rate'])
    size.append(sz)


bycase = list(zip([item.split('.')[0] for item in set_dict['t']], der, size))
desc = stats.describe(der)
settings = ['Param:', encoder_rate, '|', mal, mrt, ' - ', ms, diar_thresh]
with open(test_eval_path, 'w') as f:
    write = csv.writer(f)
    write.writerow(settings)
    write.writerows(bycase)
    write.writerow(desc)

print(settings)
print('Final results:', desc)