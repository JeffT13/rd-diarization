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
 
print('C Set Encoding (w/ labels)')
for wav in set_dict['c']:
    case = wav.split('.')[0]
    embed, splits, info, mask = casewrttm_to_dvec(audio_path+wav, rttm_path+case+'.rttm', sd_path, device=device, verbose=verbose, rate=encoder_rate)
    np.save(inf_lab_path+'{}_embeds.npy'.format(case),embed[0])
    np.save(inf_lab_path+'{}_embeds_labels.npy'.format(case),embed[1])
    np.save(inf_lab_path+'{}_embeds_times.npy'.format(case),embed[2])
print('C Set Encoding (w/ labels) saved')

 
# Build RAL
temp = [item.split('.')[0] for item in set_dict['c']] #RAL needs case name only 
scotus_ral = RefAudioLibrary(temp, inf_lab_path, rttm_path, sd_path)
metric = DiarizationErrorRate(collar=der_collar, skip_overlap=True)
 
# process development cases (no saving)
for wav in set_dict['d']:
    case = wav.split('.')[0]
    print('Encoding Case:', case)
    embed, splits, info, mask = casewrttm_to_dvec(audio_path+wav, rttm_path+case+'.rttm', sd_path, device=device, verbose=verbose, rate=encoder_rate)
    '''
    np.save(inf_lab_path+'{}_embeds.npy'.format(case),embed[0])
    np.save(inf_lab_path+'{}_embeds_labels.npy'.format(case),embed[1])
    np.save(inf_lab_path+'{}_embeds_times.npy'.format(case),embed[2])
    print('Diarizing Case:', case)
    embed = np.load(inf_lab_path+case+'_embeds.npy')
    time = np.load(inf_lab_path+case+'_embeds_times.npy')
    '''
    timelst = Diarize(scotus_ral, embed[0], embed[2], thresh=diar_thresh)
    diar_to_rttm(timelst, case, di_path)
    rttmto_RALrttm(case, scotus_ral, rttm_path, di_path)
    predict = case+'_rdsv.rttm'
    ral_label = case+'_ral.rttm'
    predictions = load_rttm(di_path+predict)[case]
    groundtruths = load_rttm(di_path+ral_label)[case]
    print(case, ' === ', metric(groundtruths, predictions, detailed=True))
    eval_dict[case] = metric(groundtruths, predictions, detailed=True)['diarization error rate']
    print('Case', case, 'DER:', eval_dict[case])
    print()
       
    
print("Avg DER on D set:", round(np.mean([i for i in eval_dict.values()]),3), "for (E,D):", encoder_rate, diar_thresh)