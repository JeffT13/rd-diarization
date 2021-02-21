
import os, json
import numpy as np
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)

eval_dict = {}
metric = DiarizationErrorRate(collar=der_collar, skip_overlap=True)

for wav in set_dict['d']:
    case = wav.split('.')[0]
    print('evaluating performance on case', case)
    predict = case+'_rdsv.rttm'
    ral_label = case+'_ral.rttm'
    predictions = load_rttm(di_path+predict)[case]
    groundtruths = load_rttm(di_path+ral_label)[case]
    print(case, ' === ', metric(groundtruths, predictions, detailed=True))
    eval_dict[case] = metric(groundtruths, predictions, detailed=True)['diarization error rate']
    print()
    
print('t set')
for wav in set_dict['t']:
    case = wav.split('.')[0]
    print('evaluating performance on case', case)
    predict = case+'_rdsv.rttm'
    ral_label = case+'_ral.rttm'
    predictions = load_rttm(di_path+predict)[case]
    groundtruths = load_rttm(di_path+ral_label)[case]
    print(case, ' === ', metric(groundtruths, predictions, detailed=True))
    eval_dict[case] = metric(groundtruths, predictions, detailed=True)['diarization error rate']
    print()
    
    
with open(eval_path,'w') as out:
    json.dump(eval_dict, out)
    print('Eval Dict Saved')
    
print("Avg DER:", np.mean([i for i in eval_dict.values()])