import os, json, glob, sh
import numpy as np
from scipy import stats
from VoiceEncoder.util import casewrttm_to_dvec
from rdsv import RefAudioLibrary, Diarize, diar_to_rttm, rttmto_RALrttm
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)
    
rate_dict = {}
per_rate_best = []
cases = [item.split('.')[0] for item in set_dict['c']] #RAL needs case name only 
metric = DiarizationErrorRate(collar=der_collar, skip_overlap=True)

for r in tune_rate:
    eval_dict = {}
    temp=100
    
    path_out = inf_lab_path+'r'+str(r)+'/'
    di_path_out = di_path+'r'+str(r)+'/'
    if os.path.exists(di_path_out):
        files = glob.glob(di_path_out+'*')
        if len(files)>0:
            sh.rm(files)
    else:
        os.mkdir(di_path_out)
    
    
    # Diarization Tuning
    for a in tune_mal:
        for t in tune_mrt:
            scotus_ral = RefAudioLibrary(cases, path_out, rttm_path, sd_path, min_audio_len=a, min_ref_thresh=t)
            for m in tune_ms:
                for d in tune_dt:
                    # diarize development cases (loaded in)
                    der = []
                    for wav in set_dict['d']:
                        case = wav.split('.')[0]
                        embed = np.load(path_out+'{}_embeds.npy'.format(case))
                        info = np.load(path_out+'{}_embeds_times.npy'.format(case))
                        timelst = Diarize(scotus_ral, embed, info, thresh=d, min_seg = m)
                        diar_to_rttm(timelst, case, di_path_out)
                        rttmto_RALrttm(case, scotus_ral, rttm_path, di_path_out)
                        predict = case+'_rdsv.rttm'
                        ral_label = case+'_ral.rttm'
                        predictions = load_rttm(di_path_out+predict)[case]
                        groundtruths = load_rttm(di_path_out+ral_label)[case]
                        der.append(metric(groundtruths, predictions, detailed=True)['diarization error rate'])
                        
                    #Hyperparam tuning results
                    ky = str(a)+str(t)+'|'+str(m)+str(d)
                    desc = stats.describe(der)
                    count = len([i for i in der if i>(desc[2]+desc[3])])
                    # min/max, mean, var, #?1std
                    eval_dict[ky] = [desc[1:4], count, der]
                    if desc[2]<temp:
                        hold = ky
                        temp = desc[2]
    rate_dict[r] = eval_dict
    per_r.append((hold, eval_dict[hold][0][1]))


    
print('Per Rate Best Param: ', per_r)
print('--- \n\n\n ---')
with open(tune_eval_path,'w') as out:
    json.dump(rate_dict, out)
    print('Diarization Tuning complete and dict saved')