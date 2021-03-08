
import random, os

seed = 13
device = 'cuda'
fp = '/scratch/jt2565/SCOTUS/'

audio_path = fp+'audio/'
trans_path = fp+'transcript/'
rttm_path =  fp+'rttm/'
di_path = fp+'diarization/'
inf_lab_path = fp+'inf_labelled/'
inf_path = fp+'inference/'
plot_path = fp+'plot/'

embed_path = fp+'embed_dict.json'
sd_path = fp+'ral_spkr_dict.json'
set_path = fp+'set_dict.json'
tune_eval_path = fp+'tune_eval.json'
test_eval_path = fp+'test_eval.csv'

#per docket
r_count = 2 
d_count = 2

#total
t_lim = 30

der_collar = .5
verbose=True
save_test_emb = True

#Test settings
encoder_rate = 5
mal = 7
mrt = 6
ms = 4
diar_thresh = .1

#tuning ranges
tune_rate = [1,3,5,7]

tune_mal = [3,5,7]
tune_mrt = [6,8,10]

tune_ms = [2,5,8]
tune_dt = [.05, .075, .1, .125, .15]

