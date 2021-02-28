
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
test_eval_path = fp+'test_eval.json'


c_set_count = 5
d_set_count = 10
t_lim = 20+(c_set_count+d_set_count)

der_collar = .5

encoder_rate = 4
diar_thresh = .1
mal = 4
mrt - 8
ms = 5

tune_rate = [1,2,3,4,5]
tune_dt = [.05, .075, .1, .125, .15]
tune_mal = [3,4,5]
tune_mrt = [6,8,10]
tune_ms = [4,8,12]


verbose=False
save_test_emb = True