
import random, os

fp = '/scratch/jt2565/SCOTUS/'

audio_path = fp+'audio/'
trans_path = fp+'transcript/'
rttm_path =  fp+'rttm/'
di_path = fp+'diarization/'
inf_lab_path = fp+'inf_labelled/'
inf_path = fp+'inference/'
sd_path = fp+'ral_spkr_dict.json'
set_path = fp+'set_dict.json'
eval_path = fp+'eval_dict.json'
plot_path = fp+'plot/'

c_set_count = 5
d_set_count = 10

der_collar = .5
encoder_rate = 4
diar_thresh = .1

device = 'cuda'
verbose=True
save_test_emb = False
t_lim = 20+(c_set_count+d_set_count)
seed = 13
tune=True

if tune:
    encoder_rate = 4
    diar_thresh = .05
    di_path=di_path+'d/'
    verbose = False