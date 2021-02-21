
import random, os

fp = '/scratch/jt2565/SCOTUS/'

audio_path = fp+'audio/'
trans_path = fp+'transcript/'
rttm_path =  fp+'rttm/'
di_path = fp+'diarization/'
inf_lab_path = fp+'inf_labelled/'
inf_path = fp+'inference/'
sd_path = fp+'ral_spkr_dict.json'
eval_path = fp+'eval_dict.json'

c_set_count = 5
d_set_count = 10
der_collar = .5

device = 'cuda'
verbose=True

t_lim = 20+(c_set_count+d_set_count)
seed = 13

#Generate random case sets
if os.path.exists(audio_path):
    cases = os.listdir(audio_path)
    if seed is not None:
        random.Random(seed).shuffle(cases)
    else:
        random.Random().shuffle(cases)
        
    c_set = cases[:c_set_count]
    d_set = cases[c_set_count:(d_set_count+c_set_count)]
    if t_lim is not None:
        t_set = cases[d_set_count:t_lim]
    else:
        t_set = cases[d_set_count:]
else:
    print('File paths do not exist')
