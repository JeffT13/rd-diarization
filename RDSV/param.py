#Parameters & HyperParameters

seed = 13
device = 'cuda'
verbose=True

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

train_dockets = ['17']
test_dockets = ['18']
#per docket
r_count = 3
d_count = 3

#total
t_lim = 30


#tuning ranges
tune_rate = [1,3,5,7]

tune_mal = [3,5,7]
tune_mrt = [2,4,7]

tune_ms = [2,5]
tune_dt = [.05, .075, .1, .125]

#Evaluation
der_collar = .5
LR_lims = [.8, .85, .9]
run_TSNE = False #ran locally

#Test settings
save_test_emb = True
encoder_rate = 5
mal = 7
mrt = 6
ms = 4
diar_thresh = .1