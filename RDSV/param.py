#Parameters & HyperParameters

seed = 13
device = 'cuda'
verbose=True

fp = '/scratch/jt2565/data/'

audio_path = fp+'audio/'
trans_path = fp+'transcript/'
rttm_path =  fp+'rttm/'
di_path = fp+'diarization/'
inf_lab_path = fp+'inf_labelled/'
inf_path = fp+'inference/'
out_path = fp+'out/'

sd_path = out_path+'ral_spkr_dict.json'
set_path = out_path+'set_dict.json'
tune_eval_path = out_path+'tune_eval.json'
test_eval_path = out_path+'test_eval.csv'

train_dockets = ['16']
test_dockets = ['17']
#per docket
r_count = 5
d_count = 5

#total
t_lim = 30


#tuning ranges
tune_rate = [1,3,5,7,9]

tune_mal = [3,5,7]
tune_mrt = [2,4,7]

tune_ms = [2,5]
tune_dt = [.05, .075, .1, .125]

#Evaluation
der_collar = .5
LR_lims = [.85, .9, .95]
run_TSNE = False #ran locally

#Test settings
save_test_emb = False
encoder_rate = 5
mal = 7
mrt = 2
ms = 2
diar_thresh = .075