#Parameters & HyperParameters

seed = 13
device = 'cuda'
verbose=True

fp = '/scratch/jt2565/gs/'

audio_path = fp+'audio/'
trans_path = fp+'transcript/'
rttm_path =  fp+'rttm/'
di_path = fp+'diarization/'
inf_lab_path = fp+'inf_labelled/'
inf_path = fp+'inference/'
out_path = fp+'out/'

sd_path = out_path+'ral_spkr_dict.json'
set_path = out_path+'set_dict.json'
lr_path = out_path+'lr_dict.json'
tune_eval_path = out_path+'tune_eval.json'
test_eval_path = out_path+'test_eval.csv'

train_dockets = ['14', '16', '18']
test_dockets = ['15', '17']
#per docket
r_count = 5
d_count = 10

#total
t_lim = None


#tuning ranges
tune_rate = [1,3,5,7]

tune_sct = [.75, .8, .85, .9]
tune_dt = [.05, .075, .1, .125]

#Evaluation
der_collar = .5
LR_lims = [.85, .9, .95]
save_LR = True

#Test settings
save_test_emb = True
encoder_rate = 5
mal = 8
score=.85
diar = .1