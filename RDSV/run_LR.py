import os, json, timeit
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from param import *


with open(set_path) as json_file: 
    set_dict = json.load(json_file)

with open(sd_path) as json_file: 
    spkr_dict = json.load(json_file)

def logit_thresh(log, l):
  pred = []
  for item in log:
    if max(item)>l:
      pred.append(np.argmax(item))
    else:
      pred.append(999)
  return pred

rate_lr = {}

for r in tune_rate:
    print('Processing for rate=', r)
    path_out = inf_lab_path+'r'+str(r)+'/'
    
    
    hold_label_r = 0
    tr_seq = []
    tr_id = []
    for wav in set_dict['r']:
        case = wav.split('.')[0]
        embed = np.load(path_out+'{}_embeds.npy'.format(case))
        label = np.load(path_out+'{}_embeds_labels.npy'.format(case))
       
        for j, id in enumerate(label):
            if id<20: #judge speakers
                tr_seq.append(embed[j])
                tr_id.append(id)
        if not hold_label_r:
            hold_label_r = len(tr_id)
    X = np.asarray(tr_seq)
    Y = np.asarray(tr_id)


    hold_label_d = 0
    dev_seq = []
    dev_id = []
    for wav in set_dict['d']:
      case = wav.split('.')[0]
      embed = np.load(path_out+'{}_embeds.npy'.format(case))
      label = np.load(path_out+'{}_embeds_labels.npy'.format(case))
      
      for j, id in enumerate(label):
        if id<900: # no overlap or padding 
            dev_seq.append(embed[j])
            if id>=20:
              dev_id.append(999)
            else:
              dev_id.append(id)
      if not hold_label_d:
        hold_label_d = len(dev_id)
    Xd = np.asarray(dev_seq)
    Yd = np.asarray(dev_id)

    # Train LR
    clf = OneVsRestClassifier(LogisticRegression()).fit(X, Y)
    logit = clf.predict_proba(Xd)
    temp = []
    for L in LR_lims:
        temp.append(round(accuracy_score(Yd, logit_thresh(logit, L)),4))
        print('Ensemble Classifier Accuracy for Threshold:', L,' === ', temp[-1])
    rate_lr[r]=temp


    if save_LR:
        save_out = out_path+'r'+str(r)+'/'
        if not os.path.exists(save_out):
            os.mkdir(save_out)
        np.save(save_out+'X.npy',X)
        np.save(save_out+'Y.npy',Y)
        np.save(save_out+'Xd.npy',Xd)
        np.save(save_out+'Yd.npy',Yd) 
            
        
with open(lr_path,'w') as out:
    json.dump(rate_lr, out)
    print('LR complete & dict saved') 
            
            
            
            
            
            
            
