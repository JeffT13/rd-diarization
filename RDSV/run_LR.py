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
    for L in LR_lims:
        print('Ensemble Classifier Accuracy for Threshold:', L,' === ', round(accuracy_score(Yd, logit_thresh(logit, L)),4))


    if run_TSNE:
        y_scotus = []
        for i in Y:
          y_scotus.append(list(spkr_dict.keys())[list(spkr_dict.values()).index(i)])
        
        y_scotusd = []
        for i in Yd:
          if not i==999:
            y_scotusd.append(list(spkr_dict.keys())[list(spkr_dict.values()).index(i)])
          else:
            y_scotusd.append('UnRefSpkr')
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, learning_rate=200, random_state=seed)
        tsne_results = tsne.fit_transform(X)
        tsne_df_scale = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        labels_tsne_scale = Y
        clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)

        tsned = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, learning_rate=200, random_state=seed)
        tsne_resultsd = tsned.fit_transform(Xd)
        tsne_df_scaled = pd.DataFrame(tsne_resultsd, columns=['tsne1', 'tsne2'])
        labels_tsne_scaled = Yd
        clusters_tsne_scaled = pd.concat([tsne_df_scaled, pd.DataFrame({'tsne_clusters':labels_tsne_scaled})], axis=1)
        
        save_out = out_path+'r'+str(r)+'/'
        if not os.path.exists(save_out):
            os.mkdir(save_out)
        np.save(save_out+'X.npy',X)
        np.save(save_out+'Y.npy',Y)
        np.save(save_out+'Xd.npy',Xd)
        np.save(save_out+'Yd.npy',Yd) 
        tsne_df_scale.to_csv(save_out+'tsne.csv')
        tsne_df_scaled.to_csv(save_out+'tsned.csv')
        
        
        
            
            
            
            
            
            
            
