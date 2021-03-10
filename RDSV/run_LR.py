import os, json, timeit
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
 
from VoiceEncoder.util import casewrttm_to_dvec
from rdsv import RefAudioLibrary, Diarize, diar_to_rttm, rttmto_RALrttm
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
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
    U = np.unique(Y)    

    y_scotus = []
    for i in Y:
      y_scotus.append(list(spkr_dict.keys())[list(spkr_dict.values()).index(i)])
  
  
    dev_seq = []
    dev_id = []
    hold_label_d = 0
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
    Ud = np.unique(Yd)    

    y_scotusd = []
    for i in Yd:
      if not i==999:
        y_scotusd.append(list(spkr_dict.keys())[list(spkr_dict.values()).index(i)])
      else:
        y_scotusd.append('UnRefSpkr')
    
    

    clf = OneVsRestClassifier(LogisticRegression()).fit(X, Y)
    logit = clf.predict_proba(Xd)
    print('Ensemble Classifier Accuracy for Rate:', r, ' === ', round(accuracy_score(Yd, logit_thresh(logit, LR_lim)),4))


    if run_TSNE:
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, learning_rate=200, random_state=seed)
        tsne_results = tsne.fit_transform(X[:hold_lim_r])
        tsne_df_scale = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        labels_tsne_scale = Y[:hold_lim_r]
        clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)

        tsned = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, learning_rate=200, random_state=seed)
        tsne_resultsd = tsned.fit_transform(Xd[:hold_lim_d])
        tsne_df_scaled = pd.DataFrame(tsne_resultsd, columns=['tsne1', 'tsne2'])
        labels_tsne_scaled = Yd[:hold_lim_d]
        clusters_tsne_scaled = pd.concat([tsne_df_scaled, pd.DataFrame({'tsne_clusters':labels_tsne_scaled})], axis=1)
