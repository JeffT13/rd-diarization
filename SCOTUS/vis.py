import os, json, random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from param import *

sns.set(style = "darkgrid")

from sklearn.manifold import TSNE
from param import *

with open(set_path) as json_file: 
    set_dict = json.load(json_file)

with open(sd_path) as json_file: 
    spkr_dict = json.load(json_file)
# plot random case from c set
temp = [item.split('.')[0] for item in set_dict['c']]
temp_case = temp[random.randint(0, c_case_count)]
embed = np.load(inf_lab_path+temp_case+'_embeds.npy')
label = np.load(inf_lab_path+temp_case+'_embeds_labels.npy')
time = np.load(inf_lab_path+temp_case+'_embeds_times.npy')

# D-vector heatmap
plt.figure(figsize=(2,5))
ax = sns.heatmap(embed.T, vmin=-.3, vmax=.3, xticklabels = False, yticklabels = False)
ax.savefig(plot_path+temp_case"_heat.png")


# TsnE
case_seq = []
case_id = []
for j, emb in enumerate(label):
    if emb<99: # no overlap or padding (All speakers)
        case_seq.append(embed[j])
        case_id.append(emb)

X = np.asarray(case_seq)
Y = np.asarray(case_id)
U = np.unique(Y)    

y_scotus = []
for i in Y:
  y_scotus.append(list(spkr_dict.keys())[list(spkr_dict.values()).index(i)])

print("# of unique speakers", len(U))

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000, learning_rate=200, random_state=seed)
tsne_results = tsne.fit_transform(X)
tsne_df_scale = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)

plt.figure(figsize = (12,12))
sns.scatterplot(clusters_tsne_scale.iloc[:,0],clusters_tsne_scale.iloc[:,1],hue=y_scotus, palette='tab10', s=100, alpha=0.6).set_title('Case '+temp_case+' - Embeddings', fontsize=15)
plt.savefig(plot_path+temp_case+'_tSNE.png')




if r==tune_rate[0] or r==encoder_rate:
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
            