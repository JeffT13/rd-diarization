
# Visualization -> ipynb
embed = np.load(inf_lab_path+temp_case+'_embeds.npy')
label = np.load(inf_lab_path+temp_case+'_embeds_labels.npy')
time = np.load(inf_lab_path+temp_case+'_embeds_times.npy')

sns.set(style = "darkgrid")
plt.figure(figsize=(2,5))
ax = sns.heatmap(embed.T, vmin=-.3, vmax=.3, xticklabels = False, yticklabels = False)

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
kmeans_tsne_scale = KMeans(n_clusters=len(U), random_state=seed).fit(tsne_df_scale)
labels_tsne_scale = Y
clusters_tsne_scale = pd.concat([tsne_df_scale, pd.DataFrame({'tsne_clusters':labels_tsne_scale})], axis=1)
centers = kmeans_tsne_scale.cluster_centers_

plt.figure(figsize = (12,12))
sns.scatterplot(clusters_tsne_scale.iloc[:,0],clusters_tsne_scale.iloc[:,1],hue=y_scotus, palette='tab10', s=100, alpha=0.6).set_title('Case '+temp_case+' - Voice Encoder embeddings \nK-means Centroid projections', fontsize=15)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='+')
plt.show()