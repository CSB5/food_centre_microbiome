# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline
from pathlib import Path
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
from sklearn import decomposition, manifold
from scipy.spatial.distance import cdist
from scipy.stats import ranksums

# %% Load Data
dataset = f'./2F_dataset.joblib'
counts_ls, mddf_ls = load(dataset)
countdf = pd.concat(counts_ls[0:2], axis=0).relabd().fillna(1e-10).drop(columns='others')[confident_location_species] #after
#-----------
mddf = pd.concat(mddf_ls[0:2], axis=0)
mddf['Location_FC2'].loc[mddf['Location_FC2']=='Ncontrol'] = np.nan
order = mddf['Location_FC2'].value_counts().index.to_series().sort_values()
mddf['Location_FC2'] = pd.Categorical(mddf['Location_FC2'], categories=order)

# %%
# Project new data onto old
old_count = countdf[mddf['Collection Round']==1]
new_count = countdf[mddf['Collection Round']==2]

metric = mgpu.spearman_distance
#metric='braycurtis'
reducer = umap.UMAP(min_dist=0.1, n_neighbors=5, metric=metric, random_state=234, n_components=2)
umap_data = pd.DataFrame(reducer.fit_transform(old_count),index = old_count.index)
umap_data2 = pd.DataFrame(reducer.transform(new_count),index = new_count.index)

# Create centroids
centroid = umap_data.groupby(mddf['Location_FC2']).mean().reset_index()
centroid['Collection Round'] = 1
centroid2 = umap_data2.groupby(mddf['Location_FC2']).mean().reset_index()
centroid2['Collection Round'] = 2

combine_umap = pd.concat([centroid,centroid2],axis=0).reset_index(drop=True)
combined_df = pd.concat([combine_umap, countdf, mddf], axis=1).dropna(subset=0)

#Plot
x,y = 0,1
figure,ax = mgpu.plotscatterplot(combine_umap,x,y, hue='Location_FC2', style='Collection Round', xlabel=f'UMAP{x}',ylabel=f'UMAP{y}', figsize=(7,5), permanova=False)

points_gb = combine_umap.groupby(['Location_FC2'])
points_gb.groups.keys()
cnt = 0
for key, subpoints in points_gb:
    x = subpoints[0]
    y = subpoints[1]
    u = np.diff(subpoints[0])
    v = np.diff(subpoints[1])
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    #norm = np.sqrt(u**2+v**2)
    color = sns.color_palette('tab20',n_colors=21)[cnt]
    opacity = np.max([(1-norm/10)[0],0.1])
    ax.plot(x,y,color=color+(opacity,))
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, headwidth=2, pivot="mid", color=color+(opacity,))
    cnt+=1
figure

# %% Boxplot insert on distance
# Calculate significant differences between distances
old_umap_gb = umap_data.groupby(mddf['Location_FC2'])
new_umap_gb = umap_data2.groupby(mddf['Location_FC2'])

sign_dist_ls = []
insign_dist_ls = []
insign = ['A','B','F','G','K','N','O']
for FC, old_group in old_umap_gb:
    new_group = new_umap_gb.get_group(FC)
    distances = cdist(new_group,old_group).flatten()
    smallest_distances = distances[distances < np.quantile(distances,0.25)].tolist() # Lowest quantile
    if FC in insign:
        insign_dist_ls = insign_dist_ls + smallest_distances
    else:
        sign_dist_ls = sign_dist_ls + smallest_distances

fig = plt.figure(figsize=(3,8))
ax = fig.add_subplot(111)
sns.boxplot(pd.concat([pd.Series(sign_dist_ls),pd.Series(insign_dist_ls)], axis=1), ax=ax)

# Calculate P-value
ranksums(sign_dist_ls,insign_dist_ls,alternative='less')
