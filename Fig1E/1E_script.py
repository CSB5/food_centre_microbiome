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
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
# %% Setup and load data
countdf = pd.read_csv('./1E_countdf.csv')

# %% Define UMAP function
def plotumap(dataf, mddf=None, x=0, y=1, min_dist=0.1, n_neighbors=10, n_components=2, metric="braycurtis", random_state=123, hue=None, style=None, ax=None, permanova=False, permanova_proj=True, returnData=False, **kwargs):
    """
    permanova_proj: True -> Use projected coordinates to calculate PERMANOVA
                    False -> Use distance matrix to calculate PERMANOVA
    """
    distmat = kwargs.pop('distmat', None)
    if distmat is None:
        distmat = gen_distmat(dataf, metric)
    permutations = kwargs.pop('permutations',10000)
    reducer = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors, n_components=n_components, metric=metric, random_state=random_state)
    if metric=='precomputed':
        X = distmat
    else:
        X = dataf
    umap_data = reducer.fit_transform(X)
    combined_df = pd.concat([pd.DataFrame(umap_data, index=dataf.index),dataf, mddf], axis=1).dropna(subset=0)
    # Permanova
    if permanova:
        if permanova_proj:
            permanova = PERMANOVA(datadf=pd.DataFrame(umap_data, index=dataf.index), group=mddf[hue], permutations=10000, method="euclidean")
        else:
            permanova = PERMANOVA(datadf=pd.DataFrame(umap_data, index=dataf.index), distmat=distmat, group=mddf[hue], permutations=10000)
        print(permanova)
    if returnData:
        return combined_df
    # Plot
    figure, ax = plotscatterplot(combined_df, x, y, ax=ax, hue=hue, style=style, xlabel=f'UMAP{x}',ylabel=f'UMAP{y}', permanova=permanova, **kwargs)

    return figure, ax

# %% Plot umap clusters
sns.set(font_scale=1,style='white')
order = mddf['Location_FC2'].value_counts().index.to_series().sort_values()
mddf['Location_FC2'] = pd.Categorical(mddf['Location_FC2'], categories=order)
fig,ax = mgpu.plotumap(countdf, mddf, x=0, y=1, min_dist=0.1, n_neighbors=10, metric=mgpu.spearman_distance, random_state=123, hue='Location_FC2', style=None, ax=None, title="UMAP, Spearman", figsize=(7,5.5), permanova=True, s=120)
