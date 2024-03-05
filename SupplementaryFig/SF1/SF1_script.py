%load_ext autoreload
%autoreload 2
%matplotlib inline
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
from sklearn import decomposition, manifold
import plotnine as pn
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from joblib import dump,load
# %% Load data
taxa = 'G'
# mddf
counts = load(f'./SF1_dataset.joblib')

# Animal
dataf = counts.relabd().get_animals()
dataf = dataf.groupby_phylo("order").sum().T
dataf = dataf.relabd()
dataf = dataf.drop(columns='others')
dataf = dataf.loc[:,(dataf>0.001).sum()>5] #filter_by_prevalence
dataf[dataf<0.001] = np.nan # To control boxplot's lower ylim
dataf = dataf.reindex(dataf.median().sort_values(ascending=False).drop(index=['Primates','Rodentia','Accipitriformes','Apterygiformes','Diptera'])[:10].index, axis=1)
print(dataf.iloc[:,0:4].columns)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
#sns.stripplot(dataf, ax=ax, jitter=0.3)
sns.set_theme(style="white")
sns.boxplot(dataf, orient='x', ax=ax, palette='tab10',flierprops={"marker": "d", "markerfacecolor":'#000000'},)
ax.set_ylim((0.0005,1.2))
ax.set_yscale("log", base=10)
ax.set_xlabel(None)
ax.set_ylabel('Relative Abundance')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, style='italic')
fig.tight_layout()
fig

# Plants
dataf = counts.relabd().get_plants()
dataf = dataf.groupby_phylo("order").sum().T
dataf = dataf.loc[:,(dataf>0.0001).sum()>5] #filter_by_prevalence
dataf[dataf<0.0001] = np.nan
dataf = dataf.reindex(dataf.median().sort_values(ascending=False)[:10].index, axis=1)
print(dataf.iloc[:,0:4].columns)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
#sns.stripplot(dataf, ax=ax, jitter=0.3)
sns.boxplot(dataf, ax=ax, palette="tab10")
ax.set_ylim((0.0004,0.1))
ax.set_yscale("log", base=10)
ax.set_xlabel(None)
ax.set_ylabel('Relative Abundance')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, style='italic')
#fig.tight_layout()
