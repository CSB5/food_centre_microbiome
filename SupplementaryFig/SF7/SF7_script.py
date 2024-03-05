# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
from itertools import combinations_with_replacement
from scipy.spatial.distance import squareform, pdist
import scipy
from scipy.stats import linregress, pearsonr
import numpy as np
import seaborn as sns
sns.set(font_scale=1)
#sns.set_style('white')
sns.set_style("ticks")
#import skbio
from skbio.diversity import alpha
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

# %% Load Data
all_countdf = pd.read_csv(f'SF7_countdf.csv')
# Load all_mddf after removing geographical content

# %% Calculate diversity
alphaname = ['shannon', 'simpson','chao1']

diversity = [all_countdf.top_OTU(500).apply(getattr(alpha,a), axis = 1).rename(a) for a in alphaname]
all_countdf[all_countdf<0.0005]=np.NaN
richness = (~all_countdf.isna()).sum(axis=1)
richness.name = 'richness'

alphaDF = mgp.countDataFrame(pd.concat(diversity, axis = 1))
alphaDF['richness'] = richness
alphaDF['Location'] = all_mddf['Location']
alphaDF['Location2'] = all_mddf['Location2']
alphaDF['Location_FC'] = all_mddf['Location_FC']
alphaDF['cohort'] = all_mddf['cohort']
alphaDF = alphaDF[alphaDF['Location']!='NControl']

locations = alphaDF['Location'].value_counts().index.sort_values()
alphaDF['Location'] = pd.Categorical(alphaDF['Location'],categories=locations)

# %% Plot
alphaDF2 = alphaDF.copy()
alphaDF2['cohort'] = alphaDF2['cohort'].map({'Hawker Centre':'Hawker','Hawker Centre (New)':'Hawker','Hospital':'Hospital','Subway':'MetaSUB','Office':'Office'})
#sns.set(font_scale=1, style='white')

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(131)
fig,ax = mgpu.boxplot_statannot(alphaDF2, x='cohort', y='shannon', box_pairs=[('Hawker','Hospital'),('Hawker','MetaSUB'),('Hawker','Office')],ax=ax,logbase=None,xlabel='',ylabel='',title='shannon', palette=sns.color_palette("tab10"))

ax = fig.add_subplot(132)
fig,ax = mgpu.boxplot_statannot(alphaDF2, x='cohort', y='simpson', box_pairs=[('Hawker','Hospital'),('Hawker','MetaSUB'),('Hawker','Office')],ax=ax,logbase=None,xlabel='',ylabel='',title='simpson',palette=sns.color_palette("tab10"))

ax = fig.add_subplot(133)
fig,ax = mgpu.boxplot_statannot(alphaDF2, x='cohort', y='richness', box_pairs=[('Hawker','Hospital'),('Hawker','MetaSUB'),('Hawker','Office')],ax=ax,logbase=None,xlabel='',ylabel='',title='richness',palette=sns.color_palette("tab10"))
fig

fig.savefig(f'./fig/{scriptname}_diversity_hawker_vs_hos_off_meta.pdf',transparent=True)
