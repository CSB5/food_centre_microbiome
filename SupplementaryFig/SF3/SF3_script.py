%load_ext autoreload
%autoreload 2
%matplotlib inline
from matplotlib import cm, colors
import matplotlib as mpl
import biom
from scipy.stats import pearsonr
from skbio.stats.ordination import cca
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display, IFrame
from scipy.stats.mstats import gmean
import scipy
import statsmodels
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import os
import sys
from joblib import dump,load

# %% Load Data
all_counts = load('./SF3_dataset.joblib')
# Load mddf without geographical data

food_genera = pd.read_csv('./SF3_food_genera_w_lineage_curated.csv').set_index('genus')


# %% Plot SF 3A
fooddf = all_counts[food_genera.index]
fooddf['total_relabd'] = fooddf.sum(axis=1)
fooddf['cohort'] = fooddf.index.map(all_mddf['cohort2'].to_dict())
input = fooddf[['total_relabd','cohort']]
input = input[input['total_relabd']>0]
input[input['cohort']=='Hospital'].sort_values('total_relabd')
#input = input.drop(index=['WEE063','WEE304','WEE267'])

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
sns.boxplot(data=input,x='cohort',y='total_relabd',palette='tab10',flierprops={"marker": "d", "markerfacecolor":'#000000',"markersize":6},ax=ax, whis=0.8, log_scale=True)
#ax.set_yscale('log',base=10)
ax.set_ylabel('Relative Abundance')
ax.set_ylim((0.00005,5))
ax.set_xlabel('')
ax.set_xticklabels(['Food centre','Hospital','Office','MetaSUB'])
fig

# %% Plot SF 3B
# Plot relative abundance across food_types in hawker centres
fooddf = all_counts[food_genera.index]
food_class_df = fooddf.T.groupby(food_genera['common_class']).sum().T
food_class_df['cohort'] = food_class_df.index.map(all_mddf['cohort2'].to_dict())
input = food_class_df.melt(id_vars='cohort', var_name='food_type', value_name='relabd').groupby('cohort').get_group('Hawker Centre')
input = input[input['relabd']>0]
order = input.groupby('food_type').median().sort_values('relabd',ascending=False).index
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
sns.boxplot(data=input,x='food_type',y='relabd',palette='tab10',flierprops={"marker": "d", "markerfacecolor":'#000000'},ax=ax, order=order, log_scale=True)
#ax.set_yscale('log',base=10)
ax.set_ylabel('Relative Abundance')
#ax.set_ylim((0.0000001,1))
ax.set_xlabel('')
fig
fig.savefig(f'./fig/{scriptname}_boxplot_individual_food_across_loc.pdf')
