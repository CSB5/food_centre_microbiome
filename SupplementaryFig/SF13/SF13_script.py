%load_ext autoreload
%autoreload 2
from pathlib import Path
from statannot import add_stat_annotation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from scipy.stats import binomtest, ranksums
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from sklearn import metrics
from sklearn import decomposition
import numpy as np
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# %% Load Data
#============ By total RPKM
arg_dataset = '../../Fig3A/3A_dataset.joblib'
Arg_df, mddf = load(arg_dataset)
RPKMdf = Arg_df.groupby(['SAMid']).agg({'RPKM':'sum','Location':'first'}).reset_index()
RPKMdf.groupby('Location')['RPKM'].median()

# %%
bla_arg = Arg_df.groupby('class').get_group('Bla')
temp_mddf = Arg_df[['SAMid', 'Location']]
hc_samid = temp_mddf[temp_mddf['Location']=='hawker_centre']['SAMid'].unique()
hc_bla_arg = bla_arg.loc[bla_arg['SAMid'].isin(hc_samid)]
hc_h_samid = temp_mddf[temp_mddf['Location'].isin(['hawker_centre','hospital'])]['SAMid'].unique()
hc_h_bla_arg = bla_arg.loc[bla_arg['SAMid'].isin(hc_h_samid)]

# Compare between hospital and hawker centres
OXA_arg = hc_h_bla_arg[hc_h_bla_arg['allele'].str.contains('OXA')]
OXA_median = OXA_arg.groupby('allele')['RPKM'].median().sort_values()
OXA_count = OXA_arg.groupby('allele')['RPKM'].count().sort_values()
OXA_of_interest = OXA_count[OXA_count>=2].index #prevalence filter
#OXA_of_interest = OXA_of_interest[OXA_of_interest.isin(OXA_median[OXA_median>1].index)] # abundance filter
OXA_of_interest = OXA_median[OXA_of_interest].sort_values(ascending=False).index
data = OXA_arg.loc[OXA_arg['allele'].isin(OXA_of_interest)]
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111)
sns.boxplot(data=data,y='RPKM',x='allele', hue='Location',palette='tab10', order=OXA_of_interest, hue_order=['hawker_centre','hospital'], ax = ax)
#data = OXA_arg
#sns.boxplot(data=data,y='RPKM',x='allele',palette='tab10', ax = ax)
ax.tick_params(axis='x', rotation=90)
ax.set_yscale("log", base=10)
fig
