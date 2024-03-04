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

"""
Code truncated from food_microbe_correlation.py
"""
# %% Load values
#Correlation and p-value generated from fastspar
corr = pd.read_csv('./1D_corr.csv')
pval = pd.read_csv('./1D_pval.csv')
#load microbial_OTU
#load food_OTU
row_mddf = pd.read_csv(f'./1D_row.csv')
col_mddf = pd.read_csv(f'./1D_col.csv')

# %% Filter Correlation
corr_fil = corr.copy()[(pval < 0.05)] # Filter for significant correlations
corr_fil[corr_fil.abs()<threshold] = np.nan # Filter for correlations more than threshold
weak_corr = corr_fil.loc[:,corr_fil.abs().fillna(0).max(axis=0)<max_threshold] # Find microbes with weak correlation
weak_corr_few_corr = weak_corr.loc[:,weak_corr.notna().sum()<=prevalence_threshold].columns # Find micrbes with few weak correlations
corr_fil = corr_fil.loc[:,~corr_fil.columns.isin(weak_corr_few_corr)] # Select microbes without weak correlations

input_df = corr_fil.fillna(0)
input_df.loc['Primates'].sort_values(ascending=False)
#input_df.to_csv(f'{FOLDER}/correlation_filtered.csv')

# %% Setup category label of figures
row_cp = mgpu.custom_palette('tab10',10)
col_cp = mgpu.custom_palette('Set2',10)
new_cmap = mgpu.custom_gradient(cmap="vlag",fraction=[0, 0.1, 0.2, 0.8, 0.9, 1])
#new_cmap = mgpu.custom_gradient(cmap="vlag")
sns.set(font_scale=0.7,style='white')

fig = mgpu.clustermap(input_df, row_mddf, col_mddf, cp=new_cmap, col_cp=col_cp, row_cp=row_cp, metric='euclidean',method='ward', figsize=(8,5), vmin=-0.8, vmax=0.8, cbar=(0.97, 0.28, .03, .65), dendrogram_ratio=(0.1,0.1),linewidths=0.00, mask=input_df.abs()<display_threshold)
