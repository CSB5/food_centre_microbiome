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

# %% Load data
#Correlation and p-value generated from fastspar
corr = pd.read_csv('../../Fig1D/1D_corr.csv')
pval = pd.read_csv('../../Fig1D/1D_pval.csv')
#load microbial_OTU
#load food_OTU
row_mddf = pd.read_csv(f'../../Fig1D/1D_row.csv')
col_mddf = pd.read_csv(f'../../Fig1D/1D_col.csv')

# %% Plot
#genuslim = 0.03
#OTUlim = 0.003
corrdf_fil = corrdf[pvaldf < 0.05]
top_corr = corrdf_fil.loc[microbial_OTU, genus].apply(
    abs).sort_values(ascending=False)
top_pvalue = pvaldf.loc[top_corr.index.to_list(), genus]

# Regression Plot
fig = plt.figure(figsize=(10, 8))
for n in range(9):
    OTU = top_corr.index[n]
    data_fil = counts.relabd()[(counts.relabd()[genus] < genuslim) & (
        counts.relabd()[OTU] < OTUlim)]  # Filter outliers for plotting purposes
    ax = fig.add_subplot(3, 3, n+1)
    #ax.set_xlim(0.0, genuslim)
    #ax.set_ylim(0.0, OTUlim)
    ax.annotate(r'$\rho$ = '+f'{top_corr[n]}, \np-value = {top_pvalue[n]:1.1e}', xy=(
        0.1, 0.93), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', fontsize='smaller')
    sns.regplot(ax=ax, x=genus, y=OTU, data=data_fil)
fig.tight_layout()
fig
