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
counts = load('./SF2_dataset.joblib')
# Load mddf without geographical data

# %% Food
food_genera_w_lineage = pd.read_csv(f'../abundance/output/food_genera_w_lineage_curated.csv',index_col='genus')

ratio_ls = []
counts_gb = counts.groupby_phylo("order")
for ord in top_order:
    ratio = counts_gb.get_group(ord)[counts_gb.get_group(ord).index.isin(food_genera_w_lineage.index)].sum() / counts_gb.get_group(ord).sum()
    ratio.name = ord
    ratio_ls.append(ratio)
    print(ord)
    print(counts_gb.get_group(ord).index.isin(food_genera_w_lineage.index))

food_ratio = pd.DataFrame(np.array(ratio_ls)).T.set_axis(top_order, axis=1)
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot()
sns.boxplot(food_ratio, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, style='italic')
fig.savefig(f'./fig/{scriptname}_boxplot_food_ratio.pdf')
