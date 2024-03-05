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

# %% load data
counts = pd.read_csv(f'./SF6_genus_counts.csv')
# Load mddf without sensitive geographical information
# %%
fig = mgpu.plot_stacked(counts.top_OTU(30), mddf, facet='Location_FC2', returnData=False, nrow=2, figsize=(16,6), package='seaborn', legend_loc='bottom', legend_fontsize = 8, legend_ncols = 5, show_xticklabels=False,cp=cp)
fig.subplots_adjust(hspace=0.15, wspace=0.1)
