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
kingdom_counts = pd.read_csv('1C_kingdom_counts.csv')

# %% plot data
sns.set_style("whitegrid")
ax = sns.boxplot(data=kingdom_counts)
ax.set_ylabel('Relative Abundance')
ax.get_figure().savefig(f'./fig/{scriptname}_kingdom_level_boxplot.pdf')
