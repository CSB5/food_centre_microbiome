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

# %% Load Data
food_order = pd.read_csv(f'./SF5_food_order_OTU_filtered_v2.csv')
microbialCounts = pd.read_csv(f'./SF5_microbe_order_OTU_filtered_v2.csv')
y = microbialCounts
x = food_order

# %% Perform CCA
for i in [results.eigvals, results.samples, results.features, results.biplot_scores, results.sample_constraints, results.proportion_explained]:
    print(i.shape)

results.samples

FC = mddf['Location_FC2']
FC[FC=='Ncontrol'] = None
category = pd.Categorical(mddf['Location_FC2'], ordered=True)
axisX = 'CCA1'
axisY = 'CCA2'
fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(data=results.samples, x=axisX,y=axisY, hue = category, palette = 'tab20', ax=ax)
ax.set_xlim((-3,1.5))
ax.set_ylim((-2,2))
for index, coord in results.biplot_scores.iterrows():
    dist = np.sqrt((coord[axisX]*3)**2+(coord[axisY]*3)**2)
    ax.arrow(0,0,coord[axisX]*3,coord[axisY]*3, color='red', head_width=0.1)
    if dist>1:
        ax.text(coord[axisX]*3,coord[axisY]*3,index,horizontalalignment='right')
fig
