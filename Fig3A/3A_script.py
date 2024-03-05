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


#============ By total RPKM
arg_dataset = './3A_dataset.joblib'
Arg_df, mddf = load(arg_dataset)
RPKMdf = Arg_df.groupby(['SAMid']).agg({'RPKM':'sum','Location':'first'}).reset_index()
RPKMdf.groupby('Location')['RPKM'].median()

sns.set(font_scale=1)
sns.set_style("ticks")
#sns.set_style('whitegrid')
fig = plt.figure(figsize=(4, 3.5))
ax = fig.add_subplot(111)
deep = sns.color_palette('deep')
sns.boxplot(data=RPKMdf, x='Location', y='RPKM', ax=ax, order=['hawker_centre','hospital','office', 'metasub'], palette = [deep[i] for i in [0,1,2,3]], flierprops={"marker": "d","markerfacecolor":'#000000'})
#ax.set(xticklabels = ['hospital','hawker\ncentre','office', 'metasub'])
ax.set_yscale("log", base=10)
ax.set(xticklabels = ['Food\ncourt','Hospital','Office', 'MetaSUB'])
ax.set_xlabel(None)
