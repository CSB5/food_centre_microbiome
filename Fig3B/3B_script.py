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
arg_dataset = '../Fig3A/3A_dataset.joblib'
Arg_df, mddf = load(arg_dataset)
RPKMdf = Arg_df.groupby(['SAMid']).agg({'RPKM':'sum','Location':'first'}).reset_index()
RPKMdf.groupby('Location')['RPKM'].median()
RPKMdf = Arg_df.groupby(['SAMid','class']).agg({'RPKM':'sum','Location':'first'}).reset_index()
RPKMdf['Location'] = RPKMdf['Location'].map({'new_swab':'food centre','hawker_centre':'food centre','office':'office','hospital':'hospital','metasub':'metasub'})
RPKMdf['Location'] = pd.Categorical(RPKMdf['Location'],['food centre','hospital','office','metasub'])
RPKMdf = RPKMdf[RPKMdf['class'].isin(['AGly','Bla','Fcyn','MLS','Tet','Phe','Sul','Col'])]

sns.set(font_scale=1.4)
sns.set_style("ticks")
#sns.set_style('whitegrid')
fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot()
sns.boxplot(data=RPKMdf, x='class', y='RPKM', hue='Location', ax=ax, flierprops={"marker": "d","markerfacecolor":'#000000'})
ax.set(yscale="log")
ax.legend(loc='upper right')
#ax.set_xticklabels(['Aminoglycosides','Beta-lactam','Fosfomycin','Macrolide','Tetracycline','Phenicol','Sulfonamide','Colistin'], horizontalalignment='center')
ax.set_xlabel('Antibiotic Class')
ax.tick_params(axis='x', rotation=45)
fig.savefig(f'./fig/{scriptname}_boxplot_RPKM_abundance_by_gene.pdf')

RPKMdf[RPKMdf['class']=='Col'].groupby('Location').median()
RPKMdf[RPKMdf['class']=='Col'].groupby('Location').mean()

# Get P Values
pv_list = []
for Ab, Abgroup in RPKMdf.groupby('class'):
    a = Abgroup.loc[Abgroup['Location']=='food centre','RPKM'].reset_index(drop=True)
    b = Abgroup.loc[Abgroup['Location']=='hospital','RPKM'].reset_index(drop=True)

    c = Abgroup.loc[Abgroup['Location']=='office','RPKM'].reset_index(drop=True)
    d = Abgroup.loc[Abgroup['Location']=='metasub','RPKM'].reset_index(drop=True)
    ranksums(a,b)
    rs = lambda u,v:ranksums(u[~np.isnan(u)],v[~np.isnan(v)]).pvalue
    pv_list.append((Ab, squareform(pdist(pd.DataFrame([a,b,c,d]),metric=rs))))
