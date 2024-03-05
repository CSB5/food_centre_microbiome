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

# %% Plot Heatmap gene / gene class : Beta Lactamase
#======== Beta Lactamase
bla_arg = Arg_df.groupby('class').get_group('Bla')
bla_arg = bla_arg[bla_arg['Location']!='new_swab']
temp_mddf = Arg_df[['SAMid', 'Location']]
hc_samid = temp_mddf[temp_mddf['Location']=='hawker_centre']['SAMid'].unique()
# chr intrinsic chromosomal β-lactamase
# ESBL extendedspectrum beta lactamases
# broad : broad spectrum
# inhR Beta-lactamases with resistance to beta-lactamase inhibitors
# Filter for Blam that are present in at least 5 hawker centres? 6? 7?

bla_arg[bla_arg['gene']=='CARB']
bla_arg['gene'] = bla_arg['gene'].str.split('_', expand=True)[0]
input_df = bla_arg.pivot_table(index='gene', columns='SAMid', values='RPKM')
input_df_hc = input_df.loc[:,input_df.columns.isin(hc_samid)]
#input_df = Arg_df.pivot_table(index='class', columns='SAMid', values='RPKM')
#input_df = input_df[input_df_hc.count(axis=1) > 2] # Only prevelant genes
input_df = input_df[input_df.count(axis=1) > 30]

input_df = input_df.fillna(0.00000001)

# Color row by bla type
gene2class = bla_arg.groupby('gene').agg({'bla_class':'first'}).to_dict()['bla_class']
classes = input_df.index.map(gene2class)
class2color = dict([i for i in zip(classes.unique(), sns.color_palette('Set2'))])
row_colors = pd.DataFrame(classes.map(class2color))
row_colors.index = input_df.index
#row_colors.columns = ['']

# Color column by location
deep = sns.color_palette('deep')
lut = dict(zip(['hawker_centre', 'hospital', 'office', 'metasub'], deep[0:4]))
samid2Location = Arg_df[['SAMid', 'Location']].groupby(
    'SAMid').agg('first').to_dict()['Location']
Location = input_df.columns.map(samid2Location)
col_colors = pd.DataFrame(Location.map(lut))
col_colors.column = ['Location']
col_colors.index = input_df.columns
col_colors = pd.DataFrame(dict([(loc, col_colors[Location==loc][0]) for loc in ['hawker_centre', 'hospital', 'office', 'metasub']])) #separates out each location

# Linkage
col_linkage = linkage(pdist(np.log(input_df).T, metric ='jaccard'), method='ward')
row_linkage = linkage(pdist(np.log(input_df), metric ='euclidean'), method='ward')

# Plot heatmap
sns.set(font_scale=0.5)
sns.set_style("ticks")
new_cmap = mgpu.custom_gradient(cmap="RdPu",fraction=[0,0.6,0.7,0.8,0.9,1])
fig = sns.clustermap(np.log10(input_df), row_linkage=row_linkage, col_linkage=col_linkage, linewidths=0, xticklabels=True, yticklabels=True, dendrogram_ratio=(.1, .1), cbar_pos=(
    1.01, 0.2, .03, .5), figsize=(8, 5), cmap="RdPu", col_colors=col_colors, row_colors=row_colors, metric='euclidean', vmin=0, vmax=1.5)
fig.ax_heatmap.get_xaxis().set_visible(False)

handles = [Patch(facecolor=color[1], label=name) for name,color in zip(['Bla','Broad Spectrum Bla','Extended Spectrum Bla','Carbapenemase','Intrisically Chromosomal Bla'],class2color.items())]
first_legend = fig.ax_heatmap.legend(handles=handles, title='Types', bbox_to_anchor=(-0.1, 1), loc='upper right' , ncols=1)
fig
