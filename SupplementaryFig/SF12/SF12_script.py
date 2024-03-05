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
genesdf = pd.read_csv(f'./SF12_feature_selection_clr_pvalue_n100.csv', index_col=0)
speciesdf_sign = genesdf[genesdf['pval']<0.05]

# %% Process data
selected_genes = speciesdf_sign.index
selected_arg = Arg_df[(Arg_df['gene'].isin(selected_genes)) & (Arg_df['Location']=='hawker_centre')]
selected_arg['gene1'] = selected_arg['gene'].str.split('_', expand=True)[0]
input_df = selected_arg.pivot_table(index='gene1', columns='SAMid', values='RPKM')
#input_df = Arg_df.pivot_table(index='class', columns='SAMid', values='RPKM')
input_df = input_df[input_df.count(axis=1) > 10] # Only prevelant genes
input_df = input_df.fillna(1e-7)

# %% Plot
gene2class = selected_arg['gene'].str.split('_', expand=True).set_index(0).to_dict()[1]
classes = input_df.index.map(gene2class)
classes.unique()
deep = sns.color_palette('deep')
class2color = dict([i for i in zip(classes.unique(), deep)])
row_colors = pd.DataFrame(classes.map(class2color))
row_colors.index = input_df.index

hclocations = mddf.reindex(input_df.columns)['Location']
colors = sns.color_palette("tab20")
order = hclocations.value_counts().index.to_series().sort_values().index
loc2color = dict([i for i in zip(order, colors)])
col_colors = pd.DataFrame(pd.Series([loc2color[i] for i in hclocations])).set_axis(hclocations.index)


sns.set(font_scale=0.5)
sns.set_style("ticks")
fig = sns.clustermap(np.log(input_df), linewidths=0, xticklabels=True, yticklabels=True, dendrogram_ratio=(.05, .05), cbar_pos=(1, 0.2, 0.02, 0.6), figsize=(8, 5), cmap="RdPu", col_colors=col_colors, row_colors=row_colors, metric='braycurtis', vmin=0,  vmax=4, col_cluster=False)
fig.ax_heatmap.get_xaxis().set_visible(False)
fig.ax_heatmap.set_ylabel("")

handles = [Patch(facecolor=color, label=name) for name,color in class2color.items()]
first_legend = fig.ax_heatmap.legend(handles=handles, title='ARG class', bbox_to_anchor=(-0.1, 1), loc='upper right' , ncols=1)
handles = [Patch(facecolor=color, label=name) for name,color in loc2color.items()]
second_legend = fig.ax_heatmap.legend(handles=handles, title='Location', bbox_to_anchor=(0.5, 1.05), loc='lower center' , ncols=8)
first_legend.set_in_layout(True)
second_legend.set_in_layout(True)
fig.ax_heatmap.add_artist(first_legend)
