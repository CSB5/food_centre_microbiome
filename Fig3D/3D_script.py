%load_ext autoreload
%autoreload 2
%matplotlib inline
from pathlib import Path
from scipy.stats import binomtest, ranksums
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
from joblib import dump, load
from sklearn import decomposition, manifold

# %% Load Data
locations = ['hawker', 'hospital', 'office', 'subway']
counts_ls,mddf_ls = load(dataset)

# %% Plot Eskape pathogens
ESKAPEE=['Enterococcus faecium', 'Staphylococcus aureus', 'Klebsiella pneumoniae', 'Acinetobacter baumannii', 'Pseudomonas aeruginosa','Enterobacter spp.','Escherichia coli']

count_tall_ls = []
for countdf, loca in zip(counts_ls,locations):
    ESKAPEdf = countdf.loc[:,countdf.columns.isin(ESKAPE)]
    Enterobacter = countdf.columns[countdf.columns.str.contains('Enterobacter')]
    ESKAPEdf['Enterobacter spp.'] = countdf.loc[:,countdf.columns.isin(Enterobacter)].sum(axis=1)
    ESKAPE_tall = ESKAPEdf.melt(ignore_index=False).reset_index()
    ESKAPE_tall['Location'] = loca
    count_tall_ls.append(ESKAPE_tall)

ESKAPE_tall = pd.concat(count_tall_ls)
ESKAPE_tall = ESKAPE_tall[ESKAPE_tall['value']>0.001]
#ESKAPE_tall['value'] = ESKAPE_tall['value']*100 # Convert proportion to percentages

# Plot
from matplotlib.ticker import ScalarFormatter, MultipleLocator, LogLocator

class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.0f"

xScalarFormatter = ScalarFormatterClass(useMathText=True)
xScalarFormatter.set_powerlimits((0,0))

datadf = []
datadf.append(ESKAPE_tall.groupby(['Location','name']).mean().reset_index())
datadf.append(ESKAPE_tall.groupby(['Location','name']).median().reset_index())

sns.set(font_scale=1, style='ticks')
fig = plt.figure()
count = 1
sns.set_style("ticks")
for data in datadf:
    ax = fig.add_subplot(1,2,count)
    sns.stripplot(data=data,y='name', x='value',hue='Location', order=ESKAPEE, jitter=False, size=8)
    if count==1:
        ax.legend().set_visible(False)
        ax.set_title('mean')
        ax.set_yticklabels(labels=ax.get_yticklabels(),fontstyle='italic')
    if count==2:
        ax.set(yticklabels=[])
        ax.set_title('median')
    ax.set_xlabel('Relative abundance')
    ax.set_ylabel('')

    ax.set_xscale("log",base=10)
    #x_major = LogLocator(base = 10, subs=[2,4,6,8])
    #ax.xaxis.set_major_locator(x_major)
    #ax.locator_params(axis='x',tight=True, nbins=4)

    #ax.xaxis.set_major_formatter(xScalarFormatter)
    count += 1

#fig.savefig(f'./fig/{scriptname}_ESKAPE_stripplot_log10_CTAB.pdf')
fig.savefig(f'./fig/{scriptname}_ESKAPE_stripplot_log10.pdf')


# Just plotting the median
sns.set(font_scale=1, style='ticks')
fig = plt.figure(figsize=(5,5))
sns.set_style("ticks")
ax = fig.add_subplot(111)
sns.stripplot(data=datadf[1],y='name', x='value',hue='Location', order=ESKAPEE, jitter=False, size=12)
#ax.set_title('median')
ax.set_xlabel('Relative abundance')
ax.set_ylabel('')
ax.set_xscale("log",base=10)
fig.savefig(f'./fig/{scriptname}_ESKAPE_stripplot_median_log10.pdf')

## Calculate p-value
for i, j in ESKAPE_tall.groupby('name'):
    try:
        a = j.groupby('Location').get_group('hawker')['value']
        b = j.groupby('Location').get_group('subway')['value']
        pval = ranksums(a,b, alternative='greater').pvalue
        print(f'{i}, {pval}')
    except:
        continue


# p-value for aggregated ESKAPE
agg_ESKAPE = ESKAPE_tall.groupby(['SAM id','Location'])['value'].sum()
aaa = agg_ESKAPE.groupby('Location').get_group('hawker')
aab = agg_ESKAPE.groupby('Location').get_group('hospital')
ranksums(aaa, aab, alternative='greater').pvalue
