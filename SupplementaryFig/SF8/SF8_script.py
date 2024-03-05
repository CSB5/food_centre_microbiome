# %%
%load_ext autoreload
%autoreload 2
from scipy.stats import wilcoxon

import pandas as pd
from itertools import combinations_with_replacement
from scipy.spatial.distance import squareform, pdist
import scipy
from scipy.stats import linregress, pearsonr
import numpy as np
import seaborn as sns
sns.set(font_scale=1)
#sns.set_style('white')
sns.set_style("ticks")
#import skbio
from skbio.diversity import alpha
import matplotlib.pyplot as plt
from statannot import add_stat_annotation

# %% Load Data
all_countdf = pd.read_csv(f'SF8_countdf.csv')
# Load all_mddf after removing geographical content

# %% Calculate diversity
alphaname = ['shannon', 'simpson','chao1']

diversity = [all_countdf.top_OTU(500).apply(getattr(alpha,a), axis = 1).rename(a) for a in alphaname]
all_countdf[all_countdf<0.0005]=np.NaN
richness = (~all_countdf.isna()).sum(axis=1)
richness.name = 'richness'

alphaDF = mgp.countDataFrame(pd.concat(diversity, axis = 1))
alphaDF['richness'] = richness
alphaDF['Location'] = all_mddf['Location']
alphaDF['Location2'] = all_mddf['Location2']
alphaDF['Location_FC'] = all_mddf['Location_FC']
alphaDF['cohort'] = all_mddf['cohort']
alphaDF = alphaDF[alphaDF['Location']!='NControl']

locations = alphaDF['Location'].value_counts().index.sort_values()
alphaDF['Location'] = pd.Categorical(alphaDF['Location'],categories=locations)

# %% Alpha Diversity vs size of hawker centre vs market presence

alphaDF2 = alphaDF.copy()
stall_number = pd.read_excel(mgp.PDIR/'data/raw/Hawker Centre Locations.xlsx')
alphaDF2['Food Stalls'] = alphaDF2['Location'].map(stall_number.set_index('Location')['Food Stalls'])
alphaDF2['Market Stalls'] = alphaDF2['Location'].map(stall_number.set_index('Location')['Market Stalls'])
alphaDF2['Market Presence'] = 'W/O Market'
alphaDF2.loc[alphaDF2['Market Stalls']>0,'Market Presence'] = 'With Market'

alphaDF_hco = alphaDF2[alphaDF2['cohort']=='Hawker Centre']
alphaDF_hcn = alphaDF2[alphaDF2['cohort']=='Hawker Centre (New)']

# Shannon vs Presence or Absence of stalls with stats
dataf = alphaDF_hco
fig = plt.figure(figsize=(2,5))
ax = fig.add_subplot(111)
sns.boxplot(x='Market Presence',y='shannon', data=dataf, ax=ax)
pairs = [('W/O Market','With Market')]
add_stat_annotation(ax, data=dataf, x='Market Presence',y='shannon', box_pairs=pairs, test='Mann-Whitney', text_format='star', loc='inside')
ax.set_xticklabels(['W/O Mkt','With Mkt'])
ax.set_xlabel('')
ax.set_ylabel('Score')
ax.set_title('Alpha Diversity Score')

fig.savefig(f'./fig/{scriptname}_market_presence.pdf')

#===== Linear Regression
import statsmodels.api as sm
# Note the difference in argument order
X = alphaDF_hco['Food Stalls'] ## X usually means our input variables (or independent variables)
y = alphaDF_hco['shannon'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

model = sm.OLS(y[:-2], X[:-2]).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
# Shannon vs Food stalls

#add regression equation to plot
ax = sns.regplot(x='Food Stalls', y='shannon', data=alphaDF_hco, x_jitter=0.5)
slope, intercept, r, p, sterr = linregress(x=ax.get_lines()[0].get_xdata(), y=ax.get_lines()[0].get_ydata())
coeff, pval = scipy.stats.pearsonr(alphaDF_hco['Food Stalls'], alphaDF_hco['shannon'], alternative='two-sided')
ax.text(0.03, 0.98, f'y = {intercept:2.2f} + {slope:2.5f}x\ncoeff = {coeff:2.2f}, p-value = {pval:2.2f}', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.get_figure().savefig(f'./fig/{scriptname}_food_stalls.pdf')

# Shannon vs Market stalls
dataf = alphaDF_hco
ax = sns.regplot(x='Market Stalls', y='shannon', data=dataf, x_jitter=0.5)
slope, intercept, r, p, sterr = linregress(x=ax.get_lines()[0].get_xdata(), y=ax.get_lines()[0].get_ydata())
coeff, pval = scipy.stats.pearsonr(dataf['Market Stalls'], dataf['shannon'], alternative='two-sided')
ax.text(0.03, 0.98, f'y = {intercept:2.2f} + {slope:2.4f}x\ncoeff = {coeff:2.2f}, p-value = {pval:2.2f}', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.get_figure().savefig(f'./fig/{scriptname}_market_stalls.pdf')
