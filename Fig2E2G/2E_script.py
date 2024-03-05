# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
import pandas as pd
import metaGPy.CoDA as mgpc
import metaGPy as mgp
import sys
import os
from pathlib import Path
from scipy.stats import binomtest, ranksums


# %% Load Data
# Load old counts
dataset = f'../Fig2D/2D_dataset.joblib'
X, Y = load(dataset) # Need to load mddf
old_counts, old_mddf = X.copy(), mddf.copy()
old_species = old_counts.columns
# Load new counts
new_dataset = './2E_new_dataset.joblib'
new_counts = load(new_dataset) # Need to load new_mddf

# %% Train old, test new
# Train classifier / Load classifier
############## old ilr-rfe performance on new swabs ################
RFECV_output_filename = f'../Fig2D/2D_ilr_rfe_iterative_output.joblib'
cv = load(RFECV_output_filename)

i=0
aucroc_score_ls = []
ap_score_ls = []
for i in range(5):
    output_filename = f'../Fig2D/2D_ilr_rfe_iterative_output_random_state_{i}.joblib'
    cv_null = load(output_filename)
    agg_score_null = cv_null.scores_list(X_new, Y_new, agg=True, roc_average='micro')
    class_scores_null = cv_null.scores_list(X_new, Y_new, agg=False)
    classes_null = cv_null.classifier_list[0].rfe_list[0].classes_
    aucroc_score_ls.append(np.array([i[-1][1] for i in class_scores_null]).flatten())
    ap_score_ls.append(np.array([i[-1][0] for i in class_scores_null]).flatten())

aucroc_score_null = np.array(aucroc_score_ls).flatten()
ap_score_null = np.array(ap_score_ls).flatten()

###### Process Scores and P-value
classes_converted = [loc_dict[i] for i in classes]
class_aucroc_n20 = pd.DataFrame([i[-1][1] for i in class_scores], columns=classes_converted)
class_aucroc_n20.mean().mean()
class_ap_n20 = pd.DataFrame([i[-1][2] for i in class_scores], columns=classes_converted)
class_ap_n20.mean().mean()

aucroc_p_val_ls = []
for i in class_aucroc_n20.T.iterrows():
    p_val = ranksums(i[1],aucroc_score_null, alternative='greater').pvalue
    aucroc_p_val_ls.append(p_val)
ap_p_val_ls = []
for i in class_ap_n20.T.iterrows():
    p_val = ranksums(i[1],ap_score_null, alternative='greater').pvalue
    ap_p_val_ls.append(p_val)

pval_df = pd.DataFrame({'aucroc':aucroc_p_val_ls,'ap':ap_p_val_ls},index=class_aucroc_n20.columns)
pval_df_annot = pval_df.copy()
pval_df_annot[pval_df>0.0000] = ''
pval_df_annot[pval_df<0.0001]='****'
pval_df_annot[(pval_df<0.001) & (pval_df>0.0001)]='***'
pval_df_annot[(pval_df<0.01) & (pval_df>0.001)]='**'
pval_df_annot[(pval_df<=0.05) & (pval_df>0.01)]='*'

######
fig = plt.figure(figsize=(8, 2.5))
ax = fig.add_subplot(211)
sns.barplot(class_aucroc_n20.melt(), x='variable',
            y='value', ax=ax, palette='tab20')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#ax.set_xlabel('Locations')
ax.set_xlabel(None)
ax.set_ylabel('AUC-ROC')
ax.set(xticklabels=[])
#ax.set_title('AUC-ROC of new swabs, 4-fold cross validation')
ax.set_ylim((0, 1))
for i,p in zip(ax.containers,pval_df_annot['aucroc']):
    ax.bar_label(i, fmt=p, label_type='edge')


ax = fig.add_subplot(212)
sns.barplot(class_ap_n20.melt(), x='variable',
            y='value',hue='variable', ax=ax, palette='tab20')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Average Precision')
for i,p in zip(ax.containers,pval_df_annot['ap']):
    ax.bar_label(i, fmt=p, label_type='edge')
fig

# %% Train old + 1 new, test other new
# Generate training dataset
counts = pd.concat([old_counts, new_counts], axis=0)
mddf = pd.concat([old_mddf, new_mddf], axis=0)
X_old = counts[mddf['Collection Round'] == 1]
X_new = counts[mddf['Collection Round'] == 2]
Y_old = mddf.loc[mddf['Collection Round'] == 1, 'Location']
Y_new = mddf.loc[mddf['Collection Round'] == 2, 'Location']

cv_list = []
for i in range(5):
    X_new_data = mgpc.train_test(X_new, Y_new, n_splits=5)
    X_4, X_1, Y_4, Y_1 = X_new_data[i]
    X_add = X_1.include(counts.columns).relabd().drop(columns="others")
    Y_add = Y_1
    X_old_clr = mgpc.clr(X_old)
    X_add_clr = mgpc.clr(X_add)
    cv = mgpc.CV(classifier=mgpc.rfe_classifier, n_splits=4)
    cv.fit(X_old_clr, Y_old, X_add=X_add_clr, Y_add=Y_add)
    cv_list.append(cv)

output_filename = f'./output/combine_new_old_old+somenew_output.joblib'
#dump(cv_list, output_filename)
cv_list = load(output_filename)

X_new_data = mgpc.train_test(X_new, Y_new, n_splits=5)
#X_4, X_1, Y_4, Y_1 = X_new_data[i]
#cv.scores_list(mgpc.clr(X_4), Y_4, agg=False)
agg_aucroc_ls = []
for cv, X_new_data in zip(cv_list, mgpc.train_test(X_new, Y_new, n_splits=5)):
    X_4, X_1, Y_4, Y_1 = X_new_data
    class_scores = cv.scores_list(mgpc.clr(X_4), Y_4, agg=False)
    num_features = [i.n_features_ for i in cv.classifier_list[0].rfe_list]
    agg_aucroc = pd.DataFrame([[np.mean(j[1]) for j in i] for i in class_scores], columns=num_features)
    agg_aucroc_ls.append(agg_aucroc)

agg_aucroc = pd.concat(agg_aucroc_ls, axis=0).reset_index(drop=True)


#=========== Old test new =========
output_filename = f'./output/iterate_ilr_rfe_rfe_(clr)_output.joblib'
cv = load(output_filename)

X_new = new_counts.reindex(columns=cv.X.columns)
Y_new = new_mddf['Location']

class_scores = cv.scores_list(mgpc.clr(X_new), Y_new, agg=False)
classes = cv.classifier_list[0].rfe_list[0].classes_
num_features = [i.n_features_ for i in cv.classifier_list[0].rfe_list]

## AUCROC
# All Classes
agg_aucroc_oldnew = pd.DataFrame(
    [[np.mean(j[1]) for j in i] for i in class_scores], columns=num_features)

## p-value
### compare differences for 20 featuresO
ranksums(agg_aucroc_oldnew,agg_aucroc, alternative='less').pvalue


# Some Classes (Ignore)
# def get_subclass(score):
#     return [score[i] for i in [3, 4, 7, 8, 9, 10, 11]]
# agg_aucroc_oldnew_top = pd.DataFrame(
#     [[np.mean(get_subclass(j[1])) for j in i] for i in class_scores], columns=features)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.lineplot(agg_aucroc_oldnew.melt(), x='variable', y='value', ax=ax)
#sns.lineplot(agg_aucroc_oldnew_top.melt(),x='variable',y='value', ax=ax)
sns.lineplot(agg_aucroc.melt(), x='variable', y='value', ax=ax)
ax.legend(['Original model', '', 'Fine-tuned model'])
ax.set_xlabel("Number of features")
ax.set_ylabel("AUC-ROC score")


#fig.savefig(f'./fig/{scriptname}_new_swab_aucroc_vs_features_3_types.pdf')
fig.savefig(f'./fig/{scriptname}_new_swab_aucroc_vs_features_2_types.pdf')
