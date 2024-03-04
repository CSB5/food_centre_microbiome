%load_ext autoreload
%autoreload 2
from pathlib import Path
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from scipy.stats import binomtest
from sklearn import metrics
from sklearn import decomposition
import numpy as np
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

#TODO extract code from mgpc

# %% Load Data
# Load Data
counts = pd.read_csv(f'./2C_counts.csv')
mddf = pd.read_csv(f'./2C_mddf.csv')
# Set up data
X = counts
Y = mddf['Location']
# Transform data
X_clr = mgpc.clr(X)
X_ilr = mgpc.ilr_transformer().fit(X).transform(X)
X_pca = pd.DataFrame(decomposition.PCA().fit(X).transform(X))
X_pca.index = X.index

# %% clr_rfe_iterative
accuracy_ls=[]
AUCROC_ls=[]
AP_ls=[]
score_ls = []
classifier_ls = []
nfeatures = np.array([2,4,6,8,10,13,16,20,25,30,35,40])
for X_input in [X_clr,X_ilr,X_pca]:
    cv = mgpc.CV(classifier = mgpc.rfe_classifier, n_splits=4, feature_range=nfeatures)
    cv.fit(X_input, Y)
    score_ls.append(cv.testing_score_list)
    classifier_ls.append(cv)

output_filename = f'./output/{scriptname}.joblib'
#dump((score_ls,classifier_ls), output_filename)
(score_ls,classifier_ls) = load(output_filename)

new_score_ls = []
for snd,score_metric in enumerate(['Accuracy','AUC-ROC','Average precision']):
    mean_score = pd.DataFrame([pd.DataFrame([np.array(score)[:,snd] for score in score_df]).mean() for score_df in score_ls]).T
    mean_score.index = nfeatures
    mean_score.columns = ['clr','ilr','pca']
    mean_score = pd.melt(mean_score, ignore_index=False, value_vars=['clr','ilr','pca']).reset_index()
    mean_score['metric'] = score_metric
    new_score_ls.append(mean_score)

scores = pd.concat(new_score_ls)
scores.columns = ['n_features','transformation','score','metric']

# %% Plot performance vs features selected
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)
sns.lineplot(data=scores,x='n_features',y='score',hue='transformation',style='metric', ax=ax)
ax.set_xlabel('No. of features')
