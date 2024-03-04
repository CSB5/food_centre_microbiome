%load_ext autoreload
%autoreload 2
from pathlib import Path
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from scipy.spatial.distance import squareform, pdist
from scipy.stats import binomtest
from sklearn import metrics
from sklearn import decomposition
import numpy as np
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dataset = f'./2D_dataset.joblib'
X,Y,mddf = load(dataset) # Selected top 200 based on gmean with pseudo count added
counts = X


# %% Cross validation based on ilr_rfe_classifier
cv = mgpc.CV(classifier = mgpc.ilr_rfe_classifier, n_splits=4, variance_threshold=0.75)
cv.fit(X, Y)
cv.training_score_list
cv.testing_score_list

# mean features, balances, accuracy, AUCROC, AP
np.mean([len(c.get_features()) for c in cv.classifier_list])
np.mean([len(c.get_balances()) for c in cv.classifier_list])
np.mean([s[-1] for s in cv.testing_score_list], axis=0)

species = cv.overlapping_features()
species.to_csv(f"./2D_ilr_rfe_iterative_species.csv")
