%load_ext autoreload
%autoreload 2
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.expanduser('~/GIS/nea'))
from src import *
sys.path.insert(0, os.path.expanduser('~/GIS/code'))
import metaGPy as mgp
import metaGPy.CoDA as mgpc
import metaGPy.utilities as mgpu
import metaGPy.normalization as mgpn
import logging
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

from itertools import combinations_with_replacement, combinations
from scipy.spatial.distance import squareform, pdist
import scipy.cluster.hierarchy as sch
from scipy.stats import gmean
import scipy.cluster.hierarchy as sch
from scipy.stats import binomtest

from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn import decomposition

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# %% Load Data
eukCounts = pd.read_csv('./2B_eukcounts.csv')
microbialCounts = pd.read_csv('./2B_microbialcounts.csv')
countdf = pd.read_csv('./2B_allcounts.csv')

# %% Create training and testing set
def train_test(datadf):
    data_split = StratifiedKFold(n_splits=4, shuffle=True, random_state=17293)
    for train_index, test_index in data_split.split(datadf, groups):
        X_train = datadf.iloc[train_index,:]
        X_test = datadf.iloc[test_index,:]
        y_train = [groups[i] for i in train_index]
        y_test = [groups[i] for i in test_index]
        break
    return X_train, X_test, y_train, y_test

# %% Transform data
## PCA
def trans_PCA(X_train, X_test):
    pca_analysis = decomposition.PCA()
    pca_analysis.fit(X_train)
    X_train_pca = pca_analysis.transform(X_train)
    X_test_pca = pca_analysis.transform(X_test)
    return X_train_pca, X_test_pca

## CSS
def trans_CSS(X_train, X_test):
    return mgpn.CSS(X_train), mgpn.CSS(X_test)

## TSS
def trans_TSS(X_train, X_test):
    return X_train.relabd(), X_test.relabd()

## CLR
def trans_clr(X_train, X_test):
    return mgpc.clr(X_train), mgpc.clr(X_test)

## PCA-CLR
def trans_pca_clr(X_train, X_test):
    X_train_clr, X_test_clr = trans_clr(X_train, X_test)
    a,b = trans_PCA(X_train_clr, X_test_clr)
    return a,b

## ILR
def variation_matrix(data):
    """
    Calculate variation matrix to find principle balance
    data (pandas.DataFrame or np.array) where columns are variables and rows are samples.
    """
    variation_func = lambda x,y : np.var(np.log(x/y))
    return pdist(data.T,variation_func)

def anti_variation_matrix(data):
    """
    Calculate variation matrix to find principle balance
    data (pandas.DataFrame or np.array) where columns are variables and rows are samples.
    """
    anti_variation_func = lambda x,y : np.var(np.log(x/y))
    varmat = pdist(data.T, anti_variation_func)
    anti_varmat = varmat.max()-varmat
    return anti_varmat

def distmat2sbp(distmat,no_plot=True):
    """
    Convert distance matrix into an SBP dendogram and SBP matrix
    """
    dim = squareform(distmat).shape[0]
    dend = sch.dendrogram(sch.linkage(distmat, method='ward'), no_plot=no_plot)
    distance = np.array(dend['dcoord'])[:,1]
    distance[::-1].sort()
    delta = np.min(np.abs(np.diff(distance)))*0.01
    # Get Clusters
    clusters = [np.ones(dim)]
    for ind,i in enumerate(distance):
        clusters.append(sch.fcluster(sch.linkage(distmat, method='ward'),i-delta,criterion='distance'))

    sbp = []
    for ind, i in enumerate(clusters[1:]):
        clustermax = np.max(clusters[ind])
        for j in range(1,int(clustermax)+1):
            clusterind = np.where(clusters[ind]==j)
            if len(i[clusterind]) > 1:
                if abs(np.diff(i[clusterind])).sum() > 0:
                    clusternums = np.unique(i[clusterind])
                    sbp_array = np.zeros(dim)
                    sbp_array[np.where(i==clusternums[0])] = 1
                    sbp_array[np.where(i==clusternums[1])] = -1
                    sbp.append(sbp_array)

    return sbp
def partitionvec2logbasis(partitionvec):
    """
    Generate log basis vector from one binary partition
    """
    r = (partitionvec==-1).sum()
    s = (partitionvec==1).sum()
    a = (s/(r*(r+s)))**0.5
    b = -(r/(s*(s+r)))**0.5
    v = np.zeros(partitionvec.shape)
    v[partitionvec==-1] = a
    v[partitionvec==1] = b
    return v

def sbp2projmat(sbp):
    """
    Generate Projection Matrix from sbp
    """
    projmat=[]
    for i in sbp:
        v = partitionvec2logbasis(i)
        #be = np.exp(v) #Balance element
        #e = np.log(be/gmean(be))
        #projmat.append(e)
        projmat.append(v)

    return np.array(projmat)

def ilr(dist_mat, X_train_clr, X_test_clr):
    sbp = distmat2sbp(dist_mat)
    projmat = sbp2projmat(sbp)
    X_train_ilr = X_train_clr.dot(projmat.T)
    X_test_ilr = X_test_clr.dot(projmat.T)
    return X_train_ilr, X_test_ilr, sbp

def trans_ilr(X_train, X_test):
    ## ILR
    # Distance Matrices
    dist_var = variation_matrix(X_train_raw)
    dist_antivar = anti_variation_matrix(X_train_raw)
    dist = [dist_var, dist_antivar]
    X_train_clr, X_test_clr = trans_clr(X_train, X_test)
    X_train_ilr, X_test_ilr, sbp = ilr(dist_antivar, X_train_clr, X_test_clr)
    return X_train_ilr, X_test_ilr

# %% Classifiers
classifiers_list = [
    LogisticRegression(max_iter=10000),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

classifiers_name = ['Logistic Regression','KNeighborsClassifier','SVC (Linear)','SVC (RBF)','Gaussian Process','Decision Tree','Random Forest','MLPClassifier','AdaBoostClassifier','GaussianNB','QuadraticDiscriminantAnalysis']

# %% Classify
def classify(X_train,y_train,X_test,Y_test):
    classifierscore_list = []
    for classifier_name, classifier in zip(classifiers_name,classifiers_list):
        #classifier = LogisticRegression(max_iter=10000)
        cv = StratifiedKFold(n_splits=5)
        train_auc = []
        train_score = []
        test_auc = []
        test_score = []
        for cv_ind, (cv_train_ind, cv_test_ind) in enumerate(cv.split(X_train, y_train)):
            cv_x_train = np.array(X_train)[cv_train_ind]
            cv_y_train = np.array(y_train)[cv_train_ind]
            classifier.fit(cv_x_train, cv_y_train)
            #======== Training set
            y_train_prob = classifier.predict_proba(cv_x_train)
            macro_roc_auc_ovr_train = metrics.roc_auc_score(cv_y_train, y_train_prob, multi_class="ovr", average="macro")
            train_auc.append(macro_roc_auc_ovr_train)
            train_score.append(classifier.score(cv_x_train,cv_y_train))
            #======== Testing set
            y_test_prob = classifier.predict_proba(X_test)
            macro_roc_auc_ovr_test = metrics.roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro")
            test_score.append(classifier.score(X_test, y_test))
            test_auc.append(macro_roc_auc_ovr_test)
        classifierscore_df = pd.DataFrame({'auc_train':train_auc, 'acc_train':train_score, 'auc_test':test_auc, 'acc_test':test_score})
        classifierscore_df['classifier'] = classifier_name
        classifierscore_list.append(classifierscore_df)
    classifierscore_agg_df = pd.concat(classifierscore_list, axis=0)
    return classifierscore_agg_df

# %% Iterate over dataset, classifier and transformation
#taxa = ['S','G','F']
counts = [countdf, eukCounts, microbialCounts]
counts_name = ['all', 'non_microbial', 'microbial']
transformations = [trans_TSS, trans_CSS, trans_clr, trans_ilr]
transformations_name = ['TSS','CSS','CLR', 'ILR']
scoredf_list = []
for count_name, count in zip(counts_name, counts):
    #count = count.top_OTU(500)
    count = count.relabd()
    try:
        count = count.drop(columns=['others'])
    except:
        pass
    X_train_raw, X_test_raw, y_train, y_test = train_test(count)
    for trans_name, trans in zip(transformations_name, transformations):
        X_train, X_test = trans(X_train_raw, X_test_raw)
        scoredf = classify(X_train,y_train,X_test,y_test)
        scoredf['read_type'] = count_name
        scoredf['transformation'] = trans_name
        scoredf_list.append(scoredf)
        print(f"{count_name} {trans_name}")

score_aggdf = pd.concat(scoredf_list, axis=0)
score_aggdf.to_csv('../manuscript/Fig2B/2B_aggdf.csv')

# %% Plot Performance
classifier_names2 = ['Logistic Regression', 'MLPClassifier', 'SVC (Linear)', 'KNeighborsClassifier', 'GaussianNB', 'Random Forest', 'Decision Tree','AdaBoostClassifier', 'QuadraticDiscriminantAnalysis', 'Gaussian Process', 'SVC (RBF)']
classifier_names3 = ['LogReg', 'MLPC', 'SVC-L', 'KNC', 'GaussNB', 'RForest', 'DTree','AdaBC', 'QDA', 'GaussP', 'SVC-RBF']
fig = plt.figure(figsize=(7,4))
count = 1
for ind, count_name in enumerate(counts_name):
    scoredf_filtered = score_aggdf[score_aggdf['read_type']==count_name]
    scoredf_filtered_agg = scoredf_filtered.groupby(['classifier','read_type','transformation']).mean().reset_index()
    scoredf2D = scoredf_filtered_agg.pivot(index='classifier',columns='transformation',values='auc_test')
    scoredf2D = scoredf2D[['ILR','CLR','CSS','TSS']].reindex(classifier_names2)
    scoredf2D.index.to_list()
    ax = fig.add_subplot(1,3,count)
    if count==3:
        sns.heatmap(scoredf2D,cmap='Reds',ax=ax, vmax=1, vmin=0.6, cbar_kws={'label': 'AUC-ROC'})
    else:
        sns.heatmap(scoredf2D,cmap='Reds',cbar=None,ax=ax, vmax=1, vmin=0.6)
    ax.set_title(count_name)
    if count!=2:
        ax.set_xlabel("")
    if count==1:
        ax.set(yticklabels=classifier_names3)
    if count!=1:
        ax.set(yticklabels=[], ylabel='')
    count+=1
