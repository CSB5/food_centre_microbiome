# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance  as ssd
from sklearn import decomposition, manifold, preprocessing
from clusim.clustering import Clustering, print_clustering
import clusim.sim as sim
from sklearn.model_selection import train_test_split, StratifiedKFold

# %% Load Data
counts_noNC_filt = pd.read_csv(f'./SF9_counts.csv')
# Load mddf_noNC_filt after removing sensitive geographical information

# %% Similarity calculation
similiarity_score_ls = []
skf = StratifiedKFold(n_splits=10)
X,y = counts_noNC_filt,mddf_noNC_filt['Location_FC2']
for train_index, test_index in skf.split(X, y):
    c = X.iloc[train_index]
    l = y.iloc[train_index]
    microbe_df = c.get_microbes().relabd()
    #food_df = c.get_plants_animals().relabd()
    food_df = get_food(c).relabd()
    #food_df.loc[['WEM430','WEM432']] has no food DNA O.o
    mcZ = sch.linkage(ssd.pdist(microbe_df, metric = mgpu.spearman_distance), method='ward')
    mcML = sch.fcluster(mcZ,16,criterion='maxclust') #microbial count Membership List
    mcML_rand = np.random.permutation(mcML)
    food_df.sum(axis=1)[(food_df.sum(axis=1)<0.5)]
    fcZ = sch.linkage(ssd.pdist(food_df, metric = mgpu.spearman_distance), method='ward')
    fcML = sch.fcluster(fcZ,16,criterion='maxclust')
    fcML_rand = np.random.permutation(fcML)
    lML = l
    lML_rand = np.random.permutation(lML)
    sim_score = lambda A,B: sim.element_sim(Clustering().from_membership_list(A),Clustering().from_membership_list(B))

    similiarity_score_ls.append([sim_score(mcML, fcML),'microbe-food','true'])
    similiarity_score_ls.append([sim_score(fcML, lML),'food-location','true'])
    similiarity_score_ls.append([sim_score(mcML, lML),'microbe-location','true'])
    similiarity_score_ls.append([sim_score(mcML_rand, fcML_rand),'microbe-food','null'])
    similiarity_score_ls.append([sim_score(fcML_rand, lML_rand),'food-location','null'])
    similiarity_score_ls.append([sim_score(mcML_rand, lML_rand),'microbe-location','null'])


sscore_df = pd.DataFrame(similiarity_score_ls, columns = ['score','comparison','model'])
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
sns.boxplot(data=sscore_df,y='score',x='comparison',hue='model', ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Similarity scores")
