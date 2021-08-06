# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import os
import math
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA





def MyDBSCAN(D, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue

        NeighborPts = regionQuery(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else:
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) > MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1


def regionQuery(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        if abs(math.sqrt((D[P][0]-D[Pn][0])*(D[P][0]-D[Pn][0])+(D[P][1]-D[Pn][1])*(D[P][1]-D[Pn][1]))) < eps:
            neighbors.append(Pn)
    return neighbors








X = pd.read_csv('/content/CC GENERAL.csv')
X = X.drop('CUST_ID', axis = 1)
# Handling the missing values
X.fillna(method ='ffill', inplace = True)
# Scaling the data to bring all the attributes to a comparable level
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizing the data so that
# the data approximately follows a Gaussian distribution
X_normalized = normalize(X_scaled)

# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_data=[]
#print(X_principal)
for i in range(len(X_principal)):
    temp=[]
    temp.append(X_principal[0][i])
    temp.append(X_principal[1][i])
    X_data.append(temp)
X_data=X_data[0:1000]
X_data=numpy.array(X_data)
#print(X_data)







###############################################################################
# My implementation of DBSCAN
# Run my DBSCAN implementation.
print('Running my implementation...')
start=time.time()
my_labels = MyDBSCAN(X_data, eps=0.3, MinPts=105)
#print(my_labels)
plt.figure()
plt.scatter(X_data[:, 0], X_data[:, 1], c=my_labels)
plt.savefig('dbscan.png')
plt.title('DBSCAN PLOT')
print("Time taken by my implementation :- ",time.time()-start)




##############################################################################

print('Runing scikit-learn implementation...')
start=time.time()
db = DBSCAN(eps=0.3, min_samples=105).fit(X_data)
skl_labels = db.labels_
# Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start
# numbering at 1, so increment the skl cluster numbers by 1.
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        skl_labels[i] += 1

#print(skl_labels)
plt.figure()
plt.scatter(X_data[:, 0], X_data[:, 1], c=skl_labels)
plt.savefig('dbscan.png')
plt.title('In-Built DBSCAN PLOT')
print("Time taken by In-built DBSCAN Algorithm :- ",time.time()-start)

print("Silhouette Score for In-Built DBSCAN")
ib_ss=metrics.silhouette_score(X_data, skl_labels)
print(ib_ss)
print("Silhouette Score for my Implementaion")
my_ss=metrics.silhouette_score(X_data, my_labels)
print(my_ss)
print("Difference Between Silhouette Score :- ",ib_ss-my_ss)
print("Adjusted Rand Index")
print(metrics.adjusted_rand_score(my_labels, skl_labels))

