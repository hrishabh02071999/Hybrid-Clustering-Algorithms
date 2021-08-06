import math
import random 
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import h5py
from sklearn.datasets.samples_generator import make_circles
from sklearn import metrics


train_data = pd.read_csv("http://cs.joensuu.fi/sipu/datasets/iris.data.txt")
train_set = train_data.values
X=train_set[:,0:-1]

def calc_cost(X,W,centroids):
    cost =0
    for i in range(X.shape[0]):
        for j in range(centroids.shape[0]):
            dist = np.sum((X[i,:]-centroids[j,:])**2)
            cost+=(W[i,j])*(dist*2)
    return cost 

def calc_centroids(X,W,p):
    #calculate centroids 
    centroids = np.divide(np.dot(X.T, W**p), np.sum(W, axis=0)).T
    return centroids        

def update_weights(X,centroids,p):
    W= np.zeros((X.shape[0],centroids.shape[0]))
    m = float(1/(p-1))
    for i in range(X.shape[0]):
        for j in range(centroids.shape[0]):
            num= np.sum((X[i,:]-centroids[j,:].reshape(1,X.shape[1]))**2,axis=1)
            den=0
            for k in range(centroids.shape[0]):
                dist = np.sum((X[i,:]-centroids[k,:].reshape(1,X.shape[1]))**2,axis=1)
                den+=  math.pow((num/dist),m)
            W[i][j]=1/(float(den))
    return W        

def get_clusters(X,W,k):
    cluster_indices = np.argmax(W,axis=1) #getting the cluster indices as the ones with highest membership values
    clusters={}
    for i in range(k):
        clusters[i]=[]
    for i in range(X.shape[0]):
        clusters[cluster_indices[i]].append(i)
    return clusters
  

def fuzzy_cmeans(X,k,p,num_iter):
    #initialize fuzzy matrix
    W=np.random.randint(low=1, high=100, size=(X.shape[0],k))
    #make sum of every row =1
    W= W/(np.sum(W,axis=0))
    for i in range(num_iter):
        #calculate centroids
        centroids = calc_centroids(X,W,p)
        cost = calc_cost(X,np.power(W,p),centroids)/X.shape[0]
        W= update_weights(X,centroids,p)
    clusters = get_clusters(X,W,k)#dictionary containing list of data point indices belonging to a cluster
    return clusters
start=time.time()
clusters = fuzzy_cmeans(X,3,1.16,20)
label=[]
for i in range(len(X)):
  label.append(0)
for i in clusters[1]:
  label[i]=1
for i in clusters[2]:
  label[i]=2
plt.figure()
plt.scatter(X[clusters[0],2], X[clusters[0],3],color='r')
plt.scatter(X[clusters[1],2], X[clusters[1],3],color='g')
plt.scatter(X[clusters[2],2], X[clusters[2],3],color='b')

print("Silhoutte :- ")
print(metrics.silhouette_score(X,label))
print("Time taken by my implementation :- ",time.time()-start)
