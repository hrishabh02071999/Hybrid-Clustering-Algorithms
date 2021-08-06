import random #generate random nos
import numpy as np
import pandas as pd
from sklearn import metrics
import time
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.patches as mpatches
import sklearn.metrics as sm
%matplotlib inline

#loading iris datasets
iris=datasets.load_iris()
print(iris)
iris.target

X=pd.DataFrame(iris.data,columns=['Sepal L','Sepal W','Petal L','Petal W'])
Y1=pd.DataFrame(iris.target,columns=['Target'])
print(X.describe())
print('SHAPE IS=', X.shape)
X.head()

#Plotting unclustered data

#plt.scatter(X['Petal L'],X['Petal W'],marker='.',c='r')
#plt.title('Petal length vs Petal width')

#plt.scatter(X['Sepal L'],X['Sepal W'],marker='.',c='g')
#plt.title('Sepal length vs Sepal width')

#X2 saves the petal length and width array of all values as a matrix of size 150*2
X2=X.iloc[:,[2,3]].values
print(X2.shape)


start=time.time()
print(start)

m=X2.shape[0]   # No. of rows=150
n=X2.shape[1]   # No. of columns=2
m,n
n_iter=50

K=3   #No. of clusters


#Centroids is a n x K dimentional matrix, where each column will be a centroid for one cluster.
#Step 1:Initialize the centroids randomly from the data points:

Centroids=np.array([]).reshape(n,0)   

for i in range(K):
    rand=random.randint(0,m-1)
    Centroids=np.c_[Centroids,X2[rand]]
    
#Step 2:a

Output={}   #The output of our algorithm should be a dictionary with 
            #cluster number as Keys and the data points which belong 
            #to that cluster as values. So let’s initialize the dictionary.

        
#We find the euclidian distance from each point to all the centroids and store in a m X K matrix. 
#So every row in EuclidianDistance matrix will have distances of that particular data point from all the centroids.
        
EuclidianDistance=np.array([]).reshape(m,0)
for k in range(K):
       tempDist=np.sum((X2-Centroids[:,k])**2,axis=1)
       EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    
#Next, we shall find the minimum distance and store the index of the column in a vector C.
C=np.argmin(EuclidianDistance,axis=1)+1     #Returns the index of the minimum values along an axis.



#b-> We need to regroup the data points based on the cluster index C and store in the Output dictionary and 
#    also compute the mean of separated clusters and assign it as new centroids.
#    Y is a temporary dictionary which stores the solution for one particular iteration

Y={}
for k in range(K):
    Y[k+1]=np.array([]).reshape(2,0)
for i in range(m):
    Y[C[i]]=np.c_[Y[C[i]],X2[i]]
     
for k in range(K):
    Y[k+1]=Y[k+1].T
    
for k in range(K):
     Centroids[:,k]=np.mean(Y[k+1],axis=0)

        
#Now we need to repeat step 2 till convergence is achieved.
#In other words, we loop over n_iter and repeat the step 2.a and 2.b as shown:
for i in range(n_iter):
     #step 2.a
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X2-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X2[i]]
     
      for k in range(K):
          Y[k+1]=Y[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y
    
    
color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

X_DATA=X.iloc[:,[2,3]].values
print("Silhoutte :- ")
print(metrics.silhouette_score(X_DATA,C))
print("Time taken by KMEANS implementation :- ",time.time()-start)
