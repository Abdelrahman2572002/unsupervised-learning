import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
 
data= loadmat('C:\\abdelrahman\\ex7data2.mat')
print(data)
print(data['X'])
print(data['X'].shape)
def init_centroids(x,k):
    m,n=x.shape
    centroids= np.zeros((k,n))
    idx= np.random.randint(0,m,k)
    for i in range(k):
        centroids[i,:]=x[idx[i],:]
         
    return centroids
def find_closest_centroids(x,centroids):
    m=x.shape[0]
    k=centroids.shape[0]
    idx=np.zeros(m)
    
    for i in range(m):
        min_dist=1000000
        for j in range(k):
            dist=np.sum((x[i,:]-centroids[j,:]) **2)
            if dist<min_dist:
                min_dist=dist
                idx[i]=j
                
    return idx
def compute_centroids(x,idx,k):
    m,n=x.shape
    centroids=np.zeros((k,n))
    for i in range(k):
        indices=np.where(idx==i)
        centroids[i,:]=(np.sum(x[indices,:],axis=1) / len(indices[0])).ravel()
        
    return centroids

def runk_means(x,initial_centroids,iters):
    m,n=x.shape
    k=initial_centroids.shape[0]
    idx=np.zeros(m)
    centroids=initial_centroids
    for i in range(iters):
        idx=find_closest_centroids(x, centroids)
        centroids= compute_centroids(x, idx, k)
        
    return idx,centroids
x=data['X']
initial_centroids= init_centroids(x,3)
idx= find_closest_centroids(x,initial_centroids)
c= compute_centroids(x,idx,3)
for a in range(6):
    idx,centroids= runk_means(x,initial_centroids,a)
    
    
    
    
    cluster1= x[np.where(idx==0)[0],:]
    cluster2= x[np.where(idx==1)[0],:]
    cluster3= x[np.where(idx==2)[0],:]
    fig, ax = plt.subplots(figsize=(9,6))
    ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
    ax.scatter(centroids[0,0],centroids[0,1],s=300, color='r')
    ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
    ax.scatter(centroids[1,0],centroids[1,1],s=300, color='g')
    ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
    ax.scatter(centroids[2,0],centroids[2,1],s=300, color='b')
    ax.legend()
    


    
    

