#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import datasets
#%%
class GMMClassifier:
    def __init__(self, num_clusters = 4):
        self.num_clusters = num_clusters
        self.sigma = None 

    def fit(self, X):
        self.sigma = np.array([np.eye(X.shape[1]) for i in range(self.num_clusters)]);#np.concatenate([np.eye(X.shape[1]).reshape(X.shape[1],X.shape[1],1) for i in range(self.num_clusters)],axis=2)
        self.mu = np.random.random((self.num_clusters,X.shape[1]))
        self.pi = np.full((self.num_clusters,1),1/self.num_clusters)
        self.membership_weights = np.zeros((X.shape[0],self.num_clusters))
        
        return self.expectation(X,self.num_clusters,self.pi,self.mu,self.sigma)

    def expectation(self,X,k,pi,mu,sigma):
        # First consider a single sample (nth). For that sample the gamma_nk is obtained using Bayes
        # the ratio of the product of the prior (Prob(cluster == k)) and the 
        # likelihood (Prob(observe xn | cluster == k)) to the total probability. 

        # Here we are vectorizing it for the whole dataset and calculating the gamma_k for the entire dataset 
        
        bayes_numerator = np.empty((X.shape[0],0))
        total_prob = np.zeros((X.shape[0],1))
        for cluster in range(k):
            # this gamma should be X.shape[0] x 1 
            gamma_k = pi[cluster] * stats.multivariate_normal.pdf(X,mu[cluster],sigma[cluster]).reshape(-1,1)
            # we column stack it 
            bayes_numerator = np.column_stack((bayes_numerator,gamma_k))
            # add it to the total probability accumulator
            total_prob += gamma_k
        # here bayes_numerator should be X.shape[0] x k, and total prob should be X.shape[0] x 1
        return bayes_numerator / total_prob
           
    def maximize_mean(self,X,k,w):
        pass
# %%
# %%
iris = datasets.load_iris()
X = iris.data
clf = GMMClassifier()
exp = clf.fit(X)
pass
# %%
