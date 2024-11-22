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
        # X is N x f
        # sigma is f x f x k (one fxf matrix for each of the k clusters)
        self.sigma = np.array([np.eye(X.shape[1]) for i in range(self.num_clusters)]);#np.concatenate([np.eye(X.shape[1]).reshape(X.shape[1],X.shape[1],1) for i in range(self.num_clusters)],axis=2)
        # mu is k x f
        self.mu = np.random.random((self.num_clusters,X.shape[1]))
        # pi is k x 1
        self.pi = np.full((self.num_clusters,1),1/self.num_clusters)
        # weights is N x k
        self.membership_weights = self.expectation(X,self.num_clusters,self.pi,self.mu,self.sigma)
        
        return self.maximize_mean(X,self.num_clusters,self.membership_weights)

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
        # w is N x k, so w_ik is a scalar
        # N_k = sum_over_N(weight_k) is scalar (Add weights for kth cluster over all samples) 
        # mu_k = sum_over_N(w_ik*x_i)/N_k, is 1xf
        # mu is k x f (one mu for each k)
        mu = np.empty((0,X.shape[1]))
        for cluster in range(k):
            N_k = np.sum(w[:,cluster],axis=0)
            mean_n = (np.dot(w[:,cluster].T,X)/N_k)
            
            mu = np.vstack((mu,np.expand_dims(mean_n,axis=0))) # 1xN x Nxf = 1xf
        return mu


# %%
# %%
iris = datasets.load_iris()
X = iris.data
clf = GMMClassifier()
exp = clf.fit(X)
pass
# %%
