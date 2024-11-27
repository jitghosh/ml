#%%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import datasets
#%%
class GaussianMixtureModel:
    def __init__(self, k = 3):
        '''
        Parameters:
            k: number of clusters we want to groupd the data into
            sigma: variable to store covariance matrices for the gaussians (one per cluster). k x f x f (k clusters , each matrix is f x f)
            mu: variable to store the mean vectors for the gaussians (one per cluster). k x f (k clusters, each mean vector is 1 x f)
            pi: variable to store the mixture weights vector (one per cluster). k x 1 - each cluster has a scalar mixture weight
            membership_weights: variable to store the membership weights matrix N x k - each sample has a weight for each cluster.
            The sample belongs to the cluster for which it has the largest weight. The weights sume to one for each sample. Think of them 
            as probabilities for cluster membership for that sample
        '''
        self.k = k
        self.sigma = None 
        self.mu = None
        self.pi = None
        self.membership_weights = None

    def fit(self, X, n_iter = 100):
        '''
        Parameters:
            X: data(features) that we want to group. N x f with N samples and f features 
            n_iter: Number of iterations we want to run 
        '''
        # X is N x f
        # sigma is k x f x f (one fxf matrix for each of the k clusters)
        # we initialize each covariance matrix to an identity matrix (np.eye)
        self.sigma = np.array([np.eye(X.shape[1]) for i in range(self.k)])
        # mu is k x f. Random initialization
        self.mu = np.random.random((self.k,X.shape[1]))
        # pi is k x 1. Each mixture weight is initialize to 1/k 
        self.pi = np.full((self.k,1),1/self.k)
        # weights is N x k
        self.membership_weights,log_likelihood = self.expectation(X,self.k,self.pi,self.mu,self.sigma)
        likelihood_history = [log_likelihood]

        for epoch in range(n_iter):
            self.mu = self.maximize_mean(X,self.k,self.membership_weights)

            self.sigma = self.maximize_covariance(X,self.k,self.membership_weights,self.mu)

            self.pi = self.maximize_mixtures(self.k,self.membership_weights)

            self.membership_weights,log_likelihood = self.expectation(X,self.k,self.pi,self.mu,self.sigma)

            likelihood_history.append(log_likelihood)
        return likelihood_history

            

    def expectation(self,X,k,pi,mu,sigma):
        # First consider a single sample (nth). For that sample the (membership weight) gamma_nk is obtained using Bayes
        # the ratio of the product of the prior (Prob(cluster == k)) and the 
        # likelihood (Prob(observe xn | cluster == k)) to the total probability. 

        # Here we are vectorizing it for the whole dataset and calculating the gamma_k for the entire dataset 
        
        # initialize a Nx0 matrix - we will column stack to it
        bayes_numerator = np.empty((X.shape[0],0))
        total_prob = np.zeros((X.shape[0],1))
        for cluster in range(k):
            # this gamma should be X.shape[0] x 1 i.e. N x 1. It is the product of the pi (mixture weight) and 
            # the gaussian prob density 
            gamma_k = pi[cluster] * stats.multivariate_normal.pdf(X,mu[cluster],sigma[cluster],allow_singular=True).reshape(-1,1)
            # we column stack it    
            bayes_numerator = np.column_stack((bayes_numerator,gamma_k))
            # add it to the total probability accumulator
            total_prob += gamma_k
        # here bayes_numerator should be N x k, and total prob should be X.shape[0] x 1
        return bayes_numerator / total_prob, np.sum(np.log(total_prob)) # we also return the log likelihood
           
    def maximize_mean(self,X,k,w):
        # w is N x k, so w_ik is a scalar
        # N_k = sum_over_N(weight_k) is scalar (Add weights for kth cluster over all samples) 
        # mu_k = sum_over_N(w_ik*x_i)/N_k, is 1xf
        # mu is k x f (one mu for each k)
        mu = np.empty((0,X.shape[1]))
        for cluster in range(k):
            N_k = np.sum(w[:,cluster],axis=0)
            mean_k = np.sum((w[:,cluster].reshape(-1,1) * X),axis=0)/N_k            
            mu = np.vstack((mu,np.expand_dims(mean_k,axis=0))) # 1xN x Nxf = 1xf
        return mu
    
    def maximize_covariance(self,X,k,w,mu):
         # w is N x k, so w_ik is a scalar
        # N_k = sum_over_N(weight_k) is scalar (Add weights for kth cluster over all samples) 
        # mu_k = sum_over_N(w_ik*x_i)/N_k, is 1xf
        # mu is k x f (one mu for each k)
        sigma = np.empty((0,X.shape[1],X.shape[1]))
        for cluster in range(k):
            N_k = np.sum(w[:,cluster],axis=0)
            sigma_k = (w[:,cluster].reshape(-1,1) * (X - mu[cluster,:])).T @ (X - mu[cluster,:])/N_k
            sigma_k = np.expand_dims(sigma_k,axis=0)
            sigma = np.vstack((sigma,sigma_k))
        return sigma

    def maximize_mixtures(self,k,w):
        new_pi = np.empty((0,1))
        for cluster in range(k):
            N_k = np.sum(w[:,cluster],axis=0)
            new_pi_k = N_k/w.shape[0]
            new_pi = np.vstack((new_pi,new_pi_k))
        return new_pi
# %%
# %%
iris = datasets.load_iris()
X = iris.data
clf = GaussianMixtureModel()
likelihood_history = clf.fit(X,n_iter = 100)
#%%
fig,ax = plt.subplots(1,1)
ax.plot(range(len(likelihood_history)),likelihood_history)
plt.plot()
# %%
