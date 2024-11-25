# %%
import numpy as np
import numpy.linalg as linalg
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# %%
class KMeansLearner:
    '''
    This class has no predict. This is unsupervised learning so there is nothing to predict. We just fit and 
    find the clusters (i.e. the cluster centroids) 
    '''
    def __init__(self, k=5,centroids = None):
        '''
        Parameters:
            k: number of clusters we want to find
            centroids: Initial value for centroids if supplied. 
            Will be a k x d array where k is the number of clusters and d is the dimension of the data 
            (i.e. number of features) i.e. one centroid per cluster
        '''
        self.k = k
        self.centroids = centroids
        # we also store the centroids at each iteration
        self.iteration_history = []

    def fit(self, train_X):
        '''
        Parameters:
            train_X: the data we want to cluster. Of shape Nxd.
        '''
        # if we were not given initial values of centroids during object ctor, we initialize now
        # to a k x d array (one centroid per cluster) of random float values
        if(self.centroids is None):
            self.centroids = np.random.random((self.k, train_X.shape[1]))
        # this is our very first set of centroids - so add them to the history
        self.iteration_history.append(self.centroids.copy())
        # list of clusters that each training sample belongs to
        cluster_memberships = None

        # we run the loop for as long as needed
        while True:
            # we calculate distances between each of our training samples and the current centroids
            # we are going to have k distances for each training sample, one for each centroid. 
            # So distances is going to be a N x k array
            # Like before we pass in the second argument to calc_distance as a named argument to
            # apply_along_axis
            distances = np.apply_along_axis(
                kml.calc_distance, arr=train_X, axis=1, centroids=self.centroids
            )
            # for each of the training samples, the cluster it belongs to will be the centroid number
            # that is has the least distance to. Argmin will give us a (150,) array of centroid number (cluster membership)
            # for each sample 
            cluster_memberships = np.argmin(distances, axis=1)

            # now we need to recalculate the centrids based on the sample cluster memberships.
            # To do that we first stack(join) the training data with the cluster membership data 
            # (effectively stacking a new column at the end of the training data).
            X_with_cluster = np.hstack((train_X, cluster_memberships.reshape(-1, 1)))
            # indicator variable to control the loop
            centroids_updated = False
            # Now for each cluster
            for cluster_idx in range(self.k):
                # we find the samples which are now predicted to be the members of this cluster in this iteration
                X_group_for_cluster = X_with_cluster[
                    X_with_cluster[:, -1] == cluster_idx, :-1 # exclude the last cluster membership column
                ]
                # if no samples are members of this cluster - continue
                if X_group_for_cluster.shape[0] == 0:
                    continue
                # calculate the new centroid of this cluster as a mean of the sample subset belonging to this cluster
                new_centroid = np.mean(X_group_for_cluster, axis=0)
                # the new centroid value is the same as the old one - continue
                if np.all(new_centroid == self.centroids[cluster_idx, :]):
                    continue
                else:
                    # update the centroid for this cluster
                    self.centroids[cluster_idx, :] = new_centroid
                    # mark the indicator to indicate that in this iteration at least one centroid was updated
                    centroids_updated = True
            # make a copy of the centroids and append it to the history. Making a copy is important. 
            # If you do not then in each loop you will be updating the same centroids.  
            self.iteration_history.append(self.centroids.copy())

            # no centroids updated - we are done
            if not centroids_updated:
                break

    def calc_distance(self, X_from, centroids):
        return linalg.norm(centroids - X_from, axis=1)


# %%
# data processing code and prediction/accuracy code here. Do not forget to scale data since KNN is scale sensitive 

df_iris = pd.read_csv(
    "./iris.data",
    header=None,
    names=["petal_length", "petal_width", "sepal_length", "sepal_width", "cls"],
)
kml = KMeansLearner(k=3)
data = StandardScaler().fit_transform(df_iris.loc[:,["petal_length","petal_width"]])
kml.fit(data)
fig,ax = plt.subplots(1,1)
ax.scatter(data[:,0],data[:,1],c=["r" if cls.endswith("setosa") else ("b" if cls.endswith("versicolor") else "g") for cls in df_iris["cls"]])
ax.scatter(kml.centroids[:,0],kml.centroids[:,1],marker='D', c='y')



# %%
kml = KMeansLearner(k=3, centroids=np.array([[6.6,3.7],[6.2,3.2],[6.5,3.0]]))

data = np.array(
    [
        [5.5, 4.2],
        [5.1, 3.8],
        [4.7, 3.2],
        [5.9, 3.2],
        [6.7, 3.1],
        [4.9, 3.1],
        [5.0, 3.0],
        [6.0, 3.0],
        [4.6, 2.9],
        [6.2, 2.8],
    ]
)


kml.fit(data)
fig, ax = plt.subplots(1, 1)
pass
# %%
