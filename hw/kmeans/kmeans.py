# %%
import numpy as np
import numpy.linalg as linalg
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# %%
class KMeansLearner:
    def __init__(self, n_clusters=5,centroids = None):
        self.n_clusters = n_clusters
        self.centroids = centroids
        self.iteration_history = []

    def fit(self, train_X):
        if(self.centroids is None):
            self.centroids = np.random.random((self.n_clusters, train_X.shape[1]))
        self.iteration_history.append(self.centroids.copy())
        # list of clusters that each training sample belongs to
        cluster_memberships = None
        while True:
            distances = np.apply_along_axis(
                kml.calc_distance, arr=train_X, axis=1, centroids=self.centroids
            )
            cluster_memberships = np.argmin(distances, axis=1)

            X_with_cluster = np.hstack((train_X, cluster_memberships.reshape(-1, 1)))
            update_count = 0
            for cluster_idx in range(self.n_clusters):
                X_group_for_cluster = X_with_cluster[
                    X_with_cluster[:, -1] == cluster_idx, :-1
                ]
                if X_group_for_cluster.shape[0] == 0:
                    continue
                new_centroid = np.mean(X_group_for_cluster, axis=0)
                if np.all(new_centroid == self.centroids[cluster_idx, :]):
                    continue
                else:
                    self.centroids[cluster_idx, :] = new_centroid
                    update_count += 1
            self.iteration_history.append(self.centroids.copy())
            if update_count == 0:
                break

    def calc_distance(self, X_from, centroids):
        return linalg.norm(centroids - X_from, axis=1)


# %%

df_iris = pd.read_csv(
    "./iris.data",
    header=None,
    names=["petal_length", "petal_width", "sepal_length", "sepal_width", "cls"],
)
kml = KMeansLearner(n_clusters=3)
data = StandardScaler().fit_transform(df_iris.loc[:,["petal_length","petal_width"]])
kml.fit(data)
fig,ax = plt.subplots(1,1)
ax.scatter(data[:,0],data[:,1],c=["r" if cls.endswith("setosa") else ("b" if cls.endswith("versicolor") else "g") for cls in df_iris["cls"]])
ax.scatter(kml.centroids[:,0],kml.centroids[:,1],marker='D', c='y')



# %%
kml = KMeansLearner(n_clusters=3, centroids=np.array([[6.6,3.7],[6.2,3.2],[6.5,3.0]]))

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
