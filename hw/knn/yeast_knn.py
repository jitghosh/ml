# %%
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import StandardScaler
random_state = 1234


# %%
'''
KNN Classifier class
'''
class KNNClassfier:
    def __init__(self, n_neighbors=7):
        '''
        Parameters:
            n_neighbors: How many neighbors should we look at when determining "closeness"
        '''

        # member variable to store the number of neighbors to look at when predicting 
        self.n_neighbors = n_neighbors
        # member variable to store the training data features. Will be set in fit()
        self.X = None
        # member variable to store the training data labels. Will be set in fit() 
        self.y = None

    def fit(self, X, y):
        '''
        We do nothing in fit() other than simply storing the training data to be used during predict.
        We traditionally return self (the classifier object itself) from fit() as a pattern so that 
        we can chain calls i.e. do something like classifier = KNNClassfier(7).fit(X,y) in one step 
        (which other wise can be done in two steps like classifier = KNNClassifier(7) and then classifier.fit(X,y))
        This is not required - but scikit learn implements this pattern. 
        '''
        self.X = X
        self.y = y
        return self

    def predict(self, test_X):
        '''
        Return the majority class for each sample in test_X

        Parameters:
            test_X: test data (features only) of shape Txd i.e. T samples with d features each
        '''

        # apply_along_axis() when called with axis=1 will supply one slice along the columns (i.e. one row) of the arr data 
        # at a time to the supplied callable (in our case find_majority_class). This is equivalent to looping over the dataset 
        # rows and calling the function ourselves with one row at a time passed into it, but this is much faster since it is 
        # vectorized
        return np.apply_along_axis(self.find_majority_class,arr=test_X,axis=1)

    def find_majority_class(self, target_x):

        # calculate the distances from each of the training samples to the target test sample 
        # Once again we use apply_along_axis to iterate over the training samples we have already
        # stored. The second parameter to calc_distance is passed as a named (X_to) parametet to 
        # apply_along_axis which in turn will pass it along to calc_distance for each call 
        distances = np.apply_along_axis(
            self.calc_distance, axis=1, arr=self.X, X_to=target_x
        )
        # The returned distances should be of shape Nx1 where we have N training samples and a distance 
        # of each training sample to the target test sample. Our goal is to find 'n_neighbors' number of samples
        # with the lowest distances from the test target sample and then take the majority label for those. 
        # So we need to sort the distances. We join the distances with the training labels (which is also of the same shape Nx1)
        # so that when we sort the distances the lables get sorted with it at the same time.
        
        # Join into a dataframe supplying the data as a dict (the keys becomes columns in the dataframe)
        y_dist = pd.DataFrame({"dist": distances, "y": self.y})
        # Sort the dataframe by the dist column in ascending order
        sorted_values = y_dist.sort_values(by=["dist"], ascending=True).values
        # take the 'n_neighbors' number of labels from the beginning
        top_n_neighbors = sorted_values[0 : self.n_neighbors, 1]
        # Use the Counter class to find the counts per label (returns a dictionary with labels as keys as counts as values)
        label_frequencies = Counter(top_n_neighbors)
        # Find the label with highest frequency - check the docs for most_common and Counter
        highest_count = label_frequencies.most_common(1)
        # return the label with the highest frequency - will be a list with a single tuple of label and its count 
        return highest_count[0][0]

    def calc_distance(self, X_from, X_to):
        return linalg.norm(X_to - X_from)

#%%
# data processing code and prediction/accuracy code here. Do not forget to scale data since KNN is scale sensitive 

# %%
col_names = [
    "seq_name",
    "mcg",
    "gvh",
    "alm",
    "mit",
    "er1",
    "pox",
    "vac",
    "nuc",
    "cls_name",
]

# %%
with open("C:\pix\ml\hw\knn\yeast.data", "r") as yeast_file, open(
    "C:\pix\ml\hw\knn\yeast2.data", "w"
) as yeast2_file:
    lns = []
    ln = yeast_file.readline()
    while ln != "":
        lns.append(
            ",".join([itm.strip() for itm in ln.split(" ") if itm.strip() != ""]) + "\n"
        )
        ln = yeast_file.readline()
    yeast2_file.writelines(lns)

# %%
df_yeast = pd.read_csv(
    "C:\pix\ml\hw\knn\yeast2.data", header=None, names=col_names, sep=","
).drop("seq_name",axis=1)

#df_yeast["seq_name"] = LabelEncoder().fit_transform(df_yeast["seq_name"])

train_X, test_X, train_y, test_y = train_test_split(
    df_yeast.drop("cls_name", axis=1),
    df_yeast["cls_name"],
    shuffle=True,
    stratify=df_yeast["cls_name"],
    test_size=0.15,
    random_state=random_state
)

scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X) 
# %%
clf = KNNClassfier(7).fit(train_X, train_y)
preds = clf.predict(test_X)
accuracy_score(test_y,preds)

# %%
