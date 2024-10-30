#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd

#%%
class AdaBoostClassfier:
    def __init__(self, num_of_trees = 10, max_depth = None, min_samples_split = None, learning_rate = 1.0):
        self.num_of_trees = num_of_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.predictors = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        weight_vector = np.full((X.shape[0],),1/X.shape[0])
        for tree_idx in range(self.num_of_trees):
            tree_clf = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split,sample_weight = weight_vector).fit(X,y)
            
            predictions = tree_clf.predict(X)
            # indices of the wrong predictions
            error_indices = np.nonzero(predictions != y)[0]

            if (error_indices.shape[0] == 0): # found the perfect tree
                weighted_error_rate = 0
            else:
            # calculate weighted error rate
                weighted_error_rate = np.sum(weight_vector[error_indices])
            # calculate predictor weight
            predictor_weight = self.learning_rate * np.log((1 - weighted_error_rate)/weighted_error_rate) if weighted_error_rate != 0 else self.learning_rate

            self.predictors.append((tree_clf,predictor_weight))
            # update weights
            weight_vector[error_indices] *= np.exp(predictor_weight)
            # normalize weights
            weight_vector /= np.sum(weight_vector)

            if (error_indices.shape[0] == 0): # found the perfect tree
                break

        
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

#%%
df_data = pd.read_csv("c:/pix/ml/titanic/train.csv")
# drop columns that are meaningless for this
df_data = df_data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1).dropna()
# get the X data
df_X = df_data.drop("Survived",axis=1)
# get the y data
df_y = df_data["Survived"]
# one hot encode categoricals
df_X = pd.get_dummies(df_X,columns=["Pclass","Sex","Embarked"])
# make all X data numeric
df_X = df_X.astype(float)
# split
train_X,test_X,train_y,test_y = train_test_split(df_X,df_y,test_size=0.15,stratify=df_y,shuffle=True,random_state=1234)

#%%
#a = np.array([[1,1,0,0,1,1],[1,0,1,0,1,1]]).T
a = np.array([1,1,0,0,1,1])
b = np.array([1,0,0,1,1,1])
# %%
np.nonzero(a != b)
# %%
a != b
# %%
np.sum(np.empty((1,3)))
# %%
