#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd

#%%
class AdaBoostClassfier:
    def __init__(self, num_of_trees = 10, max_depth = None, min_samples_split = 2):
        self.num_of_trees = num_of_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.predictors = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        weight_vector = np.full((X.shape[0],),1/X.shape[0])
        for tree_idx in range(self.num_of_trees):
            tree_clf = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split,random_state=1234).fit(X,y,sample_weight = weight_vector)
            
            predictions = tree_clf.predict(X)
            # indices of the wrong predictions
            error_indices = np.nonzero(predictions != y)[0]
            # calculate weighted error rate
            weighted_error_rate = np.sum(weight_vector[error_indices])
            # calculate predictor weight
            predictor_weight = 0.5 * np.log((1 - weighted_error_rate)/weighted_error_rate)
            self.predictors.append((tree_clf,predictor_weight))
            # update weights
            weight_vector[error_indices] *= (np.exp(predictor_weight)/np.sum(weight_vector))
            if (error_indices.shape[0] == 0): # found the perfect tree
                break
        return self

        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # should be num_of_trees x X.shape[0]
        weighted_predictions = np.array([predictor.predict(X)*weight for (predictor,weight) in self.predictors])
        return (np.sum(weighted_predictions,axis=0) > 0).astype(int)
          


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
adaBoost_clf = AdaBoostClassfier(num_of_trees=400, max_depth=16,min_samples_split=8).fit(train_X.values,train_y.values)
#%%
predictions_train = adaBoost_clf.predict(train_X.values)
predictions_test = adaBoost_clf.predict(test_X.values)
# %%
print(accuracy_score(predictions_train,train_y.values))
print(accuracy_score(predictions_test,test_y.values))
# %%
