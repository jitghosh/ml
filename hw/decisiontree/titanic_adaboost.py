#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
class AdaBoostClassfier:
    def __init__(self, num_of_trees = 10, max_depth = None, min_samples_split = 2):
        self.num_of_trees = num_of_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.predictors = []
        self.error_rates_train = []
        self.error_rates_test = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        weight_vector = np.full((X.shape[0],),1/X.shape[0])
        for tree_idx in range(self.num_of_trees):
            tree_clf = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split,random_state=1234).fit(X,y,sample_weight = weight_vector)
            
            predictions = tree_clf.predict(X)
            # calculate weighted error rate
            weighted_error_rate = np.dot(weight_vector.T, (predictions != y).astype(int))
            self.error_rates_train.append(accuracy_score(predictions,y))
            # calculate predictor weight
            predictor_weight = 0.5 * np.log((1 - weighted_error_rate)/weighted_error_rate)
            self.predictors.append((tree_clf,predictor_weight))
            # update weights
            weight_vector *= (np.exp(-1*predictor_weight*predictions*y)/np.sum(weight_vector))
        return self

        
    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # should be num_of_trees x X.shape[0]
        preds_weights = [(predictor.predict(X),weight) for (predictor,weight) in self.predictors]
        weighted_predictions = np.array([predictions*weight for (predictions,weight) in preds_weights])
        self.error_rates_test = [accuracy_score(predictions,y) for predictions,_ in preds_weights]
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
train_X,test_X,train_y,test_y = train_test_split(df_X,df_y,test_size=0.20,stratify=df_y,shuffle=True,random_state=1234)

#%%
adaBoost_clf = AdaBoostClassfier(num_of_trees=3).fit(train_X.values,train_y.values)
#%%
predictions_train = adaBoost_clf.predict(train_X.values,train_y)
predictions_test = adaBoost_clf.predict(test_X.values,test_y.values)
# %%
print(accuracy_score(predictions_train,train_y.values))
print(accuracy_score(predictions_test,test_y.values))
# %%
fig,ax = plt.subplots(1,1)
ax.plot(np.arange(0,3),adaBoost_clf.error_rates_train,adaBoost_clf.error_rates_test)


# %%
