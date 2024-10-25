#%%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os



#%%
df_data = pd.read_csv("c:/pix/ml/titanic/train.csv")
df_data.head()
# %%
df_data = df_data.drop(["PassengerId","Name"],axis=1)
# %%
df_X = df_data.drop("Survived",axis=1)
df_y = df_data["Survived"]
# %%
df_X = pd.get_dummies(df_X,columns=["Pclass","Sex","Embarked"])
# %%
df_X.columns
# %%
class Node:
    def __init__(self):
        pass
class DecisionTreeClassifier:
    def __init__(self):
        pass
    def calculate_feature_split(self, X: np.ndarray, y: np.ndarray, clsarr: list[int], categorical_col_idx = list[int]):
        cost_arr = []
        for featureidx in X.shape[1]:
            xy = np.column_stack((X[:,featureidx],y))
        
            if featureidx in categorical_col_idx:
                cost = self.calculate_split_cost(xy[xy == 0,1],xy[xy == 1, 1],clsarr)
                cost_arr.append((cost,featureidx,None))
            else:
                xy = np.sort(xy,axis=0)
                idxarr = np.array(range(1,X.shape[0]-1))
                cost = np.array([self.calculate_split_cost(xy[0:idx,1],xy[idx:,1],clsarr) for idx in idxarr])
                split_at = X[np.argmin(cost)]
                cost_arr.append((cost,featureidx,split_at))
        return sorted(cost,key = lambda x: x[0])[0]
    
    def calculate_split_cost(self,y_left: np.ndarray, y_right: np.ndarray, clsarr: list[int]):
        gini_left = 1 - np.sum([(y_left[y_left == cls].shape[0]/y_left.shape[0])**2 for cls in clsarr])
        gini_right = 1 - np.sum([(y_right[y_right == cls].shape[0]/y_right.shape[0])**2 for cls in clsarr])
        return (gini_left * y_left.shape[0]/(y_left.shape[0] + y_right.shape[0])) + (gini_right * y_right.shape[0]/(y_left.shape[0] + y_right.shape[0]))
    
   
#%%
 
node = Node()
split_at = node.calculate_split_with_min_gini(df_X["Fare"].values,df_y.values,[0,1])
    
# %%
a_sorted = np.sort(a,axis=0)

# %%
print(np.apply_along_axis(lambda x: x*100,axis=0,arr=a[:,0]))
# %%
a.shape
# %%
df_X.columns

# %%
