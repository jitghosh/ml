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
    def __init__(self, X: np.ndarray, y: np.ndarray, clsarr: set[int], parent = None):
        self.feature_idx = None
        self.categorical = None
        self.split_at = None
        self.gini = 1 - np.sum([(y[y == cls].shape[0]/y.shape[0])**2 for cls in clsarr])
        self.class_count = {cls:y[y == cls].shape[0] for cls in clsarr}
        self.parent = parent
        self.right: Node = None
        self.left: Node = None
        self.X = X
        self.y = y
        
        

class DecisionTreeClassifier:
    def __init__(self):
        self.head = None

    def fit(self, X: np.ndarray, y: np.ndarray, clsarr: set[int], categorical_col_idx: set[int]):
        self.head = Node(X,y,clsarr)
        self.split(self.head,clsarr,categorical_col_idx,None)

    
    def split(self, cur_node: Node, clsarr: set[int], categorical_col_idx: set[int], parent: Node = None):
        
        (gini_left,gini_right,feature_idx,split_at) = self.calculate_feature_to_split_based_on_min_cost(cur_node.X,cur_node.y,clsarr,categorical_col_idx)
        if feature_idx is not None: # splittable
            return None
        else:
            cur_node.feature_idx = feature_idx
            cur_node.categorical = feature_idx in categorical_col_idx
            cur_node.split_at = split_at
            if cur_node.categorical:
                left_indices = np.argwhere(cur_node.X[:,feature_idx] == 0)
                right_indices = np.argwhere(cur_node.X[:,feature_idx] == 1)
            else:
                left_indices = np.argwhere(cur_node.X[:,feature_idx] < split_at)
                right_indices = np.argwhere(cur_node.X[:,feature_idx] >= split_at)
            cur_node.left = Node(cur_node.X[left_indices,:],cur_node.y[left_indices],clsarr,cur_node)
            cur_node.right = Node(cur_node.X[right_indices,:],cur_node.y[right_indices],clsarr,cur_node)

            self.split(cur_node.left,clsarr,categorical_col_idx,cur_node)
            self.split(cur_node.right,clsarr,categorical_col_idx,cur_node)


    def calculate_feature_to_split_based_on_min_cost(self, X: np.ndarray, y: np.ndarray, clsarr: set[int], categorical_col_idx: set[int]):
        cost_arr = []
        for featureidx in X.shape[1]:
            xy = np.column_stack((X[:,featureidx],y))
        
            if featureidx in categorical_col_idx:
                cost,gini_left,gini_right = self.calculate_cost_and_gini(xy[xy == 0,1],xy[xy == 1, 1],clsarr)
                cost_arr.append((cost,gini_left,gini_right,featureidx,None))
            else:
                xy = np.sort(xy,axis=0)
                idxarr = np.array(range(1,X.shape[0]-1))
                cost,gini_left,gini_right = np.array([self.calculate_cost_and_gini(xy[0:idx,1],xy[idx:,1],clsarr) for idx in idxarr])
                split_at = X[np.argmin(cost)]
                cost_arr.append((cost,gini_left,gini_right,featureidx,split_at))

        (cost,gini_left,gini_right,featureidx,split_at) =  sorted(cost,key = lambda x: x[0])[0]
        return gini_left,gini_right,featureidx,split_at
    
    def calculate_cost_and_gini(self,y_left: np.ndarray, y_right: np.ndarray, clsarr: set[int]):
        gini_left = 1 - np.sum([(y_left[y_left == cls].shape[0]/y_left.shape[0])**2 for cls in clsarr])
        gini_right = 1 - np.sum([(y_right[y_right == cls].shape[0]/y_right.shape[0])**2 for cls in clsarr])
        return (gini_left * y_left.shape[0]/(y_left.shape[0] + y_right.shape[0])) + (gini_right * y_right.shape[0]/(y_left.shape[0] + y_right.shape[0])),gini_left,gini_right
    
   
#%%
a = df_X.loc[:,["Fare","Pclass_1","Pclass_2","Pclass_3"]].values.astype(float)

# %%
np.argwhere(a)
# %%
a[-1,:]
# %%
np.argwhere(a[:,3] == 0)
# %%
class Test:
    def __init__(self, data):
        self.data = data

test = Test(a)

np.argwhere(test.data[:,3] == 0)
# %%
