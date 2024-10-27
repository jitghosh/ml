#%%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from collections import Counter



#%%
df_data = pd.read_csv("c:/pix/ml/titanic/train.csv")
df_data = df_data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1).dropna()

df_X = df_data.drop("Survived",axis=1)
df_y = df_data["Survived"]
df_X = pd.get_dummies(df_X,columns=["Pclass","Sex","Embarked"])
df_X = df_X.astype(float)

train_X,test_X,train_y,test_y = train_test_split(df_X,df_y,test_size=0.15,stratify=df_y,shuffle=True,random_state=1234)
# %%
'''
df_data = pd.read_csv("c:/pix/ml/iris.data",header=None)
df_X = df_data.drop([4],axis=1)
df_y = df_data[4]
df_y = LabelEncoder().fit(["Iris-setosa","Iris-versicolor","Iris-virginica"]).transform(df_y)
'''
# %%
class Node:
    def __init__(self, X: np.ndarray, y: np.ndarray, parent = None):
        self.X = X
        self.y = y
         
        values,counts = np.unique(y,return_counts=True)
        self.predict_class = values[np.argmax(counts)]
        self.gini = 1 - np.sum([(y[y == cls].shape[0]/y.shape[0])**2 for cls in values])       
        self.parent = parent
        self.depth = 1 if parent is None else parent.depth + 1

        self.right: Node = None
        self.left: Node = None
        self.feature_idx = None
        self.categorical = None
        self.split_at = None
        self.cost = None
        
        
        

class DecisionTreeClassifier:
    def __init__(self, max_depth = None, min_sample_split = None):
        self.head = None
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.clsarr = None

    def predict(self,X:np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.__predict_internal,axis=1,arr=X,use_node = self.head)

    def __predict_internal(self, X:np.ndarray, use_node: Node) -> float:
        if (use_node.left is None and use_node.right is None):
            return use_node.predict_class
        elif use_node.categorical:
            return self.__predict_internal(X,use_node.left if X[use_node.feature_idx].item() == 0 else use_node.right)
        else:
            return self.__predict_internal(X,use_node.left if X[use_node.feature_idx].item() < use_node.split_at else use_node.right)
    
    def fit(self, X: np.ndarray, y: np.ndarray, categorical_col_idx: set[int]):
        self.clsarr = np.unique(y)
        self.head = Node(X,y)
        self.__split(self.head,categorical_col_idx,None)

    
    def __split(self, cur_node: Node, categorical_col_idx: set[int], parent: Node = None):
        
        if (parent is not None and self.max_depth is not None and parent.depth + 1 > self.max_depth):
            return
        if(self.min_sample_split is not None and cur_node.X.shape[0] <= self.min_sample_split):
            return
        if (np.unique(cur_node.y).shape[0] == 1):
            return

        (cost,gini_left,gini_right,feature_idx,split_at) = self.__calculate_feature_to_split_based_on_min_cost(cur_node.X,cur_node.y,categorical_col_idx)
        
        if cost == None:
            return
        
        if feature_idx in categorical_col_idx:
            left_indices = np.argwhere(cur_node.X[:,feature_idx] == 0).flatten()
            right_indices = np.argwhere(cur_node.X[:,feature_idx] == 1).flatten()
        else:
            left_indices = np.argwhere(cur_node.X[:,feature_idx] < split_at).flatten()
            right_indices = np.argwhere(cur_node.X[:,feature_idx] >= split_at).flatten()

        # if split is going to result in one side being "no samples" do not split
        if (left_indices.shape[0] == 0 or right_indices.shape[0] == 0):
            return
        
        cur_node.feature_idx = feature_idx
        cur_node.categorical = feature_idx in categorical_col_idx
        cur_node.split_at = split_at
        
        cur_node.left = Node(cur_node.X[left_indices,:],cur_node.y[left_indices],cur_node)
        cur_node.right = Node(cur_node.X[right_indices,:],cur_node.y[right_indices],cur_node)

        self.__split(cur_node.left,categorical_col_idx,cur_node)
        self.__split(cur_node.right,categorical_col_idx,cur_node)


    def __calculate_feature_to_split_based_on_min_cost(self, X: np.ndarray, y: np.ndarray, categorical_col_idx: set[int]):
        cost_arr = np.empty((0,5))
        for feature_idx in range(0,X.shape[1]):
            xy = np.column_stack((X[:,feature_idx],y))
        
            if feature_idx in categorical_col_idx:
                cost = self.__calculate_cost_and_gini(xy,feature_idx,categorical=True,split_point = None)
                if cost[0,0] != None:
                    cost_arr = np.concatenate((cost_arr,cost),axis=0)
            else:                
                possible_split_points = range(1,X.shape[0]-1)
                for split_point in possible_split_points:
                    cost_for_split_point = self.__calculate_cost_and_gini(xy,feature_idx,categorical=False,split_point=split_point)
                    if cost_for_split_point[0,0] != None:
                        cost_arr = np.concatenate((cost_arr,cost_for_split_point))
        
        if cost_arr.shape[0] == 0:
            return (None,None,None,None,None)
        
        min_cost_idx = np.argmin(cost_arr[:,0],axis = 0)
        return cost_arr[min_cost_idx,0], cost_arr[min_cost_idx,1],cost_arr[min_cost_idx,2],int(cost_arr[min_cost_idx,3]),cost_arr[min_cost_idx,4]
    
   
    def __calculate_cost_and_gini(self,xy: np.ndarray, feature_idx: int, categorical: bool, split_point: int):
        if categorical:
            y_left = xy[xy[:,0] == 0,1]
            y_right = xy[xy[:,0] == 1,1]
        else:
            y_left = xy[xy[:,0] < xy[split_point,0].item(),1]
            y_right = xy[xy[:,0] >= xy[split_point,0].item(),1]

        #a split was not possible at that split point because either side resulted in no samples satisfying the split condition
        if (y_left.shape[0] == 0 or y_right.shape[0] == 0):
            return np.array([None]*5).reshape(1,-1)
        
        gini_left = 1 - np.sum([(y_left[y_left == cls].shape[0]/y_left.shape[0])**2 for cls in self.clsarr])
        gini_right = 1 - np.sum([(y_right[y_right == cls].shape[0]/y_right.shape[0])**2 for cls in self.clsarr])
        return np.array([(gini_left * y_left.shape[0]/(y_left.shape[0] + y_right.shape[0])) + (gini_right * y_right.shape[0]/(y_left.shape[0] + y_right.shape[0])),gini_left,gini_right,feature_idx,xy[split_point,0].item() if not categorical else None]).reshape(1,-1)
    
   
#%%

tree = DecisionTreeClassifier(min_sample_split=12, max_depth=16)
tree.fit(train_X.values,train_y.values,categorical_col_idx=range(4,12))
predictions_train = tree.predict(train_X.values)
print(f"train accuracy = {accuracy_score(predictions_train,train_y.values)}")
print(f"train precision = {precision_score(predictions_train,train_y.values)}")
print(f"train recall = {recall_score(predictions_train,train_y.values)}")
predictions_test = tree.predict(test_X.values)
print(f"test accuracy = {accuracy_score(predictions_test,test_y.values)}")
print(f"test precision = {precision_score(predictions_test,test_y.values)}")
print(f"test recall = {recall_score(predictions_test,test_y.values)}")


# %%
a = np.array([[21,3,5],[14,19,2]])
np.argmin(a[:,0])

# %%
[None]*5

# %%
