#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight

#%%

class DiabBinNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ntwk = nn.Sequential(nn.Linear(123,512),
                      nn.ReLU(),
                      nn.Linear(512,1024),
                      nn.ReLU(),
                      nn.Linear(1024,1024),
                      nn.ReLU(),
                      nn.Linear(1024,1),
                      nn.Sigmoid())
        
    def forward(self,X):
        return self.ntwk(X)

#%%
#%%
diabetes_bin = pd.read_csv('/home/jitghosh/work/pix/ml/mlproject_diabetes/data/diabetes_binary_health_indicators_BRFSS2015.csv')
allcols = diabetes_bin.columns.tolist()
numeric_col = ["BMI"]
label_col = 'Diabetes_binary'

cat_cols = set(allcols).difference(set(numeric_col + [label_col]))

diabetes_bin = pd.get_dummies(data=diabetes_bin,columns=list(cat_cols))
#%%
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(diabetes_bin.iloc[:,1:], 
                                                    diabetes_bin['Diabetes_binary'], 
                                                    test_size= 0.15, random_state= 777, 
                                                    shuffle = True, stratify= diabetes_bin['Diabetes_binary'])

#%%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
eval_results = []  
epochs = 200


all_train_loss = np.empty((0,epochs))
all_train_f1 = np.empty((0,epochs))
all_valid_loss = np.empty((0,epochs))
all_valid_f1 = np.empty((0,epochs))

for fold_idx,(train_idx, valid_idx) in enumerate(kf.split(X_train_bin, y_train_bin)):
    print(f"Validation fold : {fold_idx}")

    X_train, X_valid = X_train_bin.iloc[train_idx,:], X_train_bin.iloc[valid_idx,:]
    y_train, y_valid = y_train_bin.iloc[train_idx], y_train_bin.iloc[valid_idx]

    train_ds = torchdata.TensorDataset(torch.as_tensor(X_train.values.astype(np.float32)),
                                       torch.as_tensor(y_train.values.astype(np.float32)))
    

    
    model = DiabBinNetwork()    
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    train_f1 = []
    train_loss = [] 
    valid_f1 = []
    valid_loss = []

    for epoch in range(epochs):
        model.train()
        batch_train_loss = []
        for batch_X, batch_y in torchdata.DataLoader(train_ds,batch_size=10000,shuffle=True):
        
            
            preds = model(batch_X)
            loss = loss_fn(preds.reshape((preds.shape[0],)),batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_train_loss.append(loss.item())
        
        
        with torch.no_grad():
            train_loss.append(np.mean(batch_train_loss))
            preds = model(torch.as_tensor(X_train.values.astype(np.float32)))
            
            train_f1.append(f1_score(
                (preds.reshape(preds.shape[0],).detach().numpy() > 0.5).astype(int),
                y_train.values,
                sample_weight=compute_sample_weight(class_weight="balanced",y=y_train))) 
            
            preds = model(torch.as_tensor(X_valid.values.astype(np.float32)))
            loss = loss_fn(preds.reshape((preds.shape[0],)),torch.as_tensor(y_valid.values.astype(np.float32)))
            valid_loss.append(loss.item())
            valid_f1.append(f1_score(
                (preds.reshape(preds.shape[0],).detach().numpy() > 0.5).astype(int),
                y_valid.values,
                sample_weight=compute_sample_weight(class_weight="balanced",y=y_valid)))
            
        print(f"Epoch:{epoch},Train/Validation Loss: {train_loss[-1]:0.4f}/{valid_loss[-1]:0.4f},Train/Validation F1: {train_f1[-1]*100:0.2f}/{valid_f1[-1]*100:0.2f}")
    all_train_loss = np.vstack((all_train_loss,np.array(train_loss).reshape(1,epochs)))
    all_train_f1 = np.vstack((all_train_f1,np.array(train_f1).reshape(1,epochs)))
    all_valid_loss = np.vstack((all_valid_loss,np.array(valid_loss).reshape(1,epochs)))
    all_valid_f1 = np.vstack((all_valid_f1,np.array(valid_f1).reshape(1,epochs)))
#%%
all_train_loss = np.mean(all_train_loss,axis=0)
all_train_f1 = np.mean(all_train_f1,axis=0)
all_valid_loss = np.mean(all_valid_loss,axis=0)
all_valid_f1 = np.mean(all_valid_f1,axis=0)
    


    
#%%