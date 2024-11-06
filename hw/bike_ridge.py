#%%
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import math
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#%%
# fetch dataset 
bike_sharing = fetch_ucirepo(id=275) 
  
# data (as pandas dataframes) 
# leaving year out since we want the model to work for other years as well

bike_X: pd.DataFrame = bike_sharing.data.features.loc[:,['season', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
bike_y = bike_sharing.data.targets 
# %%
bike_X = pd.get_dummies(bike_X,columns=['season','mnth','hr','weekday','weathersit'])

# %%
bike_train_X, bike_test_X, bike_train_y, bike_test_y = train_test_split(bike_X,bike_y,test_size=0.15,random_state=1234)
#%%
bike_train_X = np.hstack((np.ones((bike_train_X.shape[0],1)),bike_train_X.values))
bike_test_X = np.hstack((np.ones((bike_test_X.shape[0],1)),bike_test_X.values))

#%%
def calc_gradient_ridge(x: np.ndarray, theta: np.ndarray, y, alpha: np.float32 = 0.001):
    return 2*(x.T@((x@theta) - y) + alpha*theta)/x.shape[0]
def calc_mse(y_hat: np.ndarray, y: np.ndarray):
    y_hat = y_hat.reshape(y_hat.shape[0],)
    y = y.reshape(y.shape[0],)
    return np.sum((y_hat - y)**2)/(y.shape[0])
# %%
train_loss: np.ndarray = []
valid_loss: np.ndarray = []
k = 7
batch_size = 1000
num_epochs = 10000
alpha = 0.0001
learning_rate = 0.0001
theta = np.random.randn(bike_train_X.shape[1],1)

for epoch in range(0,num_epochs):
    
    fold_train_loss: np.ndarray = []
    fold_valid_loss: np.ndarray = []
    kfold = KFold(n_splits = 5, shuffle=True, random_state=1234)
    for fold_idx_train, fold_idx_valid in kfold.split(bike_train_X):
        train_kfolds_X = bike_train_X[fold_idx_train]
        valid_kfolds_X = bike_train_X[fold_idx_valid]
        train_kfolds_y = bike_train_y.values[fold_idx_train]
        valid_kfolds_y = bike_train_y.values[fold_idx_valid]
        batch_count = train_kfolds_X.shape[0] // batch_size + (0 if train_kfolds_X.shape[0] % batch_size == 0 else 1)
        for batch_idx in range(0,batch_count):
            batch_X = train_kfolds_X[batch_idx*batch_size:(batch_idx + 1)*batch_size]
            batch_y = train_kfolds_y[batch_idx*batch_size:(batch_idx + 1)*batch_size]
            gradients = calc_gradient_ridge(batch_X,theta,batch_y,alpha)
            theta = theta - gradients * learning_rate
        train_y_hat = np.dot(train_kfolds_X, theta).flatten()
        fold_train_loss.append(calc_mse(train_y_hat,train_kfolds_y))
        valid_y_hat = np.dot(valid_kfolds_X, theta).flatten()
        fold_valid_loss.append(calc_mse(valid_y_hat,valid_kfolds_y))
    train_loss.append(np.average(fold_train_loss))
    valid_loss.append(np.average(fold_valid_loss))
    if(epoch % 100 == 0):
        print(f"epoch {epoch}: {train_loss[-1]}, {valid_loss[-1]}")

    if epoch > 1000 and np.average(valid_loss[-50:]) - np.average(valid_loss[-100:-50]) > 10:
        print(f"stopping at epoch {epoch} - validation loss has started to increase")
        break
#%%
ax = sns.lineplot(x=range(0,len(train_loss)),y=train_loss)
sns.lineplot(x=range(0,len(train_loss)),y=valid_loss,ax=ax)
# %%
