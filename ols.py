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

#%%
data = fetch_california_housing(as_frame=True)

#%%
cali_housing_X = data.data
cali_housing_y = data.target

train_X, test_X, train_y, test_y = train_test_split(cali_housing_X,cali_housing_y,test_size=0.15,random_state=1234)
train_X, valid_X, train_y, valid_y = train_test_split(train_X,train_y,test_size=0.15,random_state=1234)

#%%
mu = train_X.mean(axis=0)
std = train_X.std(axis=0)
train_X = (train_X - mu)/std
test_X = (test_X - mu)/std
valid_X = (valid_X - mu)/std

#%%
train_X = np.hstack((np.ones((train_X.shape[0],1)),train_X.values))
valid_X = np.hstack((np.ones((valid_X.shape[0],1)),valid_X.values))
test_X = np.hstack((np.ones((test_X.shape[0],1)),test_X.values))

#%%
theta = np.dot(np.dot(np.linalg.inv(np.dot(train_X.T,train_X)),train_X.T),train_y)

#%%
valid_y_hat = np.dot(valid_X, theta)
valid_mse = np.sum((valid_y_hat - valid_y)**2)/(valid_y.shape[0])
test_y_hat = np.dot(test_X, theta)
test_mse = np.sum((test_y_hat - test_y)**2)/(test_y.shape[0])

#%%
learning_rate = 0.00001
num_epochs = 500

np.random.seed(1234)
theta = np.random.randn(train_X.shape[1],1)
#%%
def calc_gradient(x: np.ndarray, theta: np.ndarray, y):
    return (2 * np.dot(x.T,(np.dot(x,theta) - y)))/x.shape[0]
def calc_mse(y_hat: np.ndarray, y: np.ndarray):
    return np.sum((y_hat - y)**2)/(y.shape[0])

#%%
train_loss: np.ndarray = []
valid_loss: np.ndarray = []
for epoch in range(0,num_epochs):
    for iter in range(train_X.shape[0]):
        # pick a random sample
        rand_index = np.random.randint(0,train_X.shape[0])
        x_i = train_X[rand_index:rand_index+1]
        y_i =  train_y[rand_index:rand_index+1].values
        gradients = calc_gradient(x_i,theta,y_i)
        theta = theta - gradients * learning_rate
        #print(f"iter {iter}: {theta}")
    train_y_hat = np.dot(train_X, theta).flatten()
    train_loss.append(calc_mse(train_y_hat,train_y))
    valid_y_hat = np.dot(valid_X, theta).flatten()
    valid_loss.append(calc_mse(valid_y_hat,valid_y))
    print(f"epoch {epoch}: {train_loss[-1]}, {valid_loss[-1]}")
    # we stop if the validation loss is not imporving significantly (abs tolerance 0.001) over an average of 5 values say
    # for last 20 epochs 
    if epoch > 20 and math.isclose(np.average(valid_loss[-10:-5]), np.average(valid_loss[-5:]), abs_tol=0.001):
        print(f"stopping at epoch {epoch}")
        break
# %%
ax = sns.lineplot(x=range(0,len(train_loss)),y=train_loss)
sns.lineplot(x=range(0,len(train_loss)),y=valid_loss,ax=ax)

#%%
def calc_gradient_ridge(x: np.ndarray, theta: np.ndarray, y, alpha: np.float32 = 0.001):
    return 2 * (np.dot(x.T,(np.dot(x,theta).flatten() - y)) + np.dot(alpha,theta).flatten())/x.shape[0]


#%%
train_loss: np.ndarray = []
valid_loss: np.ndarray = []
batch_size = 25
num_epochs = 1000
alpha = 0.0001
theta = np.random.randn(train_X.shape[1],)
batch_count = train_X.shape[0] // batch_size + (0 if train_X.shape[0] % batch_size == 0 else 1)
for epoch in range(0,num_epochs):
    # random shuffle the training data and the test data
    random_idx = np.random.randint(0,train_X.shape[0],size = (train_X.shape[0],))
    shuffled_train_X = train_X[random_idx]
    shuffled_train_y = train_y.values[random_idx]
    for batch_idx in range(0,batch_count):
        batch_X = train_X[batch_idx*batch_size:(batch_idx + 1)*batch_size]
        batch_y = train_y[batch_idx*batch_size:(batch_idx + 1)*batch_size]
        gradients = calc_gradient_ridge(batch_X,theta,batch_y)
        theta = theta - gradients * learning_rate
        #print(f"iter {iter}: {theta}")
    train_y_hat = np.dot(shuffled_train_X, theta).flatten()
    train_loss.append(calc_mse(train_y_hat,shuffled_train_y))
    valid_y_hat = np.dot(valid_X, theta).flatten()
    valid_loss.append(calc_mse(valid_y_hat,valid_y))
    print(f"epoch {epoch}: {train_loss[-1]}, {valid_loss[-1]}")
    # we stop if the validation loss is not imporving significantly (abs tolerance 0.001) over an average of 5 values say
    # for last 20 epochs 
    if epoch > 20 and math.isclose(np.average(valid_loss[-10:-5]), np.average(valid_loss[-5:]), abs_tol=0.001):
        print(f"stopping at epoch {epoch}")
        break
# %%
ax = plt.subplot(1,1,1)
ax.set_ylabel("ridge regression loss")
ax.set_xlabel("epochs")
plt.plot(range(0,len(train_loss)),train_loss,"r",valid_loss,"g")
plt.legend(["training","validation"])

#%% Markdown
def randomized_k_fold_split(X: np.ndarray, y: np.ndarray, k = 7):
    ret_x = []
    ret_y = []
    # shuffle
    random_idx = np.random.randint(0,X.shape[0],size = (X.shape[0],))
    shuffled_X = X[random_idx]
    shuffled_y = y.values[random_idx]
    # k = 7 i.e. each fold will be (100/k) %
    folds_X = np.array_split(shuffled_X,k,0)
    folds_y = np.array_split(shuffled_y,k,0)
    return (folds_X, folds_y)
# %%
#%%
train_loss: np.ndarray = []
valid_loss: np.ndarray = []
k = 7
batch_size = 50
num_epochs = 3000
alpha = 0.0001
theta = np.random.randn(train_X.shape[1],)

for epoch in range(0,num_epochs):
    folds_X, folds_y = randomized_k_fold_split(train_X,train_y)
    fold_train_loss: np.ndarray = []
    fold_valid_loss: np.ndarray = []
    for fold_idx in range(k):
        train_kfolds_X = np.concatenate([fold for idx,fold in enumerate(folds_X) if idx != fold_idx],axis=0)
        valid_kfolds_X = folds_X[fold_idx]
        train_kfolds_y = np.concatenate([fold for idx,fold in enumerate(folds_y) if idx != fold_idx],axis=0)
        valid_kfolds_y = folds_y[fold_idx]
        batch_count = train_kfolds_X.shape[0] // batch_size + (0 if train_kfolds_X.shape[0] % batch_size == 0 else 1)
        for batch_idx in range(0,batch_count):
            batch_X = train_kfolds_X[batch_idx*batch_size:(batch_idx + 1)*batch_size]
            batch_y = train_kfolds_y[batch_idx*batch_size:(batch_idx + 1)*batch_size]
            gradients = calc_gradient_ridge(batch_X,theta,batch_y)
            theta = theta - gradients * learning_rate
        train_y_hat = np.dot(train_kfolds_X, theta).flatten()
        fold_train_loss.append(calc_mse(train_y_hat,train_kfolds_y))
        valid_y_hat = np.dot(valid_kfolds_X, theta).flatten()
        fold_valid_loss.append(calc_mse(valid_y_hat,valid_kfolds_y))
    train_loss.append(np.average(fold_train_loss))
    valid_loss.append(np.average(fold_valid_loss))
    print(f"epoch {epoch}: {train_loss[-1]}, {valid_loss[-1]}")

    if epoch > 3000 and math.isclose(np.average(valid_loss[-100:-50]), np.average(valid_loss[-50:]), abs_tol=0.001):
        print(f"stopping at epoch {epoch}")
        break
# %%
ax = plt.subplot(1,1,1)
ax.set_ylabel("ridge regression loss")
ax.set_xlabel("epochs")
plt.plot(range(0,len(train_loss)),train_loss,"r",valid_loss,"g")
plt.legend(["training","validation"])
# %%
