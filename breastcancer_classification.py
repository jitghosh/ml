#%%
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as pyplt
import pandas as pd

# %%

def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-1 * a))

def gradient_cross_entropy(x: np.ndarray, w: np.ndarray, y):
    return (x.T@(sigmoid(x@w) - y))/x.shape[0]

def cross_entropy_loss(y_hat: np.ndarray, y: np.ndarray):
    # assume batch of mxn features. So x is mxn, w is nx1 and y is nx1 
    return (-1 * np.sum(y.T@np.log(y_hat) + (1 - y).T@np.log(1 - y_hat)))/y.shape[0]

def predict(y_hat: np.ndarray, threshold: np.float32 = 0.25) -> np.ndarray:
    return np.int32(y_hat >= threshold)
#%%

df_data = pd.read_csv("wdbc.data",header=None).drop(0,axis=1)
#%%
df_data.loc[df_data[1] == 'M',1] = 1
df_data.loc[df_data[1] == 'B',1] = 0
df_data[1] = pd.to_numeric(df_data[1])
df_data.head()


# %%
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(
                                                df_data.drop(1,axis=1),
                                                df_data[1],
                                                test_size=0.15,
                                                random_state=1234,
                                                shuffle=True,
                                                stratify=df_data[1])
df_train_x, df_valid_x, df_train_y, df_valid_y = train_test_split(
                                                df_train_x,
                                                df_train_y,
                                                test_size=0.15,
                                                random_state=1234,
                                                shuffle=True,
                                                stratify=df_train_y)
#%%

mu = df_train_x.mean(axis=0)
std = df_train_x.std(axis=0)
df_train_x = (df_train_x - mu)/std
df_test_x = (df_test_x - mu)/std
df_valid_x = (df_valid_x - mu)/std
train_x = df_train_x.values
train_y = df_train_y.values.reshape(df_train_y.shape[0],1)
valid_x = df_valid_x.values
valid_y = df_valid_y.values.reshape(df_valid_y.shape[0],1)
# %%
train_loss: np.ndarray = []
valid_loss: np.ndarray = []
valid_accuracy : np.ndarray = []
k = 3
batch_size = 5
num_epochs = 200000
learning_rate = 0.0001
w = np.random.randn(df_train_x.shape[1],1)

for epoch in range(0,num_epochs):
    gradients = gradient_cross_entropy(train_x,w,train_y)
    w = w - gradients * learning_rate
    train_y_hat = sigmoid(np.dot(train_x, w))
    train_loss.append(cross_entropy_loss(train_y_hat,train_y))
    valid_y_hat = sigmoid(np.dot(valid_x, w))
    valid_loss.append(cross_entropy_loss(valid_y_hat,valid_y))
    valid_accuracy.append(accuracy_score(predict(valid_y_hat),valid_y))
    if(epoch % 50 == 0):
        print(f"epoch {epoch}: {train_loss[-1]}, {valid_loss[-1]}, {valid_accuracy[-1] * 100 : 0.02f}")

    if epoch > 1000 and np.abs(np.average(valid_loss[-25:]) - np.average(valid_loss[-50:-25])) < 0.000025:
        print(f"stopping at epoch {epoch} - validation loss is not improving significantly")
        break

# %%
