#%%
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as pyplt
import pandas as pd



# %%


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

#%%
mu = df_train_x.mean(axis=0)
std = df_train_x.std(axis=0)
df_train_x = (df_train_x - mu)/std
df_test_x = (df_test_x - mu)/std
train_x = df_train_x.values
train_y = df_train_y.values.reshape(df_train_y.shape[0],1)

#%%
def calc_weights(df_train_x, df_train_y):
    df_train_x_C1 = df_train_x.loc[(df_train_y == 1),:]
    df_train_x_C2 = df_train_x.loc[(df_train_y == 0),:]
    p_C1 = np.sum(df_train_y == 1)/df_train_y.shape[0]
    p_C2 = np.sum(df_train_y == 0)/df_train_y.shape[0]
    mu1 = np.expand_dims(np.mean(df_train_x_C1,axis=0),1)
    mu2 = np.expand_dims(np.mean(df_train_x_C2,axis=0),1)
    s1 = ((df_train_x_C1 - mu1.T).T @ (df_train_x_C1 - mu1.T))/np.sum(df_train_y == 1)
    s2 = ((df_train_x_C2 - mu2.T).T @ (df_train_x_C2 - mu2.T))/np.sum(df_train_y == 0)
    cov_mat = s1 * p_C1 + s2 * p_C2
    w0 = (-0.5 *(mu1.T@cov_mat@mu1) + 0.5 * (mu2.T@cov_mat@mu2) + np.log(p_C1/p_C2))[0][0]
    w = np.linalg.inv(cov_mat) @ (mu1 - mu2)

    return (w0,w)

def y_hats(x,w0,w):
    return x@w + w0

# %%
kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=1234)
weights = []
fold_accuracy = []
for fold_idx_train, fold_idx_valid in kfold.split(train_x,train_y):
    train_kfolds_X = train_x[fold_idx_train,:]
    valid_kfolds_X = train_x[fold_idx_valid,:]
    train_kfolds_y = train_y[fold_idx_train,:]
    valid_kfolds_y = train_y[fold_idx_valid,:]

    w0,w = calc_weights(df_train_x,df_train_y)
    
    y_preds = [0 if itm <= 0.0 else 1 for itm in y_hats(valid_kfolds_X,w0,w)]

    weights.append((w0,w))
    fold_accuracy.append(accuracy_score(y_preds,valid_kfolds_y))
final_weights = weights[np.argmax(fold_accuracy)]

#%%
test_y_hats = y_hats(df_test_x,final_weights[0],final_weights[1])               
test_y_preds = [0 if itm <= 0.0 else 1 for itm in test_y_hats.values]
#%%
accuracy_score(test_y_preds,df_test_y.values)

# %%
