#%%
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight

#%%
diabetes_bin = pd.read_csv('/home/jitghosh/work/pix/ml/mlproject_diabetes/data/diabetes_binary_health_indicators_BRFSS2015.csv')
cols = diabetes_bin.columns.tolist()
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(diabetes_bin.iloc[:,1:], 
                                                    diabetes_bin['Diabetes_binary'], 
                                                    test_size= 0.20, random_state= 777, 
                                                    shuffle = True, stratify= diabetes_bin['Diabetes_binary'])
y_train_bin = y_train_bin.values
y_test_bin = y_test_bin.values
# feature selection for features w least false positives 
fpr = SelectFpr(chi2, alpha=0.01)
X_train_bin = fpr.fit_transform(X_train_bin, y_train_bin)
features_after_fpr = fpr.get_feature_names_out()

# %%
#new 
def f1(y_pred, y_test_bin):
    y_true = y_test_bin.get_label()
    y_pred_bin = (y_pred > 0.5).astype(int)  
    f1 = f1_score(y_true, y_pred_bin, 
                  sample_weight= compute_sample_weight(class_weight='balanced', y = y_true), pos_label= 1.0)
    return ('f1', f1, True)  

def balanced_acc(y_pred, y_test_bin):
    y_true = y_test_bin.get_label()
    y_pred_bin = (y_pred > 0.5).astype(int)  
    acc = balanced_accuracy_score(y_true, y_pred_bin, adjusted=False,
                  sample_weight= compute_sample_weight(class_weight='balanced', y = y_true))
    return ('balanced accuracy score', acc, True) 

params_dict = {'objective':'binary', 
               'learning_rate': 0.01, 
               'seed': 777, 
               'is_unbalance': True, 
               'metric':'binary_logloss',
               'data_sample_strategy':'goss',
               'boosting':'dart'}

 
train_data = lgb.Dataset(X_train_bin, label=y_train_bin)

evals = lgb.cv(params = params_dict,
                    train_set=train_data, 
                    folds=StratifiedKFold(n_splits=5, shuffle=True, random_state=777),
                    feval=[f1, balanced_acc],
                    num_boost_round= 500, 
                    return_cvbooster=True,
                    categorical_feature= [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 
                    callbacks= [lgb.early_stopping(30)])

# talk about adding sample weight in f1 and adding balanced accuracy score for paper 
# %%
