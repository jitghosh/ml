#%%
import pandas as pd 
import numpy as np 
from matplotlib import pyplot
import seaborn as sns 

import json 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, RocCurveDisplay 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFpr, chi2
# %%
diabetes_bin = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
diabetes_bin.head()
# %%
#feature selection 
#split data first and do feature selection solely on split
# create select fpr, fit transform with data 
X_train, X_test, y_train, y_test = train_test_split(diabetes_bin.iloc[:,1:], diabetes_bin['Diabetes_binary'], test_size= 0.20, random_state= 777, shuffle = True, stratify= diabetes_bin['Diabetes_binary'])
X_train = SelectFpr(chi2, alpha=0.01).fit_transform(X_train, y_train)
# %%
# Random Forest (random_state = 777, oob_score = True)
# grid search (n_jobs = 4, cv = 5, return_train_score = True, scoring = ['f1_weighted','precision_weighted','recall_weighted','roc_auc'], refit = 'f1_weighted')
params = {'n_estimators':np.arange(1000,4000,1000),'max_depth':[13,14,15,16,None],'max_features':['sqrt',0.5]}

# f1 score
# %%
with open('rf_baseline_results.json','w') as results:
    json.dump(GRIDNAME.cv_results_,results)
