# %%
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import matplotlib.pyplot as plt
import os, pathlib, pickle

random_state = 1234
print(os.getcwd())


# %%
df_diab_012 = pd.read_csv("C:/pix/ml/mlproject_diabetes/data/diabetes_012_health_indicators_BRFSS2015.csv" if os.uname().sysname != "Linux" else "../../data/diabetes_012_health_indicators_BRFSS2015.csv")
# %%
# feature selection
# use a chi-squared test to select the best features
p_value = 0.05
selector = GenericUnivariateSelect(score_func=chi2, mode="fpr", param=p_value).fit(
    df_diab_012.drop("Diabetes_012", axis=1), df_diab_012["Diabetes_012"]
)
diab_012_features_selected = selector.get_feature_names_out()
df_diab_012 = df_diab_012.loc[
    :, np.concatenate((diab_012_features_selected, ["Diabetes_012"]))
]

# %%
# split
train_X, test_X, train_y, test_y = train_test_split(
    df_diab_012.drop("Diabetes_012", axis=1),
    df_diab_012["Diabetes_012"],
    test_size=0.2,
    random_state=random_state,
    shuffle=True,
    stratify=df_diab_012["Diabetes_012"],
)
all_features = set(train_X.columns)
numeric_features = ["BMI", "MentHlth", "PhysHlth"]
cat_features = all_features.difference(set(numeric_features))


# %%
# constant training parameters
const_params: dict = {
    "seed": random_state,
    "verbosity": 0,
    "objective": "multiclass",
    "num_threads": 12,
    "num_classes": 3,
    "is_unbalance": True,
    "metric": "softmax",
    "data_sample_strategy": "goss",
    "num_iterations": 800,
    "learning_rate":0.01,
    "boosting":"dart"
    #"early_stopping_min_delta": 0.0025
}


# %%

ds_train_X = lgbm.Dataset(
    train_X,
    label=train_y,
    feature_name=all_features,
    categorical_feature=cat_features
)

booster = lgbm.train(const_params, 
                    ds_train_X)
            
#%%            
preds = booster.predict(train_X.values,num_iteration=799)
print(f"train_f1_score_perclass: {f1_score(np.apply_along_axis(np.argmax,arr=preds,axis=1),train_y,average=None)*100}")
#%%
booster.best_iteration
# %%
