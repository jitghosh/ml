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



# %%
df_diab_012 = pd.read_csv("C:/pix/ml/mlproject_diabetes/data/diabetes_012_health_indicators_BRFSS2015.csv")
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
    #"early_stopping_rounds": 5,
    #"early_stopping_min_delta": 0.0025
}
# parameters to search for using grid search
search_param_grid: ParameterGrid = ParameterGrid(
    {
        "data_sample_strategy": ["goss", "bagging"],
        "num_iterations": range(100, 600, 100),
        "learning_rate": [0.01, 0.001, 0.0001],
        "boosting": ["dart", "gbdt"],
        #"max_depth": [8,16,24,32,-1],
        #"min_samples_leaf": [4,8,12,20],
        #"feature_fraction": [1.0,0.8]
    }
)

# %%
if not pathlib.Path("./diab_012_bestparam.pkl").is_file():
    k_fold_splits = 5
    param_validation_loss = []
    # hyper parameter search
    for params in search_param_grid:
        print(f"Using {params}")
        skfold = StratifiedKFold(
            n_splits=k_fold_splits, shuffle=True, random_state=random_state
        )

        validation_loss = []
        for idx, (train_idx, valid_idx) in enumerate(skfold.split(train_X, train_y)):
            fold_validation_loss = {}
            ds_train_X = lgbm.Dataset(
                train_X.iloc[train_idx, :],
                label=train_y.iloc[train_idx],
                feature_name=all_features,
                categorical_feature=cat_features,
            )
            ds_valid_X = lgbm.Dataset(
                train_X.iloc[valid_idx, :],
                label=train_y.iloc[valid_idx],
                feature_name=all_features,
                categorical_feature=cat_features,
            )

            booster = lgbm.train(const_params | params, ds_train_X, valid_sets=[ds_valid_X],valid_names=["validation"], callbacks=[lgbm.record_evaluation(fold_validation_loss)])
            
            validation_loss.append(fold_validation_loss["validation"]["multi_logloss"][-1])
            
        param_validation_loss.append((np.mean(validation_loss),params))
        print(f"Validation loss: {param_validation_loss[-1]}")

    param_validation_loss = sorted(param_validation_loss,key=lambda tup:tup[0])
    best_params = param_validation_loss[0][1]
    print(f"best_params : {best_params}, log loss : {param_validation_loss[0][0]}")
#%%
    with open("./diab_012_bestparam.pkl","wb+") as best_params_file:
        pickle.dump(best_params,best_params_file)
#%%

if pathlib.Path("./diab_012_bestparam.pkl").is_file():
    with open("./diab_012_bestparam.pkl","rb+") as best_params_file:
        best_params = pickle.load(best_params_file)
else:
    raise "Need to rerun hyper parameter search"
#%%

ds_train_X = lgbm.Dataset(
            train_X.iloc[train_idx, :],
            label=train_y.iloc[train_idx],
            feature_name=all_features,
            categorical_feature=cat_features,
        )
best_booster = lgbm.train(const_params | best_params, ds_train_X)
# %%
