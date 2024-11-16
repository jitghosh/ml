# %%
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.utils import compute_sample_weight, compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import matplotlib.pyplot as plt
import os, pathlib, pickle

random_state = 1234


# %%
df_diab_train_012 = pd.read_csv(
    "C:/pix/ml/mlproject_diabetes/data/diabetes_012_train_resamp.csv"
    if os.uname().sysname != "Linux"
    else "/home/jitghosh/work/pix/ml/mlproject_diabetes/code/gbm/diabetes_012_train_resamp.csv"
)
df_diab_test_012 = pd.read_csv(
    "C:/pix/ml/mlproject_diabetes/data/diabetes_012_test_resamp.csv"
    if os.uname().sysname != "Linux"
    else "/home/jitghosh/work/pix/ml/mlproject_diabetes/code/gbm/diabetes_012_test_resamp.csv"
)

# %%
# feature selection
# use a chi-squared test to select the best features
p_value = 0.05
selector = GenericUnivariateSelect(score_func=chi2, mode="fpr", param=p_value).fit(
    df_diab_train_012.drop("Diabetes_012", axis=1), df_diab_train_012["Diabetes_012"]
)
diab_012_features_selected = selector.get_feature_names_out()
df_diab_train_012 = df_diab_train_012.loc[
    :, np.concatenate((diab_012_features_selected, ["Diabetes_012"]))
]
df_diab_test_012 = df_diab_test_012.loc[
    :, np.concatenate((diab_012_features_selected, ["Diabetes_012"]))
]


# split
(train_X, test_X, train_y, test_y) = (
    df_diab_train_012.drop("Diabetes_012", axis=1),
    df_diab_test_012.drop("Diabetes_012", axis=1),
    df_diab_train_012["Diabetes_012"],
    df_diab_test_012["Diabetes_012"],
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
    # "is_unbalance": True,
    "metric": "softmax",
    "learning_rate": 0.01,
    # "early_stopping_rounds": 5,
    # "early_stopping_min_delta": 0.0025
}
# parameters to search for using grid search
search_param_grid: ParameterGrid = ParameterGrid(
    {
        "data_sample_strategy": ["goss", "bagging"],
        "num_iterations": range(200,1200,200),
        "boosting": ["dart","gbdt"],
        # "max_depth": [8,16,24,32,-1],
        # "min_samples_leaf": [4,8,12,20],
        # "feature_fraction": [1.0,0.8]
    }
)


# %%

k_fold_splits = 5
param_search_results = []
# hyper parameter search
for params in search_param_grid:
    print(f"Using {params}")
    skfold = StratifiedKFold(
        n_splits=k_fold_splits, shuffle=True, random_state=random_state
    )

    validation_loss = []
    validation_f1_score = np.empty((0, 3))
    for idx, (train_idx, valid_idx) in enumerate(skfold.split(train_X, train_y)):
        fold_validation_loss = {}
        ds_train_X = lgbm.Dataset(
            train_X.iloc[train_idx, :],
            label=train_y.iloc[train_idx],
            feature_name=all_features,
            categorical_feature=cat_features,
            weight=compute_sample_weight(
                class_weight="balanced", y=train_y.iloc[train_idx]
            ),
        )
        ds_valid_X = lgbm.Dataset(
            train_X.iloc[valid_idx, :],
            label=train_y.iloc[valid_idx],
            feature_name=all_features,
            categorical_feature=cat_features,
            weight=compute_sample_weight(
                class_weight="balanced", y=train_y.iloc[valid_idx]
            ),
        )

        booster = lgbm.train(
            const_params | params,
            ds_train_X,
            valid_sets=[ds_valid_X],
            valid_names=["validation"],
            callbacks=[lgbm.record_evaluation(fold_validation_loss)],
        )

        validation_loss.append(fold_validation_loss["validation"]["multi_logloss"][-1])
        raw_preds = booster.predict(train_X.iloc[valid_idx, :].values)
        preds = np.apply_along_axis(np.argmax, arr=raw_preds, axis=1)
        validation_f1_score = np.vstack(
            (
                validation_f1_score,
                f1_score(
                    preds,
                    train_y.iloc[valid_idx],
                    average=None,
                    sample_weight=compute_sample_weight(
                        class_weight="balanced", y=train_y.iloc[valid_idx]
                    ),
                ).reshape(1, 3),
            )
        )

    param_search_results.append(
        (np.mean(validation_loss), np.mean(validation_f1_score, axis=0), params)
    )
    print(f"Validation loop result: {param_search_results[-1]}")

    param_search_results_sorted_by_loss = sorted(
        param_search_results, key=lambda tup: tup[0]
    )
    best_params_by_loss = param_search_results_sorted_by_loss[0][2]
    print(
        f"best_params by loss : {best_params_by_loss}, log loss: {param_search_results_sorted_by_loss[0][0]}, mean f1_score by class: {param_search_results_sorted_by_loss[0][1]}"
    )

# %%
with open("./diab_012_resamp_bestparam_by_loss.pkl", "wb") as best_params_file:
    pickle.dump(best_params_by_loss, best_params_file)
# %%

if pathlib.Path("./diab_012_resamp_bestparam_by_loss.pkl").is_file():
    with open("./diab_012_resamp_bestparam_by_loss.pkl", "rb") as best_params_file:
        best_params = pickle.load(best_params_file)
        print(best_params)
        ds_train_X = lgbm.Dataset(
            train_X,
            label=train_y,
            feature_name=all_features,
            categorical_feature=cat_features,
            weight=compute_sample_weight(
                        class_weight="balanced", y=train_y)
        )

        best_booster = lgbm.train(const_params | best_params, ds_train_X)
        preds = best_booster.predict(test_X.values)
        print(
            f"Test F1: {f1_score(np.apply_along_axis(np.argmax,arr=preds,axis=1),test_y,average=None,sample_weight=compute_sample_weight(
                        class_weight="balanced", y=test_y))*100}"
        )
# %%
