# %%
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    log_loss,
)
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.utils import compute_sample_weight
import matplotlib.pyplot as plt
import os, pathlib, pickle
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.under_sampling import (
    NearMiss,
    EditedNearestNeighbours,
    TomekLinks,
    RandomUnderSampler,
)
from imblearn.over_sampling import SMOTENC


random_state = 1234
print(os.getcwd())


# %%

# %%
df_diab_binary = pd.read_csv(
    "/home/jitghosh/work/pix/ml/mlproject_diabetes/data/diabetes_binary_health_indicators_BRFSS2015.csv"
)
# %%
# feature selection
# use a chi-squared test to select the best features
p_value = 0.05
selector = GenericUnivariateSelect(score_func=chi2, mode="fpr", param=p_value).fit(
    df_diab_binary.drop("Diabetes_binary", axis=1), df_diab_binary["Diabetes_binary"]
)
diab_binary_features_selected = selector.get_feature_names_out()
df_diab_binary = df_diab_binary.loc[
    :, np.concatenate((diab_binary_features_selected, ["Diabetes_binary"]))
]

# %%
# split

train_X, test_X, train_y, test_y = train_test_split(
    df_diab_binary.drop("Diabetes_binary", axis=1),
    df_diab_binary["Diabetes_binary"],
    test_size=0.2,
    shuffle=True,
    stratify=df_diab_binary["Diabetes_binary"],
)


all_features = set(train_X.columns)
numeric_features = ["BMI", "MentHlth", "PhysHlth"]
cat_features = all_features.difference(set(numeric_features))
cat_features_idx = [
    idx for idx, name in enumerate(train_X.columns) if name in cat_features
]


# %%
# %%
def callable_eval_logloss(rawpreds, eval_data):
    preds = (rawpreds > 0.5).astype(int)
    y_true = eval_data.get_label()
    return (
        "weighted_log_loss",
        log_loss(
            y_true,
            preds,
            normalize=True,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
            labels=[0.0, 1.0],
        ),
        False,
    )


def callable_eval_balanced_accuracy(rawpreds, eval_data):
    preds = (rawpreds > 0.5).astype(int)
    y_true = eval_data.get_label()
    return (
        "weighted_balanced_accuracy",
        balanced_accuracy_score(
            y_true,
            preds,
            adjusted=False,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
        ),
        True,
    )


def callable_eval_f1_score(rawpreds, eval_data):
    preds = (rawpreds > 0.5).astype(int)
    y_true = eval_data.get_label()
    return (
        "weighted_f1",
        f1_score(
            y_true,
            preds,
            pos_label=1.0,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
        ),
        True,
    )


# %%

sampler_search = [
    (
        "editednn_nn10_kindselall",
        EditedNearestNeighbours(kind_sel="all", n_neighbors=10),
    ),
    (
        "editednn_nn10_kindselmode",
        EditedNearestNeighbours(kind_sel="mode", n_neighbors=10),
    ),
    (
        "editednn_nn10_kindselall",
        EditedNearestNeighbours(kind_sel="all", n_neighbors=15),
    ),
    (
        "editednn_nn10_kindselmode",
        EditedNearestNeighbours(kind_sel="mode", n_neighbors=15),
    ),
    ("tomeklinks", TomekLinks()),
    (
        "random_undersampler_withrep",
        RandomUnderSampler(random_state=random_state, replacement=True),
    ),
    (
        "random_undersampler_withoutrep",
        RandomUnderSampler(random_state=random_state, replacement=False),
    ),
    ("nearmiss_v3", NearMiss(version=3, n_neighbors=10, n_neighbors_ver3=10)),
]
# constant training parameters
const_params: dict = {
    "seed": random_state,
    "verbosity": 0,
    "objective": "binary",
    "num_threads": 12,
    "is_unbalance": True,
    "metric": "cross_entropy",
    "data_sample_strategy": "goss",
    "num_iterations": 500,
    "learning_rate": 0.01,
    "boosting": "dart",
    "device": "gpu",
    "gpu_platform_id": 1,
    "gpu_device_id": 0,
    "num_gpu": 2,
}
skfold = StratifiedKFold()
sampler_results = []
for sampler_idx, sampler_param in enumerate(sampler_search):
    print(f"Checking {sampler_param}...")
    kfold_results = []
    for idx, (train_idx, valid_idx) in enumerate(skfold.split(train_X, train_y)):
        print(f"Fold {idx}...")
        resamp_train_X, resamp_train_y = sampler_param[1].fit_resample(
            train_X.iloc[train_idx, :], train_y.iloc[train_idx]
        )
        valid_X, valid_y = train_X.iloc[valid_idx, :], train_y.iloc[valid_idx]

        ds_train_X = lgbm.Dataset(
            resamp_train_X,
            label=resamp_train_y,
            feature_name=all_features,
            categorical_feature=cat_features,
            weight=compute_sample_weight(class_weight="balanced", y=resamp_train_y),
        )

        ds_valid_X = lgbm.Dataset(
            valid_X,
            label=valid_y,
            feature_name=all_features,
            categorical_feature=cat_features,
            weight=compute_sample_weight(class_weight="balanced", y=valid_y),
        )

        eval_results = {}

        booster = lgbm.train(
            params=const_params,
            train_set=ds_train_X,
            valid_sets=[ds_valid_X],
            valid_names=["valid"],
            feval=[callable_eval_f1_score],
            callbacks=[lgbm.early_stopping(100), lgbm.record_evaluation(eval_results)],
        )
        min_loss_iter = np.argmin(eval_results["valid"]["cross_entropy"])
        kfold_results.append(
            [
                np.min(eval_results["valid"]["cross_entropy"]),
                eval_results["valid"]["weighted_f1"],
            ]
        )

    sampler_results.append(
        [
            sampler_idx,
            np.mean([itm[0] for itm in kfold_results]),
            np.mean([itm[1] for itm in kfold_results]),
        ]
    )


# %%
sampler_results_sorted_by_f1 = sorted(sampler_results, key = lambda arr: arr[2],reverse=True)
best_sampler = sampler_search[sampler_results_sorted_by_f1[0][0]]
print(
    f"Mean weighted f1 for {best_sampler}: {sampler_results_sorted_by_f1[0][2]}"
)
#%%

resamp_train_X, resamp_train_y = best_sampler[1].fit_resample(
            train_X, train_y
        )
ds_train_X = lgbm.Dataset(
            resamp_train_X,
            label=resamp_train_y,
            feature_name=all_features,
            categorical_feature=cat_features,
            weight=compute_sample_weight(class_weight="balanced", y=resamp_train_y),
        )
booster = lgbm.train(
            params=const_params,
            train_set=ds_train_X,
        )

preds = booster.predict(test_X, num_iteration=min_loss_iter)
print(
    f"f1: {f1_score(
            test_y,
            (preds > 0.5).astype(int),
            pos_label=1.0,
            sample_weight=compute_sample_weight(class_weight="balanced", y=test_y),
        )}"
)

# %%
