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

random_state = 1234
print(os.getcwd())


# %%
df_train = pd.read_csv("/home/jitghosh/work/pix/ml/mlproject_diabetes/code/gbm/diab_binary/diabetes_binary_train_overundersamp.csv")
df_test = pd.read_csv("/home/jitghosh/work/pix/ml/mlproject_diabetes/code/gbm/diab_binary/diabetes_binary_test_overundersamp.csv")
df_test_orig = pd.read_csv("/home/jitghosh/work/pix/ml/mlproject_diabetes/code/gbm/diab_binary/diabetes_binary_test_orig.csv")
# %%
# feature selection
# use a chi-squared test to select the best features
p_value = 0.05
selector = GenericUnivariateSelect(score_func=chi2, mode="fpr", param=p_value).fit(
    df_train.drop("Diabetes_binary", axis=1), df_train["Diabetes_binary"]
)
diab_binary_features_selected = selector.get_feature_names_out()
df_train = df_train.loc[
    :, np.concatenate((diab_binary_features_selected, ["Diabetes_binary"]))
]

# %%
# split
 
train_X, train_y = df_train.drop("Diabetes_binary", axis=1),df_train["Diabetes_binary"]
test_X, test_y = df_test.drop("Diabetes_binary", axis=1),df_test["Diabetes_binary"]
test_X_orig, test_y_orig = df_test_orig.drop("Diabetes_binary", axis=1),df_test_orig["Diabetes_binary"]

all_features = set(train_X.columns)
numeric_features = ["BMI", "MentHlth", "PhysHlth"]
cat_features = all_features.difference(set(numeric_features))


# %%
# constant training parameters
const_params: dict = {
    "seed": random_state,
    "verbosity": 0,
    "objective": "binary",
    "num_threads": 12,
    "is_unbalance": True,
    "metric": "binary",
    "data_sample_strategy": "goss",
    "num_iterations": 500,
    "learning_rate": 0.01,
    "boosting": "gbdt",
    "device": "gpu",
    "gpu_platform_id": 1,
    "gpu_device_id": 0,
    "num_gpu":2
}


# %%
def callable_eval_logloss(rawpreds, eval_data):
    preds = (rawpreds > 0.5).astype(int)
    y_true = eval_data.get_label()
    return (
        "log_loss",
        log_loss(
            y_true,
            preds,
            normalize = True,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
            labels = [0.0,1.0]
        ),
        False,
    )


def callable_eval_balanced_accuracy(rawpreds, eval_data):
    preds = (rawpreds > 0.5).astype(int)
    y_true = eval_data.get_label()
    return (
        "balanced_accuracy",
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
        "f1",
        f1_score(
            y_true,
            preds,
            pos_label=1.0,
            sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
        ),
        True,
    )


ds_train_X = lgbm.Dataset(
    train_X,
    label=train_y,
    feature_name=all_features,
    categorical_feature=cat_features,
    weight=compute_sample_weight(class_weight="balanced", y=train_y),
)

eval_results = lgbm.cv(
    params=const_params,
    train_set=ds_train_X,
    folds=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
    return_cvbooster=True,
    feval=[
        callable_eval_logloss,
        callable_eval_balanced_accuracy,
        callable_eval_f1_score,
    ],
    callbacks=[lgbm.early_stopping(10)]
)

# %%
print(f"Max f1 : {np.max(eval_results["valid f1-mean"])}, iteration : {np.argmax(eval_results["valid f1-mean"])}")

#%%
fig,ax = plt.subplots(1,1)
ax.plot(range(len(eval_results["valid binary_logloss-mean"])),eval_results["valid binary_logloss-mean"])
plt.show()
# %%
rawpreds = eval_results["cvbooster"].predict(test_X.values,validate_features = True)
preds = (np.max(np.array(rawpreds),axis=0) > 0.5).astype(int)
print(
    f"test_resampled_f1_score: {f1_score(preds,test_y,sample_weight=compute_sample_weight(class_weight="balanced", y=test_y),pos_label=1.0)*100}"
)
# %%
rawpreds = eval_results["cvbooster"].predict(test_X_orig.values,validate_features = True)
preds = (np.max(np.array(rawpreds),axis=0) > 0.5).astype(int)
print(
    f"test_orig_f1_score: {f1_score(preds,test_y_orig,sample_weight=compute_sample_weight(class_weight="balanced", y=test_y_orig),pos_label=1.0)*100}"
)
# %%