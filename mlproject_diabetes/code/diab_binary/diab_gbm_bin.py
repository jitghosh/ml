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
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.utils import compute_sample_weight
import matplotlib.pyplot as plt
import os, pathlib, pickle

random_state = 1234
print(os.getcwd())


# %%
df_diab_binary = pd.read_csv("/home/jitghosh/work/pix/ml/mlproject_diabetes/data/diabetes_binary_health_indicators_BRFSS2015.csv"
)
# %%
# feature selection
# use a chi-squared test to select the best features
p_value = 0.05
selector = SelectFpr(score_func=chi2, alpha=p_value).fit(
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
    random_state=random_state,
    shuffle=True,
    stratify=df_diab_binary["Diabetes_binary"],
)
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
    "boosting": "dart",
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
    categorical_feature=cat_features
)
#%%
evals = {}
booster = lgbm.train(
    params=const_params,
    train_set=ds_train_X,
    valid_sets=[ds_train_X],
    feval=[
        callable_eval_logloss,
        callable_eval_balanced_accuracy,
        callable_eval_f1_score,
    ],
    callbacks=[lgbm.record_evaluation(evals)]
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
#%%
