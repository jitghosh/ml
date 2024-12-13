# %%
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight

# %%
diabetes_bin = pd.read_csv(
    "/home/jitghosh/work/pix/ml/mlproject_diabetes/data/diabetes_binary_health_indicators_BRFSS2015.csv"
)
cols = diabetes_bin.columns.tolist()
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    diabetes_bin.iloc[:, 1:],
    diabetes_bin["Diabetes_binary"],
    test_size=0.15,
    random_state=777,
    shuffle=True,
    stratify=diabetes_bin["Diabetes_binary"],
)
y_train_bin = y_train_bin.values
y_test_bin = y_test_bin.values
# feature selection for features w least false positives
fpr = SelectFpr(chi2, alpha=0.01)
X_train_bin = fpr.fit_transform(X_train_bin, y_train_bin)
features_after_fpr = fpr.get_feature_names_out()



# %%
# new
def f1(y_pred, y_test_bin):
    y_true = y_test_bin.get_label()
    y_pred_bin = (y_pred > 0.5).astype(int)
    f1 = f1_score(
        y_true,
        y_pred_bin,
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
        pos_label=1.0,
    )
    return ("f1", f1, True)


def balanced_acc(y_pred, y_test_bin):
    y_true = y_test_bin.get_label()
    y_pred_bin = (y_pred > 0.5).astype(int)
    acc = balanced_accuracy_score(
        y_true,
        y_pred_bin,
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_true),
    )
    return ("balanced accuracy score", acc, True)


params_dict = {
    "objective": "binary",
    "learning_rate": 0.01,
    "seed": 777,
    "is_unbalance": True,
    "metric": "binary_logloss",
    "data_sample_strategy": "goss",
    "boosting": "dart",
    "device": "gpu",
    "gpu_platform_id": 1,
    "gpu_device_id": 0,
    "num_gpu": 2,
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
eval_results = []

for train_idx, valid_idx in kf.split(X_train_bin, y_train_bin):
    X_train, X_valid = X_train_bin[train_idx], X_train_bin[valid_idx]
    y_train, y_valid = y_train_bin[train_idx], y_train_bin[valid_idx]

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=[
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
    )
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=[
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
    )

    evals = {}

    model = lgb.train(
        params=params_dict,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        feval=[f1, balanced_acc],
        num_boost_round=500,
        callbacks=[lgb.record_evaluation(evals)],
    )

    eval_results.append(evals)

# talk about adding sample weight in f1 and adding balanced accuracy score for paper
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(range(len(eval_results[0]["valid"]["f1"])), eval_results[0]["valid"]["f1"])
plt.show()
# %%