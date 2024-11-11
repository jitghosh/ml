# %%
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
import matplotlib.pyplot as plt

random_state = 1234

# %%
df_diab_012 = pd.read_csv("..\..\data\diabetes_012_health_indicators_BRFSS2015.csv")
df_diab_012.head()
# %%
# feature selection
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
# training parameters
const_params: dict = {
    "seed": random_state,
    "objective": "multiclass",
    "num_threads": 12,
    "num_classes": 3,
    "is_unbalance": True,
    "metric": "softmax",
}
search_param_grid: ParameterGrid = ParameterGrid(
    {
        "data_sample_strategy": ["goss", "bagging"],
        "num_rounds": range(100, 500, 100),
        "learning_rate": [0.01, 0.001, 0.0001],
        "boosting": ["dart", "gbdt"],
    }
)
k_fold_splits = 5
# %%

param_scores = []
for params in search_param_grid:
    print(f"Using {params}")
    skfold = StratifiedKFold(
        n_splits=k_fold_splits, shuffle=True, random_state=random_state
    )
    validation_scores = []
    for idx, (train_idx, valid_idx) in enumerate(skfold.split(train_X, train_y)):
        ds_train_X = lgbm.Dataset(
            train_X.iloc[train_idx, :],
            label=train_y.iloc[train_idx],
            feature_name=all_features,
            categorical_feature=cat_features,
        )
        booster = lgbm.train(const_params | params, ds_train_X)
        validation_preds = booster.predict(train_X.iloc[valid_idx, :])

        validation_scores.append(f1_score(
            np.argmax(validation_preds, axis=1), train_y.iloc[valid_idx], pos_label=range(3), average="weighted"
        ))
    param_scores.append((np.mean(validation_scores),params))
#%%
best_param = sorted(param_scores,key=lambda tup:tup[0],reverse=True)[0][1]
print(best_param)
# %%
