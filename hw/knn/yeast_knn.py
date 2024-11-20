# %%
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import StandardScaler
random_state = 1234


# %%
class KNNClassfier:
    def __init__(self, n_neighbors=7):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, test_X):
        return np.apply_along_axis(self.find_majority_class,arr=test_X,axis=1)

    def find_majority_class(self, target_x):
        distances = np.apply_along_axis(
            self.calc_distance, axis=1, arr=self.X, X_to=target_x
        )
        y_dist = pd.DataFrame({"dist": distances, "y": self.y})
        sorted_values = y_dist.sort_values(by=["dist"], ascending=True).values
        highest_count = Counter(sorted_values[0 : self.n_neighbors, 1]).most_common(1)
        return highest_count[0][0]

    def calc_distance(self, X_from, X_to):
        return linalg.norm(X_to - X_from)


# %%
col_names = [
    "seq_name",
    "mcg",
    "gvh",
    "alm",
    "mit",
    "er1",
    "pox",
    "vac",
    "nuc",
    "cls_name",
]

# %%
with open("C:\pix\ml\hw\knn\yeast.data", "r") as yeast_file, open(
    "C:\pix\ml\hw\knn\yeast2.data", "w"
) as yeast2_file:
    lns = []
    ln = yeast_file.readline()
    while ln != "":
        lns.append(
            ",".join([itm.strip() for itm in ln.split(" ") if itm.strip() != ""]) + "\n"
        )
        ln = yeast_file.readline()
    yeast2_file.writelines(lns)

# %%
df_yeast = pd.read_csv(
    "C:\pix\ml\hw\knn\yeast2.data", header=None, names=col_names, sep=","
).drop("seq_name",axis=1)

#df_yeast["seq_name"] = LabelEncoder().fit_transform(df_yeast["seq_name"])

train_X, test_X, train_y, test_y = train_test_split(
    df_yeast.drop("cls_name", axis=1),
    df_yeast["cls_name"],
    shuffle=True,
    stratify=df_yeast["cls_name"],
    test_size=0.15,
    random_state=random_state
)

scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X) 
# %%
clf = KNNClassfier(7).fit(train_X, train_y)
preds = clf.predict(test_X)
accuracy_score(test_y,preds)

# %%
