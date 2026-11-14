"""
Train every model once and save them to models/hcahps_models.joblib.

Run:  python train_models.py
The app loads the saved file, so it never retrains while you present.
Same specification as the unified notebook: six predictors, 60/40 split,
random_state=42 (stratified for classification), tuning with 5-fold CV on the training data only.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "hcahps_model_ready.csv"
MODEL_FILE = HERE / "models" / "hcahps_models.joblib"

SEED = 42
THRESHOLD = 70
PREDICTORS = ["nurse_communication", "doctor_communication", "medicine_explanation",
              "discharge_information", "room_cleanliness", "night_quietness"]
TARGET = "target_recommend_pct"
HIGH = "target_high_recommend"
TIER_ORDER = ["Higher", "Moderate", "Lower"]


def load_data(path=DATA_FILE):
    # Read the model-ready file (facility IDs as text so leading zeros survive)
    df = pd.read_csv(path, dtype={"facility_id": str})
    if HIGH not in df.columns:
        df[HIGH] = (df[TARGET] >= THRESHOLD).astype(int)
    return df.dropna(subset=PREDICTORS + [TARGET]).reset_index(drop=True)


def train_all(df):
    # Train every model and return them with the ids of the hospitals held back for testing
    X, y, y_class = df[PREDICTORS], df[TARGET], df[HIGH].astype(int)
    ids = df["facility_id"] if "facility_id" in df.columns else pd.Series(df.index.astype(str))

    # Predicting the percentage: unstratified 60/40 split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.40, random_state=SEED)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    linear = LinearRegression().fit(X_tr, y_tr)
    tree_reg = GridSearchCV(DecisionTreeRegressor(random_state=SEED),
                            {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [5, 10, 20, 30]},
                            scoring="neg_root_mean_squared_error", cv=folds, n_jobs=-1).fit(X_tr, y_tr)

    # Flagging below 70%: stratified 60/40 split
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_class, test_size=0.40, random_state=SEED, stratify=y_class)
    sfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    logistic = GridSearchCV(Pipeline([("scaler", StandardScaler()),
                                      ("logistic", LogisticRegression(max_iter=2000, random_state=SEED))]),
                            {"logistic__C": [0.01, 0.1, 1, 10, 100]}, scoring="f1", cv=sfolds, n_jobs=-1).fit(Xc_tr, yc_tr)
    class_tree = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                              {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [5, 10, 20, 30]},
                              scoring="f1", cv=sfolds, n_jobs=-1).fit(Xc_tr, yc_tr)
    forest = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=-1).fit(Xc_tr, yc_tr)

    # Peer tiers: PCA on standardised scores, then K-means with K = 3 (recommendation not used)
    pca_scaler = StandardScaler().fit(X)
    pca = PCA(n_components=2).fit(pca_scaler.transform(X))
    scores = pca.transform(pca_scaler.transform(X))
    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(scores)
    rank = df.groupby(kmeans.labels_)[TARGET].mean().sort_values(ascending=False).index
    silhouette = {k: silhouette_score(scores, KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(scores))
                  for k in range(2, 9)}

    return {
        "linear": linear, "tree_reg": tree_reg.best_estimator_, "tree_reg_params": tree_reg.best_params_,
        "logistic": logistic.best_estimator_, "logistic_params": logistic.best_params_,
        "class_tree": class_tree.best_estimator_, "class_tree_params": class_tree.best_params_,
        "forest": forest,
        "pca_scaler": pca_scaler, "pca": pca, "kmeans": kmeans,
        "tier_names": {int(k): name for k, name in zip(rank, TIER_ORDER)},
        "silhouette": silhouette, "pca_variance": PCA().fit(pca_scaler.transform(X)).explained_variance_ratio_,
        "reg_test_ids": ids.loc[X_te.index].tolist(), "cls_test_ids": ids.loc[Xc_te.index].tolist(),
        "n_train": len(X_tr), "n_test": len(X_te),
        "train_sd": Xc_tr.std(),
        "seed": SEED, "predictors": PREDICTORS,
    }


if __name__ == "__main__":
    data = load_data()
    bundle = train_all(data)
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(f"Trained on {bundle['n_train']:,} hospitals, held back {bundle['n_test']:,}. Saved to {MODEL_FILE}")
