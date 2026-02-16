import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix
)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4

# -----------------------------
# Load the 67-subject dataset (PLV included)
# -----------------------------
lat = pd.read_csv("results/latency_subject_level.csv")
spec = pd.read_csv("results/spectral_subject_level.csv")
plv = pd.read_csv("results/plv_theta_subject_level.csv")

df_69 = lat.merge(spec, on=["Subject", "Group"], how="inner")
df = df_69.merge(plv, on=["Subject", "Group"], how="inner")  # 67 subjects

# Label: ASD=1, TD=0
y = (df["Group"].astype(str).str.upper() == "ASD").astype(int).values

SPECTRAL_FEATS = [
    "Delta_RelPower_mean",
    "Theta_RelPower_mean",
    "Alpha_RelPower_mean",
    "Beta_RelPower_mean",
    "Gamma_RelPower_mean",
]
PLV_FEATS = ["PLV_mean_all", "PLV_std_all"]

X = df[SPECTRAL_FEATS + PLV_FEATS].copy()

# -----------------------------
# Metrics helpers
# -----------------------------
def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0

# -----------------------------
# Pipeline + hyperparameter grid
# -----------------------------
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )),
])

# Grid: keep it strong but not insane (good for n=67)
param_grid = {
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.01, 0.1, 1, 10, 100],
}

outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Store outer-fold results
fold_results = []
best_params_list = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner tuning only on training fold
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params_list.append(search.best_params_)

    # Evaluate on held-out outer fold
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    fold_results.append({
        "fold": fold,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "sensitivity": sensitivity(y_test, y_pred),
        "specificity": specificity(y_test, y_pred),
    })

# Summarize
res = pd.DataFrame(fold_results)
print("\n✅ Nested CV (Spectral+PLV, Logistic Regression) Results")
print(res.round(4).to_string(index=False))

print("\nMean ± SD:")
for col in ["roc_auc", "accuracy", "f1", "sensitivity", "specificity"]:
    print(f"{col:12s}: {res[col].mean():.4f} ± {res[col].std():.4f}")

# Show which hyperparams were selected most often
bp = pd.DataFrame(best_params_list)
print("\nBest hyperparameters chosen in each outer fold:")
print(bp)

print("\nMost common best params:")
print(bp.value_counts())
