import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, confusion_matrix

# -----------------------------
# Config
# -----------------------------
OUT_DIR = Path("results")
RANDOM_STATE = 42
N_SPLITS = 5

# -----------------------------
# Load subject-level files
# -----------------------------
lat = pd.read_csv("results/latency_subject_level.csv")
spec = pd.read_csv("results/spectral_subject_level.csv")
plv = pd.read_csv("results/plv_theta_subject_level.csv")

# -----------------------------
# Merge datasets
# -----------------------------

# 69-subject dataset (latency + spectral)
df_69 = lat.merge(spec, on=["Subject", "Group"], how="inner")

# 67-subject dataset (all features)
df_67 = df_69.merge(plv, on=["Subject", "Group"], how="inner")

print("Dataset sizes:")
print("69-subject dataset:", df_69.shape)
print("67-subject dataset:", df_67.shape)

# -----------------------------
# Feature definitions
# -----------------------------
LATENCY_FEATS = ["M50_latency_ms_mean", "M100_latency_ms_mean"]

SPECTRAL_FEATS = [
    "Delta_RelPower_mean",
    "Theta_RelPower_mean",
    "Alpha_RelPower_mean",
    "Beta_RelPower_mean",
    "Gamma_RelPower_mean",
]

PLV_FEATS = ["PLV_mean_all", "PLV_std_all"]

# -----------------------------
# Models
# -----------------------------
models = {
    "LogReg": LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
        max_iter=2000,
    ),
    "SVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

# -----------------------------
# Custom metrics
# -----------------------------
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "sensitivity": make_scorer(sensitivity),
    "specificity": make_scorer(specificity),
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results = []

def evaluate_model(df, feature_list, feature_set_name):
    y = (df["Group"].str.upper() == "ASD").astype(int).values
    X = df[feature_list]

    for model_name, clf in models.items():
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        row = {
            "feature_set": feature_set_name,
            "model": model_name,
            "n_subjects": len(df),
            "n_features": len(feature_list),
        }

        for key in scores:
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                row[f"{metric}_mean"] = np.mean(scores[key])
                row[f"{metric}_std"] = np.std(scores[key])

        results.append(row)

# -----------------------------
# Evaluate models
# -----------------------------

# 69-subject models
evaluate_model(df_69, LATENCY_FEATS, "Latency_only")
evaluate_model(df_69, SPECTRAL_FEATS, "Spectral_only")
evaluate_model(df_69, LATENCY_FEATS + SPECTRAL_FEATS, "Latency+Spectral")

# 67-subject models (PLV included)
evaluate_model(df_67, PLV_FEATS, "PLV_only")
evaluate_model(df_67, SPECTRAL_FEATS + PLV_FEATS, "Spectral+PLV")
evaluate_model(df_67, LATENCY_FEATS + SPECTRAL_FEATS + PLV_FEATS, "All_features")

# -----------------------------
# Save results
# -----------------------------
res_df = pd.DataFrame(results).sort_values(
    by=["roc_auc_mean", "accuracy_mean"],
    ascending=False
)

res_df.to_csv("results/model_cv_results.csv", index=False)

print("\nTop results:")
print(res_df.head(10))
