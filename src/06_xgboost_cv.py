import pandas as pd
import numpy as np

# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("results/model_master_features_subject_level.csv")

print("Dataset shape:", df.shape)

# -----------------------------
# Encode labels
# -----------------------------

y = df["Group"].map({"TD": 0, "ASD": 1})

# -----------------------------
# Select features
# -----------------------------

features = [
    "M50_latency_ms_mean",
    "M100_latency_ms_mean",
    "Delta_RelPower_mean",
    "Theta_RelPower_mean",
    "Alpha_RelPower_mean",
    "Beta_RelPower_mean",
    "Gamma_RelPower_mean",
    "PLV_mean_all",
    "PLV_std_all"
]

X = df[features]

print("Number of subjects:", len(X))
print("Number of features:", X.shape[1])

# -----------------------------
# Cross validation
# -----------------------------

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

results = []

# for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     model = XGBClassifier(
#         n_estimators=200,
#         max_depth=3,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         eval_metric="logloss",
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:,1]

#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_prob)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

#     sensitivity = tp/(tp+fn)
#     specificity = tn/(tn+fp)

#     results.append([
#         fold,
#         acc,
#         auc,
#         precision,
#         recall,
#         f1,
#         sensitivity,
#         specificity
#     ])
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # ---- metrics ----
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # ---- store results ----
    results.append([
        fold,
        acc,
        auc,
        precision,
        recall,
        f1,
        sensitivity,
        specificity
    ])
# -----------------------------
# Convert results
# -----------------------------

results = pd.DataFrame(
    results,
    columns=[
        "fold",
        "accuracy",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity"
    ]
)

print("\nCross-validation results:\n")
print(results)

print("\nMean performance:\n")
print(results.mean())

# -----------------------------
# Save results
# -----------------------------

results.to_csv("results/xgboost_cv_results_updated.csv", index=False)

print("\nSaved: results/xgboost_cv_results_updated.csv")