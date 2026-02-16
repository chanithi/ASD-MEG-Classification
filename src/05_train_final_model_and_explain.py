import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

RANDOM_STATE = 42
N_SPLITS = 5

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

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

# Feature set: Spectral + PLV (your best model)
SPECTRAL_FEATS = [
    "Delta_RelPower_mean",
    "Theta_RelPower_mean",
    "Alpha_RelPower_mean",
    "Beta_RelPower_mean",
    "Gamma_RelPower_mean",
]
PLV_FEATS = ["PLV_mean_all", "PLV_std_all"]

FEATURES = SPECTRAL_FEATS + PLV_FEATS
X = df[FEATURES].copy()

print("Using dataset shape:", df.shape)
print("Using features:", FEATURES)
print("Group counts:\n", df["Group"].value_counts())

# -----------------------------
# Final model pipeline
# (Keep baseline settings to match earlier results)
# -----------------------------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        C=1.0,            # baseline C
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

# -----------------------------
# A) Cross-validated ROC curve (no leakage)
# -----------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Cross-validated predicted probabilities for ROC
y_prob_cv = cross_val_predict(
    pipe, X, y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

auc_cv = roc_auc_score(y, y_prob_cv)
fpr, tpr, thresholds = roc_curve(y, y_prob_cv)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title(f"Cross-validated ROC Curve (AUC = {auc_cv:.3f})")
roc_path = OUT_DIR / "roc_curve_cv.png"
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved ROC curve:", roc_path)

# -----------------------------
# B) Fit final model on all data and save it
# -----------------------------
pipe.fit(X, y)

model_path = OUT_DIR / "final_model_spectral_plv_logreg.joblib"
joblib.dump(pipe, model_path)
print("✅ Saved final trained model:", model_path)

# -----------------------------
# C) Explainability: coefficients (standardized feature space)
# -----------------------------
# LogisticRegression coef_ corresponds to scaled inputs (after scaler)
clf = pipe.named_steps["clf"]
coefs = clf.coef_.ravel()

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coef": coefs,
    "abs_coef": np.abs(coefs),
    "direction": np.where(coefs >= 0, "ASD↑", "TD↑"),
    "odds_ratio_per_1SD": np.exp(coefs)  # interpret per 1 SD increase
}).sort_values("abs_coef", ascending=False)

coef_csv_path = OUT_DIR / "final_model_coefficients.csv"
coef_df.to_csv(coef_csv_path, index=False)
print("✅ Saved coefficient table:", coef_csv_path)

# Plot top coefficients
top_k = min(10, len(coef_df))
top = coef_df.head(top_k).iloc[::-1]  # reverse for barh

plt.figure(figsize=(8, 5))
plt.barh(top["feature"], top["coef"])
plt.axvline(0)
plt.xlabel("Logistic Regression Coefficient (per 1 SD)")
plt.title(f"Top {top_k} Feature Coefficients (Spectral + PLV)")
coef_plot_path = OUT_DIR / "feature_coefficients.png"
plt.savefig(coef_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved coefficient plot:", coef_plot_path)

# Print quick summary to terminal
print("\nTop features driving prediction (by |coef|):")
print(coef_df.head(10).to_string(index=False))

print(f"\nCross-validated AUC (5-fold): {auc_cv:.3f}")
