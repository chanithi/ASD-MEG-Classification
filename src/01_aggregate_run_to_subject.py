import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

LAT_FILE  = DATA_DIR / "M50_M100_per_run.csv"
SPEC_FILE = DATA_DIR / "RelativeBandPower_per_run.csv"
PLV_FILE  = DATA_DIR / "Cleaned_Run_PLV_theta_ALL_SUBJECTS.csv"

# -----------------------------
# Latency -> subject level
# -----------------------------
lat = pd.read_csv(LAT_FILE)
lat_subj = (
    lat.groupby(["Subject", "Group"], as_index=False)
       .agg(
            M50_latency_ms_mean=("M50_latency_ms", "mean"),
            M100_latency_ms_mean=("M100_latency_ms", "mean"),
            n_epochs_mean=("n_epochs", "mean"),
            n_runs_latency=("Run", "nunique"),
       )
)
lat_out = OUT_DIR / "latency_subject_level.csv"
lat_subj.to_csv(lat_out, index=False)
print("✅ Wrote:", lat_out, "| shape:", lat_subj.shape)

# -----------------------------
# Spectral -> subject level
# -----------------------------
spec = pd.read_csv(SPEC_FILE)
spec_subj = (
    spec.groupby(["Subject", "Group"], as_index=False)
        .agg(
            Delta_RelPower_mean=("Delta_RelPower", "mean"),
            Theta_RelPower_mean=("Theta_RelPower", "mean"),
            Alpha_RelPower_mean=("Alpha_RelPower", "mean"),
            Beta_RelPower_mean=("Beta_RelPower", "mean"),
            Gamma_RelPower_mean=("Gamma_RelPower", "mean"),
            n_runs_spectral=("Run", "nunique"),
        )
)
spec_out = OUT_DIR / "spectral_subject_level.csv"
spec_subj.to_csv(spec_out, index=False)
print("✅ Wrote:", spec_out, "| shape:", spec_subj.shape)

# -----------------------------
# PLV theta -> subject level (summary features)
# -----------------------------
plv = pd.read_csv(PLV_FILE)

# Ensure consistent naming order
# (your PLV file has Group, Subject, Run,...)
plv_subj = (
    plv.groupby(["Subject", "Group"], as_index=False)
       .agg(
            PLV_mean_all=("PLV", "mean"),
            PLV_std_all=("PLV", "std"),
            n_runs_plv=("Run", "nunique"),
            n_measures=("Measure", "nunique"),
       )
)

plv_subj["PLV_std_all"] = plv_subj["PLV_std_all"].fillna(0.0)

plv_out = OUT_DIR / "plv_theta_subject_level.csv"
plv_subj.to_csv(plv_out, index=False)
print("✅ Wrote:", plv_out, "| shape:", plv_subj.shape)
