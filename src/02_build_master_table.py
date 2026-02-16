import pandas as pd
from pathlib import Path

OUT_DIR = Path("results")
LAT = OUT_DIR / "latency_subject_level.csv"
SPEC = OUT_DIR / "spectral_subject_level.csv"
PLV = OUT_DIR / "plv_theta_subject_level.csv"

lat = pd.read_csv(LAT)
spec = pd.read_csv(SPEC)
plv = pd.read_csv(PLV)

master = lat.merge(spec, on=["Subject", "Group"], how="inner")
master = master.merge(plv, on=["Subject", "Group"], how="inner")

# Checks
if master.duplicated(subset=["Subject"]).any():
    dups = master[master.duplicated(subset=["Subject"], keep=False)].sort_values("Subject")
    raise ValueError(f"Duplicate subjects after merge:\n{dups[['Subject','Group']]}")

master_path = OUT_DIR / "master_features_subject_level.csv"
master.to_csv(master_path, index=False)

print("✅ Master table created:", master_path)
print("Shape:", master.shape)
print("\nGroup counts:\n", master["Group"].value_counts())
