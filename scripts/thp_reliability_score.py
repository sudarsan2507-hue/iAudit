"""Stage R1 (part 3) - apply the IXI Guys+HH reference (mean/std/covariance,
fit ONLY on IXI Guys+HH, exactly as in reliability_score.py) to every THP scan,
TRANSFORM ONLY - the reference is never refit on THP data.

VERIFY R1: per-subject-per-vendor D table, Friedman/Wilcoxon test (within-
subject, across vendor) on D, mean D per vendor, per-vendor mean tissue
fractions, explicit no-GE-scans caveat, DECISION.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(r"D:\Projects\iAuditAge\results")
FEATURE_COLS = [
    "frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
    "frac_cerebellum", "total_brain_volume_ml",
]

print("=" * 78)
print("STAGE R1 - THP: Mahalanobis D using the UNCHANGED IXI Guys+HH reference")
print("=" * 78)

# --------------------------------------------------------------------------
# 1. Rebuild the IXI Guys+HH reference EXACTLY as reliability_score.py does
#    (fit only on IXI Guys+HH; never refit on THP).
# --------------------------------------------------------------------------
ixi_feat = pd.read_csv(RESULTS / "features.csv")
ref_mask = ixi_feat["site"].isin(["Guys", "HH"])
ref = ixi_feat[ref_mask]
print(f"\nIXI reference set (Guys+HH): n={len(ref)}  (unchanged from reliability_score.py)")

ref_mean = ref[FEATURE_COLS].mean()
ref_std = ref[FEATURE_COLS].std()
X_ref_std = (ref[FEATURE_COLS] - ref_mean) / ref_std
cov = np.cov(X_ref_std.to_numpy(), rowvar=False)
cov_pinv = np.linalg.pinv(cov)
mean_vec = np.zeros(len(FEATURE_COLS))


def mahalanobis_D(df):
    X = (df[FEATURE_COLS] - ref_mean) / ref_std
    return np.array([
        np.sqrt(max((x - mean_vec) @ cov_pinv @ (x - mean_vec), 0))
        for x in X.to_numpy()
    ])


# sanity check: recompute D for IXI itself, should match reliability_score.py
ixi_feat = ixi_feat.copy()
ixi_feat["D"] = mahalanobis_D(ixi_feat)

# --------------------------------------------------------------------------
# 2. THP: transform-only application of the same reference
# --------------------------------------------------------------------------
thp = pd.read_csv(RESULTS / "thp_features.csv")
thp = thp.copy()
thp["D"] = mahalanobis_D(thp)
thp.to_csv(RESULTS / "thp_features_with_D.csv", index=False)
print(f"THP scans: n={len(thp)}  (vendors: {thp['vendor'].value_counts().to_dict()})")
print(f"wrote {RESULTS / 'thp_features_with_D.csv'}")

# --------------------------------------------------------------------------
# VERIFY R1
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERIFY R1")
print("=" * 78)

print("\n*** CAVEAT: ds000206 (THP) contains ONLY Siemens and Philips scans; "
      "ZERO GE scans exist in this dataset (confirmed from BIDS Manufacturer "
      "metadata, not guessed). The original GE-vs-Siemens/Philips IOP-is-GE "
      "sub-hypothesis CANNOT be tested here - it is UNTESTED, not falsified. "
      "Only the general claim 'does scanner vendor alone shift D, anatomy held "
      "constant' can be evaluated with this dataset. ***")

# --- per-subject x vendor mean D table (subjects scanned at multiple sites of
#     each vendor -> average within vendor per subject for the paired test) ---
subj_vendor = thp.groupby(["subject", "vendor"])["D"].mean().unstack("vendor")
print("\n--- per-subject mean D by vendor (averaged across sites/runs of that vendor) ---")
print(subj_vendor.round(4).to_string())

vendors_present = sorted(thp["vendor"].dropna().unique())
print(f"\nvendors present: {vendors_present}")

# --- within-subject paired test across vendor on D ---
# classic Friedman requires >=3 conditions; only 2 vendors exist here, so use
# the natural 2-condition equivalent: Wilcoxon signed-rank on paired
# per-subject vendor means (every one of the 5 THP subjects has both vendors).
paired = subj_vendor.dropna()
print(f"\nsubjects with BOTH vendors present (paired, n={len(paired)}):")
print(paired.round(4).to_string())

if len(vendors_present) >= 3:
    stat, p = stats.friedmanchisquare(*[subj_vendor[v].dropna() for v in vendors_present])
    test_name = "Friedman chi-square"
else:
    stat, p = stats.wilcoxon(paired[vendors_present[0]], paired[vendors_present[1]])
    test_name = (f"Wilcoxon signed-rank ({vendors_present[0]} vs {vendors_present[1]}, "
                  f"paired within-subject) - used INSTEAD of Friedman because only 2 "
                  f"vendor conditions exist in this dataset (Friedman needs >=3)")

print(f"\n{test_name}: statistic={stat:.4f}, p={p:.6f}")

# --- mean D per vendor ---
print("\n--- mean D per vendor (all scans, not just paired subjects) ---")
print(thp.groupby("vendor")["D"].describe()[["count", "mean", "std", "min", "50%", "max"]]
      .round(4).to_string())

# --- per-vendor mean tissue fractions ---
print("\n--- per-vendor mean tissue fractions ---")
print(thp.groupby("vendor")[FEATURE_COLS].mean().round(4).to_string())

# --- explicit GE comparison statement ---
print("\n--- GE-specific IOP hypothesis ---")
print("GE-vs-Siemens/Philips comparison: CANNOT BE TESTED (0 GE scans in THP/ds000206).")

# --------------------------------------------------------------------------
# DECISION
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("DECISION - R1")
print("=" * 78)
alpha = 0.05
if p < alpha:
    print(f"{test_name} is significant (p={p:.6f} < {alpha}): scanner vendor alone "
          f"SHIFTS the OOD score D within-subject, with anatomy (same 5 phantoms) held "
          f"constant. SUPPORTED: 'scanner vendor shifts OOD score' (general claim).")
else:
    print(f"{test_name} is NOT significant (p={p:.6f} >= {alpha}): no reliable "
          f"within-subject vendor effect on D detected in this sample (n={len(paired)} "
          f"paired subjects is small). NOT SUPPORTED here (underpowered, not disproven).")
print("The GE-specific IOP-is-GE sub-hypothesis remains UNTESTED (no GE scans exist "
      "in THP) - it is neither supported nor falsified by this stage.")
print("\nSTOP after VERIFY R1 per staged-gate convention. Awaiting confirmation before Stage R2 (eNKI).")
