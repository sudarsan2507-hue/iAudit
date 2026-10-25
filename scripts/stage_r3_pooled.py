"""Stage R3 - pooled synthesis across IXI (reference + IOP), THP (Siemens/
Philips), and eNKI (community sample), all sharing the SAME unchanged IXI
Guys+HH Mahalanobis-D reference. One |error| vs D figure, pooled Spearman +
partial r, one evidence-backed universal statement, honest limitations.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(r"D:\Projects\iAuditAge\results")

print("=" * 78)
print("STAGE R3 - POOLED SYNTHESIS")
print("=" * 78)

# --------------------------------------------------------------------------
# Assemble pooled |error| vs D across every dataset where BOTH exist.
# THP has no DeepBrainNet predictions (Stage R1 only computed D, not error -
# THP subjects are physical phantoms with no chronological age to predict
# against, so |error| is undefined for THP and it is EXCLUDED from the
# error-vs-D pooling, included only in the D-by-vendor context below).
# --------------------------------------------------------------------------
ixi_feat = pd.read_csv(RESULTS / "features.csv")
FEATURE_COLS = [
    "frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
    "frac_cerebellum", "total_brain_volume_ml",
]
ref_mask = ixi_feat["site"].isin(["Guys", "HH"])
ref = ixi_feat[ref_mask]
ref_mean = ref[FEATURE_COLS].mean()
ref_std = ref[FEATURE_COLS].std()
cov = np.cov(((ref[FEATURE_COLS] - ref_mean) / ref_std).to_numpy(), rowvar=False)
cov_pinv = np.linalg.pinv(cov)
mean_vec = np.zeros(len(FEATURE_COLS))


def mahalanobis_D(df):
    X = (df[FEATURE_COLS] - ref_mean) / ref_std
    return np.array([np.sqrt(max((x - mean_vec) @ cov_pinv @ (x - mean_vec), 0))
                      for x in X.to_numpy()])


ixi_feat = ixi_feat.copy()
ixi_feat["D"] = mahalanobis_D(ixi_feat)
dbn = pd.read_csv(RESULTS / "predictions_deepbrainnet.csv").drop(columns=["site"])
ixi = dbn.merge(ixi_feat[["subject_id", "site", "D"]], on="subject_id", how="inner")
ixi["abs_error"] = (ixi["predicted_age"] - ixi["chronological_age"]).abs()
ixi["dataset"] = ixi["site"].map(lambda s: "IXI-reference (Guys+HH)" if s in ("Guys", "HH") else "IXI-IOP")

enki = pd.read_csv(RESULTS / "enki_features_with_D.csv")
enki["dataset"] = "eNKI"
enki_pooled = enki[["chronological_age", "D", "abs_error", "dataset"]]

pooled = pd.concat([
    ixi[["chronological_age", "D", "abs_error", "dataset"]],
    enki_pooled,
], ignore_index=True)
pooled.to_csv(RESULTS / "pooled_error_vs_D.csv", index=False)
print(f"\npooled n={len(pooled)}  by dataset:\n{pooled['dataset'].value_counts().to_string()}")

# --------------------------------------------------------------------------
# Pooled figure
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"IXI-reference (Guys+HH)": "tab:blue", "IXI-IOP": "tab:red", "eNKI": "tab:purple"}
for ds, color in colors.items():
    sub = pooled[pooled["dataset"] == ds]
    ax.scatter(sub["D"], sub["abs_error"], s=12, alpha=0.5, color=color, label=f"{ds} (n={len(sub)})")
slope, intercept, *_ = stats.linregress(pooled["D"], pooled["abs_error"])
xs = np.linspace(pooled["D"].min(), pooled["D"].max(), 50)
ax.plot(xs, slope * xs + intercept, color="black", linewidth=2, label="pooled fit")
ax.set_xlabel("Mahalanobis distance D (IXI Guys+HH reference)")
ax.set_ylabel("|DeepBrainNet error| (years)")
ax.set_title("Pooled |error| vs OOD distance D, across datasets/vendors")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "fig_pooled_error_vs_D.png", dpi=120)
plt.close(fig)
print(f"\nwrote {RESULTS / 'fig_pooled_error_vs_D.png'}")

# --------------------------------------------------------------------------
# Pooled Spearman + partial r (age-controlled)
# --------------------------------------------------------------------------
rho, p = stats.spearmanr(pooled["D"], pooled["abs_error"])
age = pooled["chronological_age"].to_numpy()
D_vals = pooled["D"].to_numpy()
err_vals = pooled["abs_error"].to_numpy()
slope_d, int_d, *_ = stats.linregress(age, D_vals)
resid_D = D_vals - (slope_d * age + int_d)
slope_e, int_e, *_ = stats.linregress(age, err_vals)
resid_err = err_vals - (slope_e * age + int_e)
rho_partial, p_partial = stats.spearmanr(resid_D, resid_err)

print(f"\npooled Spearman r(D, |error|): r={rho:.4f}, p={p:.2e}  (n={len(pooled)})")
print(f"pooled partial Spearman r (age-controlled): r={rho_partial:.4f}, p={p_partial:.2e}")

print("\n" + "=" * 78)
print("VERIFY R3")
print("=" * 78)
verdict = ("SUPPORTED" if (p_partial < 0.05 and rho_partial > 0) else
           "NOT SUPPORTED at this pooled sample")
print(f"\nOne evidence-backed universal statement:")
print(f"  Across {pooled['dataset'].nunique()} independently-collected datasets "
      f"({', '.join(sorted(pooled['dataset'].unique()))}, pooled n={len(pooled)}), "
      f"distance from the IXI Guys+HH reference distribution (D) is "
      f"{'positively' if rho_partial > 0 else 'not reliably'} associated with "
      f"DeepBrainNet's absolute prediction error, after controlling for chronological "
      f"age (partial Spearman r={rho_partial:.3f}, p={p_partial:.2e}). "
      f"CLAIM STATUS: {verdict}.")

print("\n--- LIMITATIONS (honest) ---")
print("""\
  1. Single feature-extraction pipeline (deep_atropos 6-tissue-fraction + total
     brain volume) and a single reference (IXI Guys+HH) - not validated against
     any alternative OOD-detection feature space.
  2. THP n=5 unique subjects (65 scans, heavily repeated-measures) - D-by-vendor
     comparisons in Stage R1 are underpowered (max attainable p=0.0625 with a
     5-pair Wilcoxon test); THP has no chronological-age ground truth (physical
     phantoms) so it contributes to the vendor/D story only, not to the pooled
     error-vs-D analysis here.
  3. eNKI capped at n=40 (not the originally-planned ~150) because anonymous
     downloads from the public FCP-INDI S3 bucket ran at ~30-40 KB/s in this
     environment, making 150 full T1w downloads impractical within this session;
     this is a session-time constraint, not a data-access restriction (eNKI
     itself is genuinely public/no-DUA).
  4. eNKI is a community-volunteer sample (age/sex/handedness only, no clinical
     health-screen flag in the public participants.tsv) - NOT a curated
     clinically-healthy cohort, unlike IXI. This is disclosed, not fabricated
     as "healthy".
  5. Population/scanner confounds are NOT disentangled here: eNKI differs from
     IXI in country, scanner site(s), acquisition protocol, and demographics
     simultaneously - the pooled analysis shows association with D, not a
     controlled causal test of any single confound.
  6. OASIS-3 transfer test (Task 1) and CamCAN are still pending / not
     attempted in this pipeline run - the pooled statement above covers only
     IXI + THP(D-only) + eNKI, not the full originally-scoped dataset list.
""")
