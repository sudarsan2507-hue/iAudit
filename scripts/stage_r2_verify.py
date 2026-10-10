"""Stage R2 (part 3) - VERIFY R2: eNKI mean D vs IXI-reference D vs IXI-IOP D,
AUC (eNKI-vs-IXI-reference on D), DeepBrainNet MAE on eNKI vs IXI-reference,
Spearman r(D,|error|) with age-controlled partial r. Reuses the same IXI
Guys+HH reference fit as thp_reliability_score.py (transform-only on eNKI).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RESULTS = Path(r"D:\Projects\iAuditAge\results")
FEATURE_COLS = [
    "frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
    "frac_cerebellum", "total_brain_volume_ml",
]

print("=" * 78)
print("STAGE R2 - eNKI: Mahalanobis D + DeepBrainNet MAE using the UNCHANGED IXI reference")
print("=" * 78)

ixi_feat = pd.read_csv(RESULTS / "features.csv")
ref_mask = ixi_feat["site"].isin(["Guys", "HH"])
ref = ixi_feat[ref_mask]
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


ixi_feat = ixi_feat.copy()
ixi_feat["D"] = mahalanobis_D(ixi_feat)

enki = pd.read_csv(RESULTS / "enki_features_predictions.csv")
enki = enki.copy()
enki["D"] = mahalanobis_D(enki)
enki["abs_error"] = (enki["predicted_age"] - enki["chronological_age"]).abs()
enki.to_csv(RESULTS / "enki_features_with_D.csv", index=False)
print(f"\neNKI scans: n={len(enki)}  vendors: {enki['vendor'].value_counts(dropna=False).to_dict()}")
print(f"wrote {RESULTS / 'enki_features_with_D.csv'}")

n_missing_vendor = enki["vendor"].isna().sum()
if n_missing_vendor:
    print(f"NOTE: {n_missing_vendor}/{len(enki)} eNKI scans had no Manufacturer field in "
          f"their JSON sidecar (not guessed, reported as missing).")

print("\n" + "=" * 78)
print("VERIFY R2")
print("=" * 78)

print("\nNOTE: eNKI participants.tsv provides only age/sex/handedness - there is "
      "no explicit clinical-health-screen flag available in the public BIDS "
      "metadata used here, so this is a community-volunteer sample filtered to "
      "age 20-86, not a clinically-verified-healthy sample. Reported honestly, "
      "not fabricated as 'healthy'.")

d_ixi_ref = ixi_feat.loc[ref_mask, "D"]
d_ixi_iop = ixi_feat.loc[ixi_feat["site"] == "IOP", "D"]
d_enki = enki["D"]

print("\n--- mean D: eNKI vs IXI-reference (Guys+HH) vs IXI-IOP ---")
summary = pd.DataFrame({
    "group": ["IXI reference (Guys+HH)", "IXI IOP", "eNKI"],
    "n": [len(d_ixi_ref), len(d_ixi_iop), len(d_enki)],
    "mean_D": [d_ixi_ref.mean(), d_ixi_iop.mean(), d_enki.mean()],
    "std_D": [d_ixi_ref.std(), d_ixi_iop.std(), d_enki.std()],
})
print(summary.round(4).to_string(index=False))

# AUC: eNKI (label=1) vs IXI-reference (label=0) on D
labels = np.concatenate([np.zeros(len(d_ixi_ref)), np.ones(len(d_enki))])
scores = np.concatenate([d_ixi_ref.to_numpy(), d_enki.to_numpy()])
auc = roc_auc_score(labels, scores)
print(f"\nAUC (eNKI vs IXI-reference, D alone): {auc:.4f}")

# DeepBrainNet MAE: eNKI vs IXI-reference
dbn = pd.read_csv(RESULTS / "predictions_deepbrainnet.csv")
dbn_ref = dbn.merge(ixi_feat.loc[ref_mask, ["subject_id"]], on="subject_id", how="inner")
mae_ixi_ref = (dbn_ref["predicted_age"] - dbn_ref["chronological_age"]).abs().mean()
mae_enki = enki["abs_error"].mean()
print(f"\nDeepBrainNet MAE: IXI-reference (Guys+HH) = {mae_ixi_ref:.3f}y (n={len(dbn_ref)})  "
      f"|  eNKI = {mae_enki:.3f}y (n={len(enki)})")

# Spearman r(D, |error|) + age-controlled partial, on eNKI
rho, p = stats.spearmanr(enki["D"], enki["abs_error"])
age = enki["chronological_age"].to_numpy()
D_vals = enki["D"].to_numpy()
err_vals = enki["abs_error"].to_numpy()
slope_d, int_d, *_ = stats.linregress(age, D_vals)
resid_D = D_vals - (slope_d * age + int_d)
slope_e, int_e, *_ = stats.linregress(age, err_vals)
resid_err = err_vals - (slope_e * age + int_e)
rho_partial, p_partial = stats.spearmanr(resid_D, resid_err)

print(f"\nSpearman r(D, |error|) on eNKI: r={rho:.4f}, p={p:.6f}  (n={len(enki)})")
print(f"partial Spearman r (age-controlled): r={rho_partial:.4f}, p={p_partial:.6f}")

print("\n" + "=" * 78)
print("DECISION - R2")
print("=" * 78)
fires = auc >= 0.6 or d_enki.mean() > d_ixi_ref.mean() * 1.2
predicts = p_partial < 0.05
print(f"Does the OOD detector fire on eNKI (AUC>=0.6 or mean D notably elevated vs "
      f"IXI-reference)? {'YES' if fires else 'NO'} (AUC={auc:.3f}, "
      f"mean_D_enki={d_enki.mean():.3f} vs mean_D_ixi_ref={d_ixi_ref.mean():.3f})")
print(f"Does D predict |error| on eNKI after age control? "
      f"{'YES' if predicts else 'NO'} (partial r={rho_partial:.4f}, p={p_partial:.6f})")
print("\nSTOP after VERIFY R2 per staged-gate convention. Awaiting confirmation before Stage R3.")
