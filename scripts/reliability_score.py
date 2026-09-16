"""Reliability score - model-agnostic, unsupervised feature-space distance
from the "seen" (Guys+HH) distribution, tested for whether it predicts
per-subject error across all 4 models, recovers IOP without site labels, and
enables useful selective prediction (abstain on high-distance scans).

IOP is NEVER used to build the reference distribution - only Guys+HH define
"seen", so IOP's distances are an honest out-of-distribution (OOD) test.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS = Path(r"D:\Projects\iAuditAge\results")
FEATURE_COLS = [
    "frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
    "frac_cerebellum", "total_brain_volume_ml",
]
MODELS = ["DeepBrainNet", "LinearRegression", "RandomForest", "XGBoost"]
np.random.seed(42)

print("=" * 70)
print("RELIABILITY SCORE - feature-space distance from Guys+HH distribution")
print("=" * 70)

# --------------------------------------------------------------------------
# 1-2. Reference distribution (Guys+HH ONLY) + Mahalanobis distance for all
# --------------------------------------------------------------------------
feat = pd.read_csv(RESULTS / "features.csv")
ref_mask = feat["site"].isin(["Guys", "HH"])
ref = feat[ref_mask]
print(f"\nreference set (Guys+HH): n={len(ref)}  |  full cohort: n={len(feat)}  "
      f"|  IOP (never in reference): n={(feat['site'] == 'IOP').sum()}")

ref_mean = ref[FEATURE_COLS].mean()
ref_std = ref[FEATURE_COLS].std()

X_all = (feat[FEATURE_COLS] - ref_mean) / ref_std
X_ref_std = (ref[FEATURE_COLS] - ref_mean) / ref_std  # for covariance fit, mean ~0

cov = np.cov(X_ref_std.to_numpy(), rowvar=False)
# the 6 fractions sum to 1 -> perfectly collinear -> covariance is rank-deficient.
# Use the pseudo-inverse (robust to singular covariance) rather than np.linalg.inv.
cov_pinv = np.linalg.pinv(cov)

mean_vec = np.zeros(len(FEATURE_COLS))  # reference mean in standardized space is 0 by construction
D = np.array([
    np.sqrt(max((x - mean_vec) @ cov_pinv @ (x - mean_vec), 0))
    for x in X_all.to_numpy()
])
feat = feat.copy()
feat["D"] = D

print("\n--- Mahalanobis distance D summary (all sites) ---")
print(feat["D"].describe().round(3).to_string())
print("\n--- D summary by site ---")
print(feat.groupby("site")["D"].describe()[["count", "mean", "std", "min", "50%", "max"]].round(3).to_string())

# --------------------------------------------------------------------------
# 3. Join D to per-subject error, per model
# --------------------------------------------------------------------------
d_table = feat[["subject_id", "site", "chronological_age", "D",
                "total_brain_volume_ml"]].copy()

model_frames = {}
for model in MODELS:
    fname = f"predictions_{model.lower()}.csv"
    preds = pd.read_csv(RESULTS / fname)
    preds["abs_error"] = (preds["predicted_age"] - preds["chronological_age"]).abs()
    merged = preds.merge(d_table[["subject_id", "D"]], on="subject_id", how="inner")
    model_frames[model] = merged
    print(f"\n{model}: merged n={len(merged)} (predictions n={len(preds)}, "
          f"with D available n={len(merged)})")

# --------------------------------------------------------------------------
# 4. CORE RESULT - does D predict error, before/after age control?
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. CORE RESULT - Spearman r(D, |error|) and partial correlation (age-controlled)")
print("=" * 70)

core_rows = []
for model in MODELS:
    df = model_frames[model]
    D_vals = df["D"].to_numpy()
    err_vals = df["abs_error"].to_numpy()
    age_vals = df["chronological_age"].to_numpy()

    rho, p = stats.spearmanr(D_vals, err_vals)

    # partial correlation controlling for age: residualize D and |error| on
    # age via linear regression, then Spearman-correlate the residuals.
    slope_d, int_d, *_ = stats.linregress(age_vals, D_vals)
    resid_D = D_vals - (slope_d * age_vals + int_d)
    slope_e, int_e, *_ = stats.linregress(age_vals, err_vals)
    resid_err = err_vals - (slope_e * age_vals + int_e)
    rho_partial, p_partial = stats.spearmanr(resid_D, resid_err)

    core_rows.append({
        "model": model, "n": len(df),
        "spearman_r": rho, "p": p,
        "partial_spearman_r_age_ctrl": rho_partial, "p_partial": p_partial,
    })

core_table = pd.DataFrame(core_rows)
print(core_table.round(5).to_string(index=False))

# fig_error_vs_distance.png: 4 panels
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
site_colors = {"Guys": "tab:blue", "HH": "tab:green", "IOP": "tab:red"}
for ax, model in zip(axes.flat, MODELS):
    df = model_frames[model]
    for site, color in site_colors.items():
        sub = df[df["site"] == site]
        ax.scatter(sub["D"], sub["abs_error"], s=10, alpha=0.5, color=color, label=site)
    slope, intercept, *_ = stats.linregress(df["D"], df["abs_error"])
    xs = np.linspace(df["D"].min(), df["D"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.5)
    ax.set_title(model)
    ax.set_xlabel("Mahalanobis distance D")
    ax.set_ylabel("|error| (years)")
axes.flat[0].legend()
fig.tight_layout()
fig.savefig(RESULTS / "fig_error_vs_distance.png", dpi=120)
plt.close(fig)
print("\nwrote fig_error_vs_distance.png")

# --------------------------------------------------------------------------
# 5. Unsupervised site recovery
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. UNSUPERVISED SITE RECOVERY")
print("=" * 70)

fig, ax = plt.subplots(figsize=(6, 5))
SITE_ORDER = ["Guys", "HH", "IOP"]
data = [feat.loc[feat["site"] == s, "D"] for s in SITE_ORDER]
ax.boxplot(data, tick_labels=SITE_ORDER)
ax.set_ylabel("Mahalanobis distance D")
ax.set_title("Distance-from-reference by site")
fig.tight_layout()
fig.savefig(RESULTS / "fig_distance_by_site.png", dpi=120)
plt.close(fig)
print("wrote fig_distance_by_site.png")

is_iop_label = (feat["site"] == "IOP").astype(int).to_numpy()
auc = roc_auc_score(is_iop_label, feat["D"].to_numpy())
print(f"\nROC AUC (D alone, classifying IOP vs not-IOP, no site labels used to fit D): {auc:.4f}")

# --------------------------------------------------------------------------
# 6. Selective prediction
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. SELECTIVE PREDICTION (abstain on high-D scans)")
print("=" * 70)

PRINT_COVERAGES = [1.00, 0.90, 0.85, 0.80]
COVERAGE_GRID = np.round(np.arange(1.00, 0.49, -0.02), 2)
COVERAGE_GRID = np.round(np.unique(np.concatenate([COVERAGE_GRID, PRINT_COVERAGES])), 2)[::-1]
N_SHUFFLES = 20

selective_rows = []
curves = {}  # model -> {"D": [(cov, mae)], "random": [(cov, mae_mean)]}

for model in MODELS:
    df = model_frames[model].sort_values("D", ascending=True).reset_index(drop=True)
    n = len(df)
    errs = df["abs_error"].to_numpy()

    d_curve = []
    rand_curve = []
    for cov in COVERAGE_GRID:
        k = int(round(cov * n))
        k = max(k, 1)
        d_mae = errs[:k].mean()  # lowest-D k subjects (D-sorted ascending)
        d_curve.append((cov, d_mae))

        rand_maes = []
        for _ in range(N_SHUFFLES):
            idx = np.random.choice(n, size=k, replace=False)
            rand_maes.append(errs[idx].mean())
        rand_curve.append((cov, np.mean(rand_maes)))

    curves[model] = {"D": d_curve, "random": rand_curve}

    d_dict = dict(d_curve)
    rand_dict = dict(rand_curve)
    mae_100 = d_dict[1.00]
    for cov in PRINT_COVERAGES:
        drop = mae_100 - d_dict[cov]
        selective_rows.append({
            "model": model, "coverage": cov,
            "MAE_D_based": d_dict[cov], "MAE_random_baseline": rand_dict[cov],
            "abs_drop_from_100pct_D_based": drop,
            "D_beats_random": d_dict[cov] < rand_dict[cov],
        })

selective_table = pd.DataFrame(selective_rows)
print(selective_table.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
colors = {"DeepBrainNet": "tab:blue", "LinearRegression": "tab:orange",
          "RandomForest": "tab:green", "XGBoost": "tab:purple"}
for model in MODELS:
    d_cov, d_mae = zip(*curves[model]["D"])
    r_cov, r_mae = zip(*curves[model]["random"])
    ax.plot(d_cov, d_mae, color=colors[model], linewidth=2, label=f"{model} (D-based)")
    ax.plot(r_cov, r_mae, color=colors[model], linewidth=1, linestyle="--",
             label=f"{model} (random)")
ax.set_xlabel("coverage (fraction retained)")
ax.set_ylabel("MAE (years)")
ax.set_title("Selective prediction: MAE vs coverage")
ax.invert_xaxis()
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(RESULTS / "fig_selective_prediction.png", dpi=120)
plt.close(fig)
print("\nwrote fig_selective_prediction.png")

# --------------------------------------------------------------------------
# 7. Confound checks
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. CONFOUND CHECKS")
print("=" * 70)

tb_dev = (feat["total_brain_volume_ml"] - ref_mean["total_brain_volume_ml"]).abs()
r_tb, p_tb = stats.pearsonr(feat["D"], tb_dev)
rho_tb, p_tb_s = stats.spearmanr(feat["D"], tb_dev)
print(f"\n(a) is D just total_brain outlierness?")
print(f"    Pearson r(D, |total_brain - ref_mean|) = {r_tb:.4f}  (p={p_tb:.2e})")
print(f"    Spearman r(D, |total_brain - ref_mean|) = {rho_tb:.4f}  (p={p_tb_s:.2e})")
if abs(r_tb) > 0.8:
    print("    -> D is STRONGLY driven by total_brain outlierness alone (not adding much")
    print("       beyond that single feature).")
elif abs(r_tb) > 0.5:
    print("    -> D is MODERATELY related to total_brain outlierness but clearly reflects")
    print("       other feature dimensions too.")
else:
    print("    -> D is NOT simply total_brain outlierness; it reflects the joint")
    print("       tissue-fraction distribution, not just head size.")

print(f"\n(b) does the D-vs-error link survive within Guys+HH ALONE (excluding IOP)?")
within_rows = []
for model in MODELS:
    df = model_frames[model]
    df_gh = df[df["site"].isin(["Guys", "HH"])]
    rho_gh, p_gh = stats.spearmanr(df_gh["D"], df_gh["abs_error"])
    within_rows.append({"model": model, "n_Guys_HH": len(df_gh),
                         "spearman_r_within_GuysHH": rho_gh, "p": p_gh})
within_table = pd.DataFrame(within_rows)
print(within_table.round(5).to_string(index=False))

# --------------------------------------------------------------------------
# Plain-language summary
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY - 3-line verdict")
print("=" * 70)

sig_after_age_ctrl = (core_table["p_partial"] < 0.05).sum()
a_answer = "YES" if sig_after_age_ctrl == len(MODELS) else (
    "PARTIALLY" if sig_after_age_ctrl > 0 else "NO")
print(f"(a) Does D predict per-subject error across ALL 4 models, after age control? "
      f"{a_answer} ({sig_after_age_ctrl}/{len(MODELS)} models significant, p_partial<0.05)")

b_answer = "YES" if auc >= 0.7 else ("WEAKLY" if auc >= 0.6 else "NO")
print(f"(b) Does D recover IOP unsupervised (no site labels used to fit it)? "
      f"{b_answer} (AUC={auc:.3f})")

beats_random_count = selective_table.groupby("model")["D_beats_random"].all().sum()
c_answer = "YES" if beats_random_count == len(MODELS) else (
    "PARTIALLY" if beats_random_count > 0 else "NO")
print(f"(c) Does D-based selective prediction beat random abstention at every printed "
      f"coverage, for all 4 models? {c_answer} ({beats_random_count}/{len(MODELS)} models)")
