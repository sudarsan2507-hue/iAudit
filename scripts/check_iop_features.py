"""Stage V - IOP feature-distribution check.

Tests whether IOP's segmentation-derived tissue-fraction features are
systematically off-distribution vs Guys/HH, independent of age - the
data-level mechanism candidate for why IOP is hard for the tabular models.
"""
import numpy as np
import pandas as pd
from scipy import stats

feat = pd.read_csv(r"D:\Projects\iAuditAge\results\features.csv")

FRACTIONS = ["frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem", "frac_cerebellum"]
SITE_ORDER = ["Guys", "HH", "IOP"]

print("=" * 70)
print("STAGE V - IOP feature-distribution check")
print("=" * 70)

# --------------------------------------------------------------------------
# 1. Per-site summary
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. PER-SITE SUMMARY")
print("=" * 70)
summary = feat.groupby("site").agg(
    n=("subject_id", "count"),
    mean_age=("chronological_age", "mean"),
    median_age=("chronological_age", "median"),
    mean_total_brain_ml=("total_brain_volume_ml", "mean"),
).reindex(SITE_ORDER)
print(summary.round(2).to_string())

# --------------------------------------------------------------------------
# 2. Outlier-segmentation check
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. OUTLIER-SEGMENTATION CHECK (total_brain_volume_ml > 1800 mL)")
print("=" * 70)
outliers = feat[feat["total_brain_volume_ml"] > 1800]
outlier_counts = outliers.groupby("site")["subject_id"].count().reindex(SITE_ORDER, fill_value=0)
print(f"\ntotal subjects with total_brain > 1800 mL: {len(outliers)} / {len(feat)}")
print("\ncount by site:")
print(outlier_counts.to_string())
print("\nIDs by site:")
for site in SITE_ORDER:
    ids = outliers[outliers["site"] == site]["subject_id"].tolist()
    print(f"  {site}: {ids if ids else '(none)'}")

n_iop_total = (feat["site"] == "IOP").sum()
n_iop_outlier = outlier_counts.get("IOP", 0)
frac_iop_outlier = n_iop_outlier / n_iop_total if n_iop_total else 0
n_rest_total = len(feat) - n_iop_total
n_rest_outlier = len(outliers) - n_iop_outlier
frac_rest_outlier = n_rest_outlier / n_rest_total if n_rest_total else 0
print(f"\nIOP outlier rate: {n_iop_outlier}/{n_iop_total} = {frac_iop_outlier:.1%}")
print(f"Guys+HH outlier rate: {n_rest_outlier}/{n_rest_total} = {frac_rest_outlier:.1%}")
if n_iop_outlier == 0:
    print("-> over-segmentation (>1800mL) is NOT observed at IOP in this data.")
elif frac_iop_outlier > frac_rest_outlier:
    print("-> over-segmentation is CONCENTRATED at IOP relative to Guys+HH.")
else:
    print("-> over-segmentation is NOT concentrated at IOP relative to Guys+HH.")

# --------------------------------------------------------------------------
# 3. Raw per-site tissue fractions
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. RAW PER-SITE MEAN TISSUE FRACTIONS")
print("=" * 70)
raw_frac_table = feat.groupby("site")[FRACTIONS].mean().reindex(SITE_ORDER)
print(raw_frac_table.round(4).to_string())

# --------------------------------------------------------------------------
# 4. Age-adjusted comparison
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. AGE-ADJUSTED COMPARISON: IOP vs (Guys+HH) residuals, per fraction")
print("=" * 70)

age = feat["chronological_age"].to_numpy()
is_iop = (feat["site"] == "IOP").to_numpy()
is_rest = feat["site"].isin(["Guys", "HH"]).to_numpy()

results = []
for frac_col in FRACTIONS:
    y = feat[frac_col].to_numpy()
    slope, intercept, r, p_fit, se = stats.linregress(age, y)
    pred = slope * age + intercept
    resid = y - pred

    resid_iop = resid[is_iop]
    resid_rest = resid[is_rest]

    u_stat, p_raw = stats.mannwhitneyu(resid_iop, resid_rest, alternative="two-sided")
    median_diff = np.median(resid_iop) - np.median(resid_rest)
    results.append({
        "fraction": frac_col,
        "age_fit_slope": slope,
        "age_fit_r": r,
        "n_IOP": len(resid_iop),
        "n_rest": len(resid_rest),
        "median_resid_diff_IOP_minus_rest": median_diff,
        "p_raw": p_raw,
    })

n_tests = len(FRACTIONS)
for row in results:
    row["p_bonf"] = min(row["p_raw"] * n_tests, 1.0)

age_adj_table = pd.DataFrame(results)[
    ["fraction", "age_fit_slope", "age_fit_r", "n_IOP", "n_rest",
     "median_resid_diff_IOP_minus_rest", "p_raw", "p_bonf"]
]
print(age_adj_table.round(5).to_string(index=False))

n_significant = (age_adj_table["p_bonf"] < 0.05).sum()
sig_fractions = age_adj_table.loc[age_adj_table["p_bonf"] < 0.05, "fraction"].tolist()
print(f"\nfractions significant after Bonferroni correction (p_bonf < 0.05): "
      f"{n_significant}/{n_tests}")
if sig_fractions:
    print(f"  {sig_fractions}")

# --------------------------------------------------------------------------
# 5. Interpretation
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. INTERPRETATION")
print("=" * 70)

MEANINGFUL_THRESHOLD = 2  # require >=2 of 6 fractions significant to call it "several"
if n_significant >= MEANINGFUL_THRESHOLD:
    print(f"{n_significant}/6 tissue fractions remain significantly shifted at IOP AFTER")
    print("age adjustment (Bonferroni-corrected Mann-Whitney U, IOP vs Guys+HH residuals).")
    print()
    print("-> IOP scans are OFF-DISTRIBUTION at the feature level, independent of age.")
    print("This SUPPORTS the scanner-artifact mechanism for the tabular models: they")
    print("were trained mostly on Guys+HH-distributed volume features, so they")
    print("extrapolate poorly to IOP's shifted feature distribution -> high IOP MAE")
    print("for LinearRegression/RandomForest/XGBoost. STATED AS SUPPORTED.")
else:
    print(f"Only {n_significant}/6 tissue fractions remain significantly shifted at IOP")
    print("after age adjustment (Bonferroni-corrected).")
    print()
    print("-> The tabular models' IOP difficulty is NOT well explained by shifted")
    print("tissue-volume features once age is accounted for. Mechanism SOFTENED to:")
    print("\"IOP is hard for all models, but the specific pathway (feature-distribution")
    print("shift vs some other cause) is UNDETERMINED by this check.\"")

print()
print("CAVEAT (always print, regardless of branch above):")
print("This check speaks directly to the TABULAR pathway (volume/fraction features).")
print("DeepBrainNet uses the raw image, not these volumes, so a shifted-volumes")
print("finding does not by itself explain DeepBrainNet's own IOP error. However,")
print("shifted volumes ARE concrete evidence that the IOP scans differ at the data")
print("level, which is consistent with (not proof of) a common upstream cause - the")
print("GE scanner / acquisition protocol at IOP - degrading both pathways through")
print("different mechanisms. This check does NOT claim to explain DeepBrainNet.")
