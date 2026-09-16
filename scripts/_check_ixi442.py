"""ADDENDUM check - is IXI442's max total_brain (2088 mL) a real large brain
or a corrupted deep_atropos segmentation (e.g. skull/dura included)?

Compares IXI442's 6 tissue volumes against the cohort mean/std per tissue.
"""
import pandas as pd

feat = pd.read_csv(r"D:\Projects\iAuditAge\results\features.csv")

TISSUES = ["CSF", "GM", "WM", "deepGM", "brainstem", "cerebellum"]
vol_cols = [f"volume_{t}_ml" for t in TISSUES]
frac_cols = [f"frac_{t}" for t in TISSUES]

row = feat[feat["subject_id"] == "IXI442"].iloc[0]
cohort = feat[feat["subject_id"] != "IXI442"]

print("=" * 70)
print("IXI442 segmentation sanity check")
print("=" * 70)
print(f"\nIXI442: age={row['chronological_age']:.1f}  sex={row['sex']}  site={row['site']}")
print(f"total_brain_volume_ml = {row['total_brain_volume_ml']:.1f}")
print(f"cohort (n={len(cohort)}) total_brain_volume_ml: mean={cohort['total_brain_volume_ml'].mean():.1f}  "
      f"std={cohort['total_brain_volume_ml'].std():.1f}  max={cohort['total_brain_volume_ml'].max():.1f}  "
      f"min={cohort['total_brain_volume_ml'].min():.1f}")
z = (row["total_brain_volume_ml"] - cohort["total_brain_volume_ml"].mean()) / cohort["total_brain_volume_ml"].std()
print(f"IXI442 total_brain z-score vs cohort: {z:.2f}")

print("\n--- per-tissue volumes: IXI442 vs cohort mean (z-score) ---")
print(f"{'tissue':<12} {'IXI442_ml':>10} {'cohort_mean_ml':>15} {'cohort_std_ml':>14} {'z':>7} {'IXI442_frac':>12} {'cohort_mean_frac':>17}")
for t, vc, fc in zip(TISSUES, vol_cols, frac_cols):
    v_ixi = row[vc]
    v_mean = cohort[vc].mean()
    v_std = cohort[vc].std()
    vz = (v_ixi - v_mean) / v_std
    f_ixi = row[fc]
    f_mean = cohort[fc].mean()
    print(f"{t:<12} {v_ixi:>10.1f} {v_mean:>15.1f} {v_std:>14.1f} {vz:>7.2f} {f_ixi:>12.4f} {f_mean:>17.4f}")

print("\n--- interpretation ---")
print("A REAL large brain: all 6 tissue fractions stay close to cohort-typical")
print("  proportions (uniform scale-up), especially GM/WM fraction not way off.")
print("A CORRUPTED segmentation (skull/dura/non-brain grabbed): usually shows an")
print("  outlier volume concentrated in ONE tissue class (commonly CSF, since")
print("  dura/meninges misclassify as CSF) while other tissue fractions distort.")
