"""One-off PRE-CHECK before Stage T2 - not part of the pipeline proper.

Verifies the 16 deep_atropos MemoryError drops didn't gut any site, and flags
the max-total_brain subject as a watch-item.
"""
import pandas as pd

feat = pd.read_csv(r"D:\Projects\iAuditAge\results\features.csv")
dbn = pd.read_csv(r"D:\Projects\iAuditAge\results\predictions_deepbrainnet.csv")

print("=" * 60)
print("PRE-CHECK (before Stage T2)")
print("=" * 60)

print(f"\nfeatures.csv subjects: {len(feat)}")
print(f"predictions_deepbrainnet.csv subjects: {len(dbn)}")

print("\n--- site breakdown: 544 feature subjects vs 560 DeepBrainNet subjects ---")
feat_by_site = feat["site"].value_counts().sort_index()
dbn_by_site = dbn["site"].value_counts().sort_index()
cmp = pd.DataFrame({"features_544": feat_by_site, "deepbrainnet_560": dbn_by_site})
cmp["dropped"] = cmp["deepbrainnet_560"] - cmp["features_544"]
print(cmp.to_string())

iop_n = feat_by_site.get("IOP", 0)
print(f"\nIOP subjects remaining in features.csv: {iop_n}")
if iop_n >= 50:
    print(f"IOP check: PASS ({iop_n} >= 50)")
else:
    print(f"IOP check: *** FLAG *** ({iop_n} < 50) - IOP comparison is weakened")

print("\n--- max total_brain_volume_ml watch-item ---")
row = feat.loc[feat["total_brain_volume_ml"].idxmax()]
print(f"subject_id={row['subject_id']}  total_brain_volume_ml={row['total_brain_volume_ml']:.1f}  "
      f"age={row['chronological_age']:.1f}  sex={row['sex']}  site={row['site']}")
print("(noted as a watch-item - check if this subject gets a wild predicted_age in T2)")
