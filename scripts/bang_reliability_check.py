"""Does the reliability score D predict BrainAgeNeXt's errors, as it does DeepBrainNet's?

Same distance and statistics as stage_r3_pooled.py: Mahalanobis D on tissue-fraction
features, fit on IXI Guys+HH only; partial Spearman r between D and |error| after
regressing chronological age out of both. Both models are tested on the identical
subjects (IXI Guys+HH with features + eNKI; IOP was not in the Colab run).

Input: transfer_head_predictions.csv from the Colab export (default path below).
Writes results/bang_reliability_check.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_IN = RESULTS / "transfer_head_predictions.csv"   # copied from the Colab export
FEATURE_COLS = ["frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
                "frac_cerebellum", "total_brain_volume_ml"]
MODELS = {"dbn_raw": "DeepBrainNet", "bang_raw": "BrainAgeNeXt", "pred BANG 512": "BrainAgeNeXt+ridge"}


def reference_D():
    """Mahalanobis D exactly as in stage_r3_pooled.py (reference = IXI Guys+HH)."""
    feat = pd.read_csv(RESULTS / "features.csv")
    ref = feat[feat["site"].isin(["Guys", "HH"])]
    mean, std = ref[FEATURE_COLS].mean(), ref[FEATURE_COLS].std()
    cov_pinv = np.linalg.pinv(np.cov(((ref[FEATURE_COLS] - mean) / std).to_numpy(), rowvar=False))
    X = ((feat[FEATURE_COLS] - mean) / std).to_numpy()
    feat["D"] = [np.sqrt(max(x @ cov_pinv @ x, 0)) for x in X]
    ixi = feat[["subject_id", "D"]].rename(columns={"subject_id": "scan_id"})
    enki = pd.read_csv(RESULTS / "enki_features_with_D.csv")
    enki = pd.DataFrame({"scan_id": "eNKI_" + enki["participant_id"], "D": enki["D"]})
    return pd.concat([ixi, enki], ignore_index=True)


def partial_spearman(age, d, err):
    rd = d - np.polyval(np.polyfit(age, d, 1), age)
    re = err - np.polyval(np.polyfit(age, err, 1), age)
    return stats.spearmanr(rd, re)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    pred = pd.read_csv(path).merge(reference_D(), on="scan_id", how="inner")
    print(f"subjects with D and predictions: {len(pred)}  {pred['site'].value_counts().to_dict()}")
    rows = []
    for scope, mask in [("IXI Guys+HH + eNKI", np.ones(len(pred), bool)),
                        ("IXI Guys+HH only", pred["site"].isin(["Guys", "HH"]).to_numpy())]:
        sub = pred[mask]
        for col, name in MODELS.items():
            err = (sub[col] - sub["age"]).abs().to_numpy()
            r, p = partial_spearman(sub["age"].to_numpy(), sub["D"].to_numpy(), err)
            rows.append({"scope": scope, "model": name, "n": len(sub), "partial_spearman_r": r, "p": p})
            print(f"[{scope}] {name:20s} n={len(sub)}  partial r={r:+.3f}  p={p:.3g}")
    out = RESULTS / "bang_reliability_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
