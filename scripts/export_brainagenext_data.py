"""Package BrainAgeNeXt's per-scan outputs from the Colab run into the repo.

Inputs:
  results/transfer_head_predictions.csv     (Colab export: ages, raw and head predictions)
  <bang_embed dir>/*.npz                    (Colab export: 5 model predictions + 5x512 features per scan)
  results/predictions_deepbrainnet.csv      (IXI sex / ethnicity)
  results/enki_features_predictions.csv     (eNKI sex)

Writes:
  results/brainagenext_predictions.csv  one row per scan: demographics, ensemble median,
                                         all 5 model predictions, ensemble SD, ridge-head prediction
  results/brainagenext_features.npz     scan_id + (n, 5, 512) float16 encoder features
  results/brainagenext_uncertainty.csv  does 5-model disagreement predict error? (age-controlled)

Named brainagenext_* (not predictions_*) so fairness_audit.py's predictions_*.csv glob,
which covers the full IXI cohort incl. IOP, is not changed by this subset.

Usage: python scripts/export_brainagenext_data.py [bang_embed_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_EMBED = Path(r"E:\iAuditData\colab_outputs\bang_embed")


def demographics():
    ixi = pd.read_csv(RESULTS / "predictions_deepbrainnet.csv")[["subject_id", "sex", "ethnicity"]]
    ixi = ixi.rename(columns={"subject_id": "scan_id"})
    enki = pd.read_csv(RESULTS / "enki_features_predictions.csv")[["participant_id", "sex"]]
    enki = pd.DataFrame({"scan_id": "eNKI_" + enki["participant_id"],
                         "sex": enki["sex"].str[0].str.upper(), "ethnicity": "not recorded"})
    return pd.concat([ixi, enki], ignore_index=True)


def partial_spearman(age, x, err):
    rx = x - np.polyval(np.polyfit(age, x, 1), age)
    re = err - np.polyval(np.polyfit(age, err, 1), age)
    return stats.spearmanr(rx, re)


def main():
    embed = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EMBED
    pred = pd.read_csv(RESULTS / "transfer_head_predictions.csv").rename(
        columns={"pred BANG 512": "pred_bang_ridge"})
    rows, feats = [], []
    for r in pred.itertuples():
        z = np.load(embed / f"{r.scan_id}.npz", allow_pickle=False)
        if abs(float(z["predicted_age"]) - r.bang_raw) > 1e-3:
            raise SystemExit(f"{r.scan_id}: feature file does not match transfer_head_predictions.csv")
        p = z["preds"].astype(float)
        rows.append({"scan_id": r.scan_id, "dataset": "eNKI" if r.site == "eNKI" else "IXI", "site": r.site,
                     "chronological_age": r.age, "predicted_age": float(np.median(p)),
                     **{f"pred_model{i + 1}": v for i, v in enumerate(p)},
                     "ensemble_sd": float(p.std()), "pred_ridge_head": r.pred_bang_ridge,
                     "orientation": str(z["orientation"])})
        feats.append(z["feat"])
    out = pd.DataFrame(rows)
    assert np.allclose(out["predicted_age"], pred["bang_raw"], atol=1e-3)
    out = out.merge(demographics(), on="scan_id", how="left")
    if out["sex"].isna().any():
        raise SystemExit(f"missing sex for {out.loc[out['sex'].isna(), 'scan_id'].tolist()[:5]}")
    cols = ["scan_id", "dataset", "site", "chronological_age", "sex", "ethnicity", "predicted_age",
            *[f"pred_model{i}" for i in range(1, 6)], "ensemble_sd", "pred_ridge_head", "orientation"]
    out[cols].to_csv(RESULTS / "brainagenext_predictions.csv", index=False, float_format="%.4f")
    np.savez_compressed(RESULTS / "brainagenext_features.npz", scan_id=out["scan_id"].to_numpy(dtype="U20"),
                        features=np.stack(feats).astype(np.float16))
    print(f"wrote brainagenext_predictions.csv ({len(out)} rows) and brainagenext_features.npz "
          f"{np.stack(feats).shape}")

    unc = []
    for scope, m in [("all", np.ones(len(out), bool)), ("IXI Guys+HH", out["dataset"].eq("IXI").to_numpy()),
                     ("eNKI", out["dataset"].eq("eNKI").to_numpy())]:
        s = out[m]
        err = (s["predicted_age"] - s["chronological_age"]).abs().to_numpy()
        r, p = partial_spearman(s["chronological_age"].to_numpy(), s["ensemble_sd"].to_numpy(), err)
        unc.append({"scope": scope, "n": len(s), "median_ensemble_sd": s["ensemble_sd"].median(),
                    "partial_spearman_r_sd_vs_abs_error": r, "p": p})
        print(f"[{scope:11s}] n={len(s):3d}  median SD {s['ensemble_sd'].median():.2f}y  "
              f"partial r(SD, |error|)={r:+.3f}  p={p:.3g}")
    pd.DataFrame(unc).to_csv(RESULTS / "brainagenext_uncertainty.csv", index=False)


if __name__ == "__main__":
    main()
