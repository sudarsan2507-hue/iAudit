"""Export every scan and every model's prediction into one table, from the repo's results/ files only.

Run from anywhere:  python website/source/export_all_values.py   (needs pandas, numpy)
Writes website/data/all_predictions.csv  - one row per scan (IXI incl. IOP, eNKI, SALD)
       website/data/metrics.json         - cohort-level results the site and README quote
Every value is copied or recomputed from results/; nothing is entered by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = REPO / "website" / "data"
FEATURES = ["frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem", "frac_cerebellum",
            "total_brain_volume_ml"]


def distance_D():
    """Mahalanobis distance from the IXI Guys+HH tissue-volume distribution (same as scripts/stage_r3_pooled.py)."""
    f = pd.read_csv(RES / "features.csv")
    ref = f[f["site"].isin(["Guys", "HH"])]
    mean, std = ref[FEATURES].mean(), ref[FEATURES].std()
    pinv = np.linalg.pinv(np.cov(((ref[FEATURES] - mean) / std).to_numpy(), rowvar=False))
    X = ((f[FEATURES] - mean) / std).to_numpy()
    ixi = pd.DataFrame({"scan_id": f["subject_id"], "reliability_D": [np.sqrt(max(x @ pinv @ x, 0)) for x in X]})
    e = pd.read_csv(RES / "enki_features_with_D.csv")
    enki = pd.DataFrame({"scan_id": "eNKI_" + e["participant_id"], "reliability_D": e["D"]})
    return pd.concat([ixi, enki], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # IXI (all sites incl. IOP): DeepBrainNet from the original laptop run + 3 tabular models
    dbn0 = pd.read_csv(RES / "predictions_deepbrainnet.csv").rename(columns={
        "subject_id": "scan_id", "chronological_age": "age", "predicted_age": "dbn_original_run"})
    ixi = dbn0[["scan_id", "site", "age", "sex", "ethnicity", "dbn_original_run"]].assign(cohort="IXI")
    for m, col in [("linearregression", "linear_regression"), ("randomforest", "random_forest"), ("xgboost", "xgboost")]:
        t = pd.read_csv(RES / f"predictions_{m}.csv")[["subject_id", "predicted_age"]]
        ixi = ixi.merge(t.rename(columns={"subject_id": "scan_id", "predicted_age": col}), on="scan_id", how="left")

    # Colab run (IXI Guys+HH + eNKI): both CNNs, BrainAgeNeXt's 5 models, ridge heads
    b = pd.read_csv(RES / "brainagenext_predictions.csv").rename(columns={
        "predicted_age": "brainagenext", "pred_ridge_head": "brainagenext_ridge",
        **{f"pred_model{i}": f"brainagenext_model{i}" for i in range(1, 6)}, "ensemble_sd": "brainagenext_sd"})
    t = pd.read_csv(RES / "transfer_head_predictions.csv").rename(columns={
        "dbn_raw": "deepbrainnet", "pred DBN gap512": "dbn_gap512_ridge", "pred DBN dense1024": "dbn_dense1024_ridge",
        "pred DBN gap + BANG": "dbn_gap_plus_brainagenext_ridge"})
    colab = b.drop(columns=["orientation"]).merge(
        t[["scan_id", "deepbrainnet", "dbn_gap512_ridge", "dbn_dense1024_ridge", "dbn_gap_plus_brainagenext_ridge"]],
        on="scan_id")
    colab = colab.rename(columns={"dataset": "cohort", "chronological_age": "age"})

    # SALD (held out)
    s = pd.read_csv(RES / "sald_predictions.csv").rename(columns={
        "dbn_raw": "deepbrainnet", "bang_raw": "brainagenext", "bang_sd": "brainagenext_sd",
        "pred DBN dense1024 + ridge": "dbn_dense1024_ridge", "pred BANG 512 + ridge": "brainagenext_ridge",
        "pred DBN gap + BANG + ridge": "dbn_gap_plus_brainagenext_ridge",
        **{f"bang_model{i}": f"brainagenext_model{i}" for i in range(1, 6)}}).drop(columns=["age_bin"])
    s = s.assign(cohort="SALD", site="SALD", ethnicity="Chinese (cohort)")

    # combine: IXI rows get the Colab columns where they exist; eNKI and SALD rows as they are
    ixi_all = ixi.merge(colab[colab["cohort"] == "IXI"].drop(columns=["cohort", "site", "age", "sex", "ethnicity"]),
                        on="scan_id", how="left")
    enki = colab[colab["cohort"] == "eNKI"]
    allp = pd.concat([ixi_all, enki, s], ignore_index=True).merge(distance_D(), on="scan_id", how="left")
    allp["in_brainagenext_training_data"] = allp["cohort"].map({"IXI": "yes (openBHB)", "eNKI": "yes (NKI-RS)",
                                                                 "SALD": "no (held out)"})
    allp["brainagenext_run_on_this_scan"] = allp["brainagenext"].notna()
    cols = ["scan_id", "cohort", "site", "age", "sex", "ethnicity", "in_brainagenext_training_data",
            "deepbrainnet", "dbn_original_run", "brainagenext", *[f"brainagenext_model{i}" for i in range(1, 6)],
            "brainagenext_sd", "dbn_gap512_ridge", "dbn_dense1024_ridge", "brainagenext_ridge",
            "dbn_gap_plus_brainagenext_ridge", "linear_regression", "random_forest", "xgboost", "reliability_D"]
    allp = allp[cols].sort_values(["cohort", "scan_id"])
    allp.round(4).to_csv(OUT / "all_predictions.csv", index=False)

    metrics = {name: pd.read_csv(RES / f).to_dict("records") for name, f in [
        ("mae_by_cohort_with_ci", "sald_summary.csv"), ("transfer_heads_ixi_enki", "transfer_head_mae.csv"),
        ("transfer_heads_sald", "sald_mae.csv"), ("fairness_brainagenext_vs_deepbrainnet", "bias_report_brainagenext.csv"),
        ("sald_sex_audit", "sald_sex_audit.csv"), ("reliability_score_vs_error", "bang_reliability_check.csv"),
        ("brainagenext_uncertainty", "brainagenext_uncertainty.csv"), ("four_model_fairness_ixi", "bias_report.csv")]}
    metrics["brainagenext_orientation"] = json.loads((RES / "bang_orientation.json").read_text())
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    print(f"all_predictions.csv: {len(allp)} rows  {allp['cohort'].value_counts().to_dict()}")
    print("non-empty values per model column:")
    print(allp[cols[7:]].notna().sum().to_string())


if __name__ == "__main__":
    main()
