"""Check that every number the website shows matches the project's result files.

Run:  python website/source/verify_data.py   (needs pandas)
Compares each case's age and every model prediction in website/data/cases.json with
website/data/all_predictions.csv (itself rebuilt from results/ by export_all_values.py), and the
cohort table with results/sald_summary.csv. Exits non-zero on any mismatch.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
COLS = {"DeepBrainNet": "deepbrainnet", "BrainAgeNeXt": "brainagenext",
        "DBN GAP-512 + ridge": "dbn_gap512_ridge", "DBN dense-1024 + ridge": "dbn_dense1024_ridge",
        "BrainAgeNeXt + ridge": "brainagenext_ridge", "DBN GAP + BrainAgeNeXt + ridge": "dbn_gap_plus_brainagenext_ridge",
        "Linear regression": "linear_regression", "Random forest": "random_forest", "XGBoost": "xgboost"}


def main():
    cases = json.loads((ROOT / "data" / "cases.json").read_text(encoding="utf-8"))
    allp = pd.read_csv(ROOT / "data" / "all_predictions.csv").set_index("scan_id")
    bad, checked = [], 0
    for c in cases["cases"]:
        row = allp.loc[c["id"]]
        if abs(row["age"] - c["age"]) > 0.051:
            bad.append(f"{c['id']} age {c['age']} vs {row['age']}")
        for name, col in COLS.items():
            if name in c["preds"]:
                checked += 1
                if abs(row[col] - c["preds"][name]) > 0.006:
                    bad.append(f"{c['id']} {name}: site {c['preds'][name]} vs results {row[col]}")
        for i, v in enumerate(c["preds"].get("BrainAgeNeXt models", []), 1):
            checked += 1
            if abs(row[f"brainagenext_model{i}"] - v) > 0.006:
                bad.append(f"{c['id']} BrainAgeNeXt model {i}: {v} vs {row[f'brainagenext_model{i}']}")
    summ = pd.read_csv(REPO / "results" / "sald_summary.csv")
    for r in cases["summary"]:
        m = summ[(summ.cohort == r["cohort"]) & (summ.model == r["model"])].iloc[0]
        checked += 1
        if abs(m["MAE"] - r["MAE"]) > 1e-6:
            bad.append(f"summary {r['cohort']} {r['model']}: {r['MAE']} vs {m['MAE']}")
    print(f"checked {checked} values across {len(cases['cases'])} cases + cohort table: "
          f"{'all match' if not bad else f'{len(bad)} MISMATCHES'}")
    for b in bad:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
