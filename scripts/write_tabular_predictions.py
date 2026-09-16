"""Stage T3 (part 1) - write the 3 tabular models' OOF predictions in the
same schema as predictions_deepbrainnet.csv so fairness_audit.py picks them
up unchanged.
"""
import pandas as pd

RESULTS = r"D:\Projects\iAuditAge\results"
oof = pd.read_csv(fr"{RESULTS}\_oof_predictions.csv")

MODEL_COL_TO_NAME = {
    "pred_LinearRegression": "LinearRegression",
    "pred_RandomForest": "RandomForest",
    "pred_XGBoost": "XGBoost",
}

SCHEMA = ["subject_id", "chronological_age", "predicted_age", "sex", "site", "ethnicity", "model_name"]

for col, model_name in MODEL_COL_TO_NAME.items():
    out = oof[["subject_id", "chronological_age", "sex", "site", "ethnicity"]].copy()
    out["predicted_age"] = oof[col]
    out["model_name"] = model_name
    out = out[SCHEMA]
    fname = f"predictions_{model_name.lower()}.csv"
    out.to_csv(fr"{RESULTS}\{fname}", index=False)
    print(f"wrote {fname}  ({len(out)} rows, model_name={model_name})")
