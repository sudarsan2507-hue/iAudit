"""Stage T2 - train 3 plain tabular regressors on the volumetric features,
producing leak-free out-of-fold (OOF) predictions for the fairness audit.

No hyperparameter tuning by design: these models are expected to be less
accurate than DeepBrainNet (a CNN trained on raw MRI). The comparison of
interest is bias SHAPE (does the IOP site effect / sex split replicate),
not accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from xgboost import XGBRegressor

FEATURES = pd.read_csv(r"D:\Projects\iAuditAge\results\features.csv")

FEATURE_COLS = [
    "volume_CSF_ml", "volume_GM_ml", "volume_WM_ml", "volume_deepGM_ml",
    "volume_brainstem_ml", "volume_cerebellum_ml", "total_brain_volume_ml",
    "frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
    "frac_cerebellum",
]
assert len(FEATURE_COLS) == 13

X = FEATURES[FEATURE_COLS].to_numpy()
y = FEATURES["chronological_age"].to_numpy()
site = FEATURES["site"].to_numpy()
n = len(FEATURES)

MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(random_state=42),
    "XGBoost": lambda: XGBRegressor(random_state=42),
}

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
folds = list(skf.split(X, site))

# ---------------- LEAKAGE ASSERTION ----------------
seen_test_idx = set()
for fold_i, (train_idx, test_idx) in enumerate(folds):
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    assert train_set.isdisjoint(test_set), f"fold {fold_i}: train/test overlap!"
    assert test_set.isdisjoint(seen_test_idx), f"fold {fold_i}: subject already tested in another fold!"
    seen_test_idx |= test_set
assert seen_test_idx == set(range(n)), "not every subject got exactly one OOF prediction!"
print("OOF verified: no train/test overlap")

oof_preds = {name: np.full(n, np.nan) for name in MODELS}

for name, make_model in MODELS.items():
    for train_idx, test_idx in folds:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]

        model = make_model()
        model.fit(X_train, y_train)
        oof_preds[name][test_idx] = model.predict(X_test)

assert not np.isnan(oof_preds[name]).any(), f"{name}: not all subjects got an OOF prediction"

# ---------------- VERIFY T2 ----------------
print("\n" + "=" * 60)
print("VERIFY T2")
print("=" * 60)

for name in MODELS:
    pred = oof_preds[name]
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    r, _ = stats.pearsonr(y, pred)
    flag = ""
    if r < 0.5 or mae > 12:
        flag = "  *** FLAG: r<0.5 or MAE>12 ***"
    print(f"{name:18s} MAE={mae:.2f}y  RMSE={rmse:.2f}y  R2={r2:.3f}  Pearson r={r:.3f}{flag}")

# watch-item check from PRE-CHECK: IXI442 (max total_brain, IOP)
watch = FEATURES[FEATURES["subject_id"] == "IXI442"]
if len(watch):
    wi = watch.index[0]
    print(f"\nwatch-item IXI442 (max total_brain, IOP) predicted ages:")
    for name in MODELS:
        print(f"  {name}: predicted={oof_preds[name][wi]:.1f}y  actual={y[wi]:.1f}y")

# save OOF predictions for Stage T3 to consume
oof_df = FEATURES[["subject_id", "chronological_age", "sex", "site", "ethnicity"]].copy()
for name in MODELS:
    oof_df[f"pred_{name}"] = oof_preds[name]
oof_df.to_csv(r"D:\Projects\iAuditAge\results\_oof_predictions.csv", index=False)
print(f"\nwrote results\\_oof_predictions.csv ({len(oof_df)} rows)")
