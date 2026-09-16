# iAudit — Brain-Age Fairness Audit

A fairness audit of brain-age prediction on the [IXI](https://brain-development.org/ixi-dataset/)
neuroimaging dataset. DeepBrainNet (a CNN trained on raw T1 MRI) is compared
against three plain tabular regressors trained on simple volumetric features,
to test whether the fairness disparities found in one architecture are
scanner/acquisition artifacts (replicate across architectures) or
model-specific.

## Pipeline

| Stage | What | Script |
|---|---|---|
| 0–1 | Environment setup; join T1 filenames to IXI demographics | `scripts/load_demographics.py`, `scripts/fetch_ixi_nitrc.py` |
| 2 | Single-scan DeepBrainNet smoke test | `scripts/predict_one.py` |
| 3 | 20-scan pilot batch, pass/fail gate | `scripts/run_pilot.py` |
| 4A | Full-cohort (n=560) DeepBrainNet inference, checkpointed | `scripts/run_full.py` |
| 5 | Fairness audit: pooled age-bias correction, subgroup stats | `scripts/fairness_audit.py` |
| T1 | Volumetric feature extraction (`deep_atropos`, 6-tissue segmentation) | `scripts/extract_features.py` |
| T2 | Train 3 plain tabular models (LinearRegression, RandomForest, XGBoost), leak-free 5-fold out-of-fold predictions | `scripts/train_tabular.py` |
| T3 | Write tabular predictions in shared schema; re-run the audit unchanged on all 4 models | `scripts/write_tabular_predictions.py` |
| T4 | Cross-model comparison on the common cohort | `scripts/compare_models.py` |
| V | IOP feature-distribution check (age-adjusted, is IOP off-distribution at the data level?) | `scripts/check_iop_features.py` |
| — | Unsupervised Mahalanobis reliability score (feature-space distance from the "seen" Guys+HH distribution) | `scripts/reliability_score.py` |

Every stage is checkpointed/resumable where it runs long, and every number
in `results/` comes from actually running these scripts on the real IXI
cohort — nothing is hand-entered.

## Data

- Source: T1 NIfTI scans and demographics (`IXI.xls`) via NITRC's public
  XNAT mirror (the official IXI host returns 403 for direct downloads).
- 560 subjects with a usable T1 scan and known age/sex/site/ethnicity;
  544 of those also have volumetric features (16 dropped to transient
  `deep_atropos` MemoryErrors during segmentation, caught and logged —
  see `scripts/extract_features.py`).
- Sites: Guys, HH, IOP (three different scanners/acquisition protocols).

## Setup

```
python -m venv venv
venv\Scripts\pip install antspyx antspynet tf-keras scikit-learn xgboost scipy pandas matplotlib
```

DeepBrainNet ships as a legacy Keras-2 model; set `TF_USE_LEGACY_KERAS=1`
before importing TensorFlow (already done in every script that needs it).

## Results

See `results/SESSION_REPORT.md` for the full Stage 0–5 writeup, and the
Results section below for the tabular-model comparison and reliability
score.
