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

See `results/SESSION_REPORT.md` for the full Stage 0–5 writeup. Summary of
the full pipeline (Stage 0 through the reliability score) below.

### Per-model accuracy (CV / out-of-fold, n=544 common cohort)

| Model | MAE | R² | Pearson r |
|---|---|---|---|
| DeepBrainNet (CNN, full n=560) | 5.45y | — | 0.910 |
| LinearRegression | 7.13y | 0.695 | 0.834 |
| RandomForest | 6.76y | 0.716 | 0.846 |
| XGBoost | 7.20y | 0.681 | 0.827 |

Tabular models are less accurate than the CNN by design (no hyperparameter
tuning, simple volumetric features) — the point of comparison is bias
*shape*, not accuracy.

### Cross-model fairness comparison (Stage T4, common cohort n=544)

**MAE by site — IOP is the worst site for all 4 models:**

| Site | DeepBrainNet | LinearRegression | RandomForest | XGBoost |
|---|---|---|---|---|
| Guys | 4.43y | 6.62y | 6.24y | 6.72y |
| HH | 4.87y | 6.84y | 6.67y | 7.25y |
| **IOP** | **11.49y** | **10.25y** | **9.33y** | **9.23y** |

This is the headline finding: IOP degrades every architecture tested,
regardless of whether the model sees the raw image (DeepBrainNet) or
simple tissue-volume features (tabular models) — consistent with a
scanner/acquisition artifact at the data level, not a DeepBrainNet quirk.

Stage V confirms this mechanistically for the tabular pathway: 5/6
tissue-volume fractions remain significantly shifted at IOP after
age-adjustment (Bonferroni-corrected Mann-Whitney U), i.e. IOP scans are
genuinely off-distribution at the feature level, independent of the
younger age mix at that site.

The male-older/female-younger directional bias seen in DeepBrainNet does
**not** cleanly replicate in the tabular models — see
`results/fairness_checklist.md` and `results/bias_report.csv` for the
full subgroup breakdown.

### Unsupervised reliability score

A Mahalanobis distance from the Guys+HH ("seen") feature distribution,
fit **without ever using IOP**, was tested as a model-agnostic reliability
signal:

- **Recovers IOP unsupervised:** ROC AUC = 0.967 (distance alone, no site
  labels used to fit it).
- **Predicts per-subject error, age-controlled:** real and significant for
  DeepBrainNet (partial Spearman r=0.215, p<0.00001); null for all 3
  tabular models (p>0.5) — a genuine negative result, not glossed over.
- **Selective prediction beats random abstention** for all 4 models, but
  the effect size is only practically meaningful for DeepBrainNet
  (0.80y MAE drop at 80% coverage vs ~0.02–0.32y for the tabular models).

### Key figures

| | |
|---|---|
| `fig_pred_vs_actual.png` | DeepBrainNet predicted vs actual age |
| `fig_crossmodel_site_bias.png` / `_sex_bias.png` | Corrected gap by site/sex, all 4 models side by side |
| `fig_bland_altman_*.png` | Gap vs age, before/after correction, per model |
| `fig_distance_by_site.png` | Reliability-score distance by site (IOP separation) |
| `fig_error_vs_distance.png` | Per-subject error vs reliability score, per model |
| `fig_selective_prediction.png` | MAE vs coverage, reliability-score-based vs random abstention |
