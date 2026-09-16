# iAudit — Brain-Age Fairness Audit (DeepBrainNet on IXI)

**Status:** Stages 0–5 complete. Stage 4B (second model) skipped by decision — see below.

---

## 1. Environment (Stage 0)

- Python 3.11.9 venv at `E:\iAuditData\venv` (kept off C: — pip cache and model weights redirected too)
- antspyx 0.6.3, antspynet 0.3.2, TensorFlow 2.21
- **Fix required:** DeepBrainNet ships as a legacy Keras-2 HDF5 model; Keras 3 can't rebuild it (`dense_3 expects 1 input(s), but received 2`). Resolved by installing `tf-keras` and setting `TF_USE_LEGACY_KERAS=1` before TensorFlow import.

## 2. Data (Stage 1)

- **Source:** the official IXI host (`biomedic.doc.ic.ac.uk`) returns 403 Forbidden for everyone, browser included. Sourced the same raw files from NITRC's public XNAT mirror (T1 NIfTIs, original filenames) and a GitHub-hosted copy of the genuine `IXI.xls` (authored by IXI's PI, Jo Hajnal).
- **Data-quality bug found and fixed:** `IXI.xls` had 52 duplicate `IXI_ID` rows (26 subjects) — a naive join would have corrupted ~5% of the cohort. Resolved by preferring the row with a non-missing ethnicity code.
- **Join independently verified:** all subjects' joined ages cross-checked exactly against XNAT's separately-recorded session `age` field.
- **29/619 subjects** in `IXI.xls` have no DOB on file (age unrecoverable) — excluded before inference rather than wasting compute predicting on them.
- **581 T1 scans downloaded** — the complete available IXI T1 archive (of 584 sessions, 3 have no T1 modality).

## 3. Single-scan smoke test (Stage 2)

IXI090: predicted 52.8y vs actual 41.8y. Preprocessing pipeline (brain extraction, rigid+affine registration to template) completed cleanly. *(Note: DeepBrainNet's registration step samples stochastically, so repeat runs on the same scan vary by ~±1.5y — expected, not a bug.)*

## 4. Pilot batch, n=20 (Stage 3)

| Metric | Value | Threshold | Result |
|---|---|---|---|
| Pearson r | 0.908 | >0.6 | PASS |
| MAE | 5.52y | <8y | PASS |

**PILOT GATE PASSED.**

## 5. Full-scale inference (Stage 4A)

Ran DeepBrainNet over the entire available IXI T1 cohort, with checkpointed resume (the job was interrupted many times by session/terminal boundaries — every restart resumed from the last completed scan with zero lost work).

| | n | Pearson r | MAE |
|---|---|---|---|
| Pilot | 20 | 0.908 | 5.52y |
| Mid-run checkpoint | 482 | 0.909 | 5.47y |
| **Final** | **560** | **0.910** | **5.45y** |

- **560 succeeded, 3 failed** (IXI074, IXI265, IXI527 — all corrupted/empty downloads, not model or data errors)
- No NaNs in predictions
- r and MAE essentially unchanged across a 28× scale-up — strong evidence the signal is real, not a small-sample artifact

**PILOT GATE PASSED at full scale.**

## 6. Second model (Stage 4B) — **skipped by decision**

Given the session had already consumed significant time on infrastructure issues (NITRC network hangs, Keras version conflict, repeated process restarts), and Stage 4B was explicitly a "nice-to-have, don't sink the session" item, we chose to go straight to the fairness analysis on DeepBrainNet alone rather than risk another multi-hour dependency saga installing a second model (BrainAgeNeXt/pyment). This is a scope decision, not a blocker — a second model could be added in a future session for direct bias-profile comparison.

## 7. Fairness audit (Stage 5)

### Method
- `gap = predicted_age − chronological_age`
- **Age-bias correction:** fit one linear regression `gap ~ chronological_age` on the pooled cohort (never per-subgroup, which would absorb the bias being measured); `corrected_gap = gap − (a·age + b)`. Mirrors de Lange & Cole's standard brain-age bias correction. *Limitation, stated for the record: fitting on the same sample being audited removes this cohort's population-average trend, not an independent reference trend.*
- Subgroups stratified by sex, site, ethnicity (only tested if ≥2 groups have n≥10); intersectional sex×site cells also reported.
- 2-group comparisons: Mann-Whitney U. 3+ groups: Kruskal-Wallis + Bonferroni-corrected pairwise post-hoc, with median difference as effect size.

### Sanity checks — all passed
| Check | Result |
|---|---|
| \|r(corrected_gap, age)\| < 0.10 | PASS (r = 0.0000) |
| Bland-Altman slope before correction (expect negative) | **−0.231** — PASS |
| Bland-Altman slope after correction (expect ~flat) | **0.000** — PASS |
| Mean corrected_gap ≈ 0 (±0.2y) | **+0.000** — PASS |
| Subgroup n's sum to total | 560 / 560 — PASS |

### Findings

**Sex — statistically significant.**

| | n | MAE | mean corrected gap |
|---|---|---|---|
| Female | 311 | 4.91y | **−0.80y** |
| Male | 249 | 4.14y | **+1.00y** |

Mann-Whitney p = **0.00016**. The model reads male brains as ~1.8–2.0 years older than female brains, relative to true age — a real effect, not chance.

**Site — the largest, most robust disparity.**

| | n | MAE | mean corrected gap |
|---|---|---|---|
| Guys | 312 | 4.14y | −0.30y |
| HH | 180 | 4.29y | −1.33y |
| **IOP** | **68** | **7.28y** | **+4.89y** |

Kruskal-Wallis p = **6.0 × 10⁻¹¹**. IOP is a clear outlier: substantially worse accuracy (MAE 7.3y vs ~4.2y elsewhere) and a strong positive bias (reads brains ~5 years older than true age). Guys-vs-IOP and HH-vs-IOP both significant after Bonferroni correction (p ≈ 7×10⁻⁹ and 5×10⁻¹⁰); Guys-vs-HH is not significant. **This pattern (accuracy tracking site, not demographics) points to a scanner/acquisition-protocol effect rather than a population effect.**

**Ethnicity — suggestive, not confirmed.** Omnibus Kruskal-Wallis p = 0.046 (borderline), but *no* individual pairwise comparison survives Bonferroni correction — Black/BlackBrit (n=15) and Chinese (n=14) are too small to confirm anything definitively, despite Asian/AsianBrit and Black/BlackBrit both showing a numerically large positive gap (+2 to +3y).

**Intersectional (sex × site):** IOP females are the worst-served group — MAE 8.1y, mean gap +5.0y, n=44 (adequately powered, not flagged underpowered). All 6 sex×site cells had n≥10.

**Multi-model comparison:** not available (only DeepBrainNet run this session).

### Headline takeaway

> DeepBrainNet shows a **real, statistically significant sex bias** (better accuracy for female subjects) and a **much larger, highly significant site-driven bias concentrated at the IOP scanner site** — the model reads IOP subjects' brains as nearly 5 years older than their true age, roughly double the error rate of the other two sites. The site effect looks like a scanner/protocol artifact rather than a demographic one, and is the single largest fairness concern in this audit.

---

## Deliverables

| File | Contents |
|---|---|
| `results/predictions_deepbrainnet.csv` | Per-subject predictions, fixed schema |
| `results/bias_report.csv` | Every subgroup × model row: n, MAE, mean gap, p-value |
| `results/fairness_checklist.md` | Machine-written summary of which axes were tested/skipped and why |
| `results/fig_pred_vs_actual.png` | Scatter, predicted vs actual age |
| `results/fig_boxplot_sex_DeepBrainNet.png` | Corrected gap distribution by sex |
| `results/fig_mae_by_site_DeepBrainNet.png` | MAE by site |
| `results/fig_bland_altman_DeepBrainNet.png` | Gap vs age, before/after correction |
| `results/fig_multimodel_subgroup_comparison.png` | Scaffold for future multi-model comparison |
| `scripts/*.py` | All pipeline code (fetch, demographics, inference, fairness audit) |

## Known limitations / caveats for anyone reading these results

1. **Single model.** No second brain-age model to check whether the site/sex effects are DeepBrainNet-specific or general.
2. **Ethnicity underpowered** for two of five categories — treat the ethnicity findings as suggestive, not confirmed.
3. **Age-correction fit on the same audited sample** (standard practice, but a real methodological limitation — see Method above).
4. **3 corrupted downloads** excluded (IXI074, IXI265, IXI527) — negligible (0.5% of cohort), unrelated to model behavior.
5. **IOP has the smallest site n (68)** — the effect is large and highly significant, but a truly independent replication sample would strengthen it further.
