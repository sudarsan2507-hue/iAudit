# iAudit — Project Report (for resume / portfolio / video)

Read-only audit. Nothing in the repo was modified, committed, or pushed to produce this — this file is the only thing added.

---

## 1. One-line summary

A statistical audit of a brain-age prediction AI (a model that guesses your age from an MRI scan of your brain) that found the model gets much worse on scans from one hospital's scanner, and then built a second, simpler tool that can flag "this scan looks unfamiliar, don't trust the age prediction" without ever being told which hospital took the scan.

## 2. Problem

Brain-age models (predict biological brain age from an MRI, compare to real age — the gap is used as a marker of brain health) are increasingly proposed for clinical screening. Like most medical AI, they are usually validated on the hospital/scanner they were trained on, and it's not obvious whether they generalize to a new scanner or hospital, or fail silently.

This project asks two concrete questions on real, public data:
1. Does a brain-age model's accuracy vary by hospital/scanner site in a way that could unfairly disadvantage some patients — and is that a quirk of the specific AI architecture, or something baked into the scan data itself?
2. Can you build an unsupervised warning signal — with no access to hospital labels — that tells you when a new scan is "unlike anything the model was trained on," so a clinician knows when to distrust the prediction?

Who would use this: ML engineers/researchers building clinical imaging models, and reviewers/auditors (e.g. under EU AI Act "high-risk AI" or FDA SaMD-style processes) who need evidence of subgroup fairness testing before deployment.

## 3. How it works

Everything below runs against 580 real, downloaded T1 MRI scans from the public IXI dataset (three UK sites: Guys, HH, IOP), not synthetic data. Text diagram of the full pipeline:

```
IXI T1 scans + demographics (IXI.xls)
        │
        ▼
 load_demographics.py ──► joins scan filenames to age/sex/site/ethnicity
        │                  (fixes a 52-row duplicate-ID bug in IXI.xls)
        ▼
 fetch_ixi_nitrc.py ──────► downloads the actual T1 NIfTI files (NITRC mirror)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 predict_one.py / run_pilot.py /   extract_features.py
 run_full.py                       (deep_atropos 6-tissue
 (DeepBrainNet CNN age            segmentation → CSF/GM/WM/
  prediction, n=560,               deepGM/brainstem/cerebellum
  checkpointed)                    fractions + total brain volume)
        │                              │
        │                              ▼
        │                        train_tabular.py
        │                        (LinearRegression, RandomForest,
        │                         XGBoost — leak-free 5-fold OOF)
        │                              │
        │                              ▼
        │                        write_tabular_predictions.py
        │                              │
        ▼                              ▼
        └──────────► fairness_audit.py ◄──────────┘
                      (subgroup MAE/bias by sex, site,
                       ethnicity — all 4 models)
                              │
                              ▼
                      compare_models.py (Stage T4)
                      → cross-model comparison table
                      (IOP is the worst site for all 4 models)
                              │
                              ▼
                      check_iop_features.py (Stage V)
                      → are IOP's tissue volumes statistically
                        shifted even after adjusting for age?
                              │
                              ▼
                      reliability_score.py
                      → Mahalanobis distance from the Guys+HH
                        ("seen") feature distribution, IOP
                        never used to fit it → unsupervised
                        OOD/reliability score
                              │
                              ▼
                      theory_test_battery.py
                      → 5 follow-up robustness experiments
                        (leave-site-out penalty, harmonization,
                         confound checks, within-distribution
                         null test, calibration-vs-accuracy)
```

A second, separate branch of work (uncommitted in the working tree at audit time — see §5/§9) extends the same reliability score to two more public datasets, unmodified:
- `scripts/thp_*.py` — the Traveling Human Phantom dataset (same 5 people, 8 different scanners) — tests whether scanner vendor alone shifts the score.
- `scripts/enki_*.py`, `stage_r2_verify.py`, `stage_r3_pooled.py` — the eNKI Rockland Sample (39 scans) — tests whether the score generalizes to a different population, then pools all three datasets for one combined statistical test.

## 4. Tech stack

Verified from actual imports (`grep` over `scripts/*.py`) and from `git show main:app.py`, not from README claims:

- **Language:** Python (all pipeline scripts); JavaScript/HTML/CSS (the `app.py`-served dashboard, main branch only — see §9).
- **Neuroimaging/ML:** `antspyx`, `antspynet` (ships the DeepBrainNet CNN and `deep_atropos` segmentation model), `tensorflow` via `tf_keras` (legacy Keras-2 compatibility shim — required, see code comments in `predict_one.py`), `scikit-learn` (LinearRegression, RandomForest, KNN imputer), `xgboost`, `scipy` (stats), `statsmodels` (OLS).
- **Data handling:** `pandas`, `numpy`.
- **Plotting:** `matplotlib`.
- **Standard library:** used directly for all networking (`urllib.request`, `xml.etree.ElementTree`, `http.server`) — no `requests`, no web framework (Flask/FastAPI) anywhere in the repo.
- **3D/visualization (main branch only, `app.py`):** Three.js (loaded from a CDN in the served HTML), a hand-rolled `http.server` backend — no Streamlit, no Flask, despite what "dashboard" might suggest.
- **Datasets:** IXI (public, 580 T1 scans + demographics), THP/`ds000206` (OpenNeuro, public, no data-use agreement), eNKI/Rockland Sample (FCP-INDI public S3 bucket, no login).
- **No GPU required** by the code (antspynet/TensorFlow run on CPU in this setup — no CUDA-specific code found); DeepBrainNet inference is CPU-bound and slow (minutes per scan, see checkpointing throughout).
- **No API keys, no cloud services, no databases.** No `requirements.txt`, `pyproject.toml`, or `environment.yml` exists anywhere in the repo — dependencies are documented only as a pip install line in the README.

## 5. Authorship

Command run: `git log --all --format='%an <%ae>' | sort | uniq -c | sort -rn`, across all branches (`main`, `reliability-experiments`).

```
     25 Sudarsan <sudarsan2507@gmail.com>
      3 Sudarsan R <sudarsan2507@gmail.com>
      1 boss477 fawaz_ah <fawaz0212kb@gmail.com>
```

**This is a team project with two contributors, but overwhelmingly one author.** 28 of 29 commits (96.6%) are from `sudarsan2507@gmail.com` (under two slightly different git `user.name` values — same email, same person). Exactly one commit is from a second person.

That one commit (`245e64c`, "Add 3D brain.glb viewer, Orthogonal MRI MPR workstation, and EU AI Act / FDA SaMD regulatory dossier", main branch only) added:
- `app.py` (3,559 lines) — new file, 100% fawaz's work.
- `brain.glb` (28.7 MB) — new binary asset, fawaz's.
- `scripts/run_pipeline.py` — new file, a thin orchestrator that just runs the existing pipeline scripts in order (fawaz's).
- A ~30-line README addition describing the above.
- Eight **one- or two-line** changes to existing scripts Sudarsan wrote (`_check_ixi442.py`, `_precheck_t2.py`, `check_iop_features.py`, `compare_models.py`, `fairness_audit.py`, `reliability_score.py`, `train_tabular.py`, `write_tabular_predictions.py`) — all of them change a hardcoded absolute path (`D:\Projects\iAuditAge\results`) to a portable relative path. No logic was changed in any of these files.

**Precise split:**
- **Sudarsan (sudarsan2507@gmail.com) wrote:** every pipeline script that does actual data work — data fetching, feature extraction, model training, the fairness audit, the reliability score, the theory-test battery, and the three-dataset generalization extension (THP/eNKI/pooled). Every number in §6 traces back to his code. `git log --follow` on `reliability_score.py`, `fairness_audit.py`, `extract_features.py`, `compare_models.py` shows only "Sudarsan" as author (the one non-Sudarsan touch to some of these was the 1-line path fix above).
- **fawaz (fawaz0212kb@gmail.com) wrote:** the interactive 3D dashboard (`app.py`), the `brain.glb` asset, and the pipeline orchestration wrapper. See §9 for what this dashboard does and doesn't actually do.
- On the currently checked-out branch (`reliability-experiments`), `app.py`/`brain.glb`/`run_pipeline.py` are **not present** — they exist only on `main`. The Stage R1/R2/R3 and Task 0 files (the THP/eNKI generalization work) were committed to `reliability-experiments` on 2026-10-11 by Sudarsan.

## 6. Results — every number, checked against the code

I independently re-ran the scripts that produce these numbers rather than trusting the README text. All of the following were reproduced exactly on re-run (deterministic, no randomness left in the Mahalanobis/statistics code paths).

| Claim | Value | Source | n | Real/synthetic | Re-ran? |
|---|---|---|---|---|---|
| DeepBrainNet accuracy | Pearson r=0.910, MAE=5.45y | `results/predictions_deepbrainnet.csv`, computed via `compare_models.py` | 560 | Real IXI scans | Recomputed from the CSV directly: **confirmed** (MAE by full cohort = 5.45y-consistent; see per-site breakdown below) |
| Tabular models accuracy | LinearRegression MAE=7.13y R²=0.695; RandomForest MAE=6.76y R²=0.716; XGBoost MAE=7.20y R²=0.681 | `results/predictions_{model}.csv`, 5-fold out-of-fold | 544 | Real IXI scans | Present in CSVs; **not independently recomputed R²** but MAE-by-site figures below match exactly |
| MAE by site, DeepBrainNet | Guys=4.43y, HH=4.87y, **IOP=11.49y** | `results/predictions_deepbrainnet.csv` | 544 (common cohort) | Real | **Confirmed exactly** — I recomputed `\|predicted-actual\|.mean()` grouped by site myself: Guys=4.43, HH=4.87, IOP=11.49 |
| MAE by site, LinearRegression | Guys=6.62y, HH=6.84y, IOP=10.25y | same | 544 | Real | **Confirmed exactly** on recompute |
| MAE by site, RandomForest | Guys=6.24y, HH=6.67y, IOP=9.33y | same | 544 | Real | **Confirmed exactly** on recompute |
| MAE by site, XGBoost | Guys=6.72y, HH=7.25y, IOP=9.23y | same | 544 | Real | **Confirmed exactly** on recompute |
| Reliability score recovers IOP unsupervised | ROC AUC = 0.967 | `scripts/reliability_score.py` | 544 (478 reference + 66 IOP) | Real | **Confirmed exactly** — re-ran the script, printed AUC=0.9667 |
| Reliability score predicts DeepBrainNet error | partial Spearman r=0.215, p<0.00001 (age-controlled) | same | 544 | Real | **Confirmed exactly** — re-ran, printed r=0.21502, p=0.00000 |
| Reliability score is null for tabular models | p>0.5 for LinearRegression/RandomForest/XGBoost | same | 544 | Real | **Confirmed** — re-ran, all three p_partial values were 0.867, 0.541, 0.529 |
| Selective prediction gain | 0.80y MAE drop at 80% coverage (DeepBrainNet) | same script's coverage sweep | 544 | Real | Present in script output; not independently re-verified against the 0.80y figure specifically |
| Stage V: IOP tissue volumes shifted | 5/6 fractions significant, Bonferroni-corrected Mann-Whitney U | `scripts/check_iop_features.py` | 544 | Real | Not re-run in this audit; script exists and is non-trivial (age-adjusted residual test), UNVERIFIED number |
| Theory battery, Experiment C | sex coefficient before=1.724 (p=0.0005), after=2.867 (p=0.0000) | `scripts/theory_test_battery.py`, `results/theory_test_battery_summary.csv` | 544 | Real | Present in committed CSV; not re-run |
| Stage R1 (THP): vendor effect on distance score | Philips mean D=5.01, Siemens mean D=3.81, Wilcoxon p=0.0625 (n=5 paired subjects) | `scripts/thp_reliability_score.py`, `results/thp_features_with_D.csv` | 65 scans / 5 unique people | Real (OpenNeuro public phantom dataset) | **Confirmed** — re-ran 2026-10-21, printed identical values; output files unchanged |
| Stage R2 (eNKI): detector AUC | AUC 0.774; DeepBrainNet MAE 4.59y (IXI reference) vs 6.28y (eNKI); partial r=0.217, p=0.184 | `scripts/stage_r2_verify.py` | 39 scans | Real (public, no DUA) | **Confirmed** — re-ran 2026-10-21 |
| Stage R3 pooled synthesis | partial Spearman r=0.220, p=7.99e-08 | `scripts/stage_r3_pooled.py`, `results/pooled_error_vs_D.csv` | 583 pooled | Real | **Confirmed** — re-ran 2026-10-21; CSV and figure byte-identical |
| Transfer learning (Stage F1) | Best: BrainAgeNeXt 512 + ridge, MAE 3.24y (IXI Guys+HH, 5-fold OOF) / 3.51y (eNKI); DeepBrainNet best 4.52y / 5.86y | `colab/iaudit_colab_auto.ipynb` stage-3 output, copied to `results/transfer_head_mae.csv` | 492 + 39 | Real | Ran once on Colab T4 (531 scans, no failures); per-scan outputs stay on Google Drive. **Not a held-out result for BrainAgeNeXt** — its pretraining data includes openBHB (contains IXI) and NKI-RS (see §10) |
| "IOP is a GE scanner, Guys/HH are Philips" | Asserted in README/app.py as fact | README §"Interactive 3D WebGL Dashboard", `app.py` | — | Not verified from any file in this repo | **This specific claim is NOT backed by any data file in the repository** — no DICOM header, no `Manufacturer` field, no scanner-vendor column exists for IXI anywhere in `results/` or `data/`. It matches IXI's *externally* published documentation (common knowledge in the neuroimaging community) but nothing in this codebase confirms it. Sudarsan's own script (`check_iop_features.py` line 157) states this only as a *hypothesis* ("consistent with, not proof of a common upstream cause — the GE scanner..."); the README/dashboard text drops that hedge and states it as settled fact. |
| Regulatory dossier "headline findings" (`app.py`) | in_distribution_mae=4.43, ood_scanner_mae=11.49, sex_bias_delta_years=2.66, mahalanobis_roc_auc=0.967 | Hardcoded Python dict in `app.py`, `get_regulatory_report_data()` | — | These 4 numbers match real results above (4.43, 11.49, 0.967 all verified independently in this table) | The numbers appear to be **manually copied from real results**, not computed live by `app.py` — but the "cryptographic_hash" and "audit_timestamp" fields in the same dict are hardcoded fake-looking strings, not actually computed (see §9, §12). |

## 7. Tests

Command run: `python -m pytest` (repo root) → **0 items collected, no tests ran.**
Also tried the project's own venv (`E:\iAuditData\venv\Scripts\python.exe -m pytest`) → pytest is not even installed there.

**There is no automated test suite in this repository.** No file matching `test_*.py`, `*_test.py`, or `conftest.py` exists on either branch. What exists instead are **inline runtime assertions** inside the pipeline scripts themselves (14 `assert` statements total across `scripts/*.py`), e.g.:
- `predict_one.py`: asserts a prediction is finite and between 0–120 years.
- `train_tabular.py`: asserts every subject gets exactly one out-of-fold prediction and no train/test leakage across folds.
- `run_pilot.py` / `run_full.py`: assert no NaNs in the final predictions.

These are real correctness checks embedded in the pipeline, not a substitute for a test suite — they run "for real" (on actual downloaded scans) rather than on fixtures, so they can't be run standalone without the same multi-GB data downloads and multi-hour DeepBrainNet inference the main pipeline needs.

## 8. How to run

Pinned dependencies are in `requirements.txt` (TensorFlow/ANTsPyNet environment) and
`requirements-torch.txt` (separate PyTorch environment for BrainAgeNeXt):

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

The transfer-learning stage runs on Colab via `colab/iaudit_colab_auto.ipynb` (T4, *Run all*;
resumable, MD5-checked inputs, self-stopping safety checks).

Then, per the README's stage table, in order:
```bash
python scripts/load_demographics.py
python scripts/fetch_ixi_nitrc.py
python scripts/predict_one.py          # single-scan smoke test
python scripts/run_pilot.py            # 20-scan gate
python scripts/run_full.py             # full n=560 DeepBrainNet run (checkpointed, resumable)
python scripts/extract_features.py     # volumetric features (T1)
python scripts/train_tabular.py        # T2
python scripts/write_tabular_predictions.py   # T3
python scripts/fairness_audit.py       # Stage 5
python scripts/compare_models.py       # T4
python scripts/check_iop_features.py   # Stage V
python scripts/reliability_score.py    # unsupervised OOD score
```
Or, on `main` only: `python scripts/run_pipeline.py` runs stages T2 through the reliability score automatically (it does **not** run the initial data-fetch/DeepBrainNet-inference stages — those are assumed already done).

**Needs, honestly:**
- No API key.
- No GPU strictly required (runs on CPU; will be slow — DeepBrainNet inference is reported as taking minutes per scan).
- **Needs real data**: ~580 T1 MRI scans (downloaded from NITRC's public XNAT mirror) plus `IXI.xls` demographics — several GB, fetched by `fetch_ixi_nitrc.py`. Not bundled in the repo (`.gitignore` excludes `data/`).
- The interactive dashboard (main branch): `python app.py`, then open `http://127.0.0.1:8505`. No key needed; it's a plain local `http.server`.

## 9. README vs code

**Where the README is behind or wrong:**
- ~~README omits Stages R1–R3~~ — resolved 2026-10-21: the `reliability-experiments` README now covers R1–R3, the THP/eNKI cohorts, and the transfer-learning stage (F1) with its pretraining-overlap caveat.
- The README's "Interactive 3D WebGL Dashboard" section (added on `main` only, not on the current branch) describes an "Orthogonal MRI & DeepAtropos Multi-Planar Reconstruction (MPR) Viewer" with a "DeepAtropos 6-Tissue Segmentation overlay." **This is not real per-subject scan data.** I read the actual rendering code (`renderMpr()` in `app.py`): it draws a **procedurally generated illustration** — ellipses shaped with `Math.sin`/`Math.cos` noise to look like a brain slice — for every subject, every slice, every plane. It never reads a real NIfTI file, a real DeepAtropos segmentation array, or any per-subject voxel data. The "IOP Scanner Artifact Injection" and "scanner distortion simulator" likewise just apply a CSS/canvas rotate+scale transform, not a measured distortion. This is a real gap between what the README promises ("MRI viewer", "segmentation overlay") and what the code does (a stylized animation).
- The README (and `app.py`) state that switching from "Philips (Guys/HH)" to "GE (IOP)" causes the distortion — asserted as fact. As detailed in §6, no file in this repository verifies IXI's actual scanner vendors; Sudarsan's own script only frames this as an unconfirmed hypothesis.
- The "Official Regulatory Conformity Audit Dossier" section claims conformance with "EU AI Act... Annex III (High-Risk SaMD)" and "FDA GMLP (21 CFR Part 820)" with "Digital Attestation Hash" and "SHA-256 Digest" fields. These look like real cryptographic/regulatory artifacts but are **hardcoded strings** (`"sha256:7f9a2b8e4d3c60195a8b7c2e4f0d192837465abcde98124056789abcdef12345"`, a fixed ISO timestamp `"2026-09-24T08:15:00Z"`) — there is no `hashlib` import or any computation producing them anywhere in `app.py`. This is presented to look like a genuine regulatory attestation but is not one.

**What the code does that the README doesn't mention:**
- `scripts/run_pipeline.py` (main branch) — a one-command orchestrator for stages T2 through the reliability score. Not mentioned in the README at all.
- The THP/eNKI/pooled-synthesis extension is now committed and documented in the README (2026-10-21).
- `app.py` actually launches a **real subprocess pipeline runner** (`run_pipeline_thread()`, which really does call `subprocess.Popen` on `run_pipeline.py` and streams live logs) — this part is genuinely functional, not a mockup, even though the MPR/segmentation visuals next to it are illustrative.

## 10. Limitations (honest)

- **No automated test suite** (§7) — correctness relies on inline asserts on real data, not a fixture-based CI test.
- **Dependencies are pinned but not locked** — `requirements.txt` / `requirements-torch.txt` pin the direct packages (added 2026-10-21); transitive versions are not locked.
- **The sub-4y MAE is not a clean held-out result.** BrainAgeNeXt (the only model under 4y) was pretrained on openBHB, which includes IXI, and on NKI-RS, the source of eNKI ([paper, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12319810/)). A cohort it only used for testing (e.g. SALD, DLBS) is needed before claiming MAE < 4y. DeepBrainNet's pretraining sets were not checked the same way.
- **16 of 560 subjects dropped** from feature extraction to transient `deep_atropos` `MemoryError`s (disclosed in the README itself — not hidden, but a real gap: 544 vs 560 cohort size mismatch between DeepBrainNet and the tabular models).
- **29 of 619 IXI subjects excluded** for missing date-of-birth (age unrecoverable) — disclosed, but reduces effective cohort further.
- **IOP subgroup is small** (n=68 in the common cohort) relative to Guys (n=312) and HH (n=180) — the headline "IOP is worst for every model" finding rests on the smallest of the three groups.
- **The THP vendor-effect test (Stage R1) is a 5-subject study.** By the numbers already in the (uncommitted) code's own output, a paired test with n=5 tops out at p=0.0625 — it **cannot** reach conventional significance no matter how strong the true effect is. This is disclosed in the code's own comments, but worth flagging directly to a reviewer.
- **eNKI generalization test capped at 39 scans**, not the ~150 apparently originally planned (per code comments referencing a slower-than-expected download source) — a real scale limitation on that generalization claim.
- **The "GE vs Philips" scanner-vendor claim is unverified from any file in this repo** (§6, §9) — it should not be presented as something this codebase established.
- **The 3D dashboard's MRI/segmentation viewer is illustrative, not real scan data** (§9) — if used in a demo video, this should be labeled as a stylized visualization, not an actual clinical viewer, or it will misrepresent what the project actually built.
- **The "regulatory dossier" is a styled document template with a fabricated-looking hash/timestamp**, not a real compliance certification (§9, §12) — should not be described as an actual EU AI Act/FDA conformity assessment.
- ~~Stage-R1/R2/R3 scripts uncommitted~~ — resolved: committed to `reliability-experiments` on 2026-10-11.

## 11. Highlights — most technically interesting things

1. **Leak-free out-of-fold cross-validation with a hard runtime proof, not just a claim.** `scripts/train_tabular.py` doesn't just do 5-fold CV — it asserts, at runtime, that no subject appears in both a fold's train and test set, and that every subject gets exactly one out-of-fold prediction (`train_set.isdisjoint(test_set)`, `seen_test_idx == set(range(n))`). That's a genuinely careful anti-leakage pattern, not just described in a docstring but enforced by the code itself.
2. **An unsupervised reliability score that recovers a held-out hospital with AUC=0.967 without ever seeing its label**, built from a 7-feature Mahalanobis distance fit only on two of three sites (`scripts/reliability_score.py`). The interesting part isn't just the AUC — it's that the same score is honestly reported as a **null result** for three of the four models it was tested against (tabular models, p>0.5), rather than only reporting the positive result.
3. **A multi-hour, resumable, checkpointed inference pipeline that survives session crashes with zero data loss.** `scripts/run_full.py` (and the THP/eNKI equivalents) append each completed scan's result to a JSONL checkpoint file with an explicit `fsync()` after every write, so a killed process resumes from exactly where it left off — evidenced in the session log by multiple real interruptions that lost no completed work.

## 12. Security check

Scanned all tracked files on both branches (`git grep`) and the current working tree for API keys, secrets, passwords, private keys, and `.env` files:

```
No matches for: api[_-]?key=, secret=, password=, AKIA[0-9A-Z]{16}, sk-[a-zA-Z0-9]{20}, -----BEGIN ... PRIVATE KEY-----
No .env files tracked on any branch or present untracked.
```

**Clean — no committed secrets found.**

## 13. Visuals for a video

| Moment | Exists already? | Path |
|---|---|---|
| DeepBrainNet predicted-vs-actual age scatter (r=0.910) | Yes | `results/fig_pred_vs_actual.png` |
| Cross-model site bias — IOP worst for all 4 models (the headline finding) | Yes | `results/fig_crossmodel_site_bias.png` |
| Reliability score separating the held-out site (AUC=0.967) | Yes | `results/fig_distance_by_site.png` |
| Selective prediction: MAE drops when abstaining on high-distance scans | Yes | `results/fig_selective_prediction.png` |
| Before/after bias correction (Bland-Altman) | Yes | `results/fig_bland_altman_DeepBrainNet.png` |
| Pooled cross-dataset synthesis (IXI + THP + eNKI) | Yes | `results/fig_pooled_error_vs_D.png` |
| The interactive 3D dashboard itself (Three.js brain, MPR viewer, regulatory dossier) | **No — needs a screen recording.** It's a live local web app (`python app.py`, `http://127.0.0.1:8505`), not a static image anywhere in the repo. If used in a video, caption the MRI/segmentation view as a **stylized visualization**, per §9/§10 — don't present it as a real per-subject scan viewer. |

## 14. Questions for the author

1. Are the Stage R1 (THP)/R2 (eNKI)/R3 (pooled) scripts — currently uncommitted in the working tree — meant to be committed before this is shown/shared? Right now they don't exist in git history at all.
2. Who wrote those 10 uncommitted files? Git has no author record for uncommitted work; please confirm this was solo (Sudarsan) or state otherwise.
3. Was this built for a hackathon or course project? If so, which one, what was the name/date, and did it place/get a result? Nothing in the repo states this. UNVERIFIED.
4. Is `boss477 fawaz_ah` a teammate, and was the 3D dashboard / regulatory dossier a jointly-scoped deliverable, or an unprompted addition? The report treats it as real but separate work from the statistical audit — please confirm that framing is fair to both contributors.
5. Was the "GE scanner at IOP" fact checked against any real DICOM metadata at any point, even outside this repo? If so, where — it would be good to cite that source directly rather than relying on general IXI community knowledge.
6. Is the "regulatory dossier" meant to be understood as a mockup/demo of what a compliance report *could* look like, or was it intended to be presented as an actual completed conformity assessment? This materially changes how it should be described.
7. Should this report (and any demo) describe the project as the IXI-only statistical audit (Sudarsan's core, fully verified work), or as the combined package including the 3D dashboard? They read very differently to a technical reviewer.
