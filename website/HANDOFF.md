# iAudit Case Viewer - handoff

This folder is everything needed to run, rebuild and extend the iAudit brain-age demo website.
Every number on the site comes from the project's real runs in `../results/`; nothing is invented,
and `source/verify_data.py` proves it.

## 1. What the project is (two-minute version)

**Brain age** is a model's estimate of how old a brain looks on an MRI. The gap between predicted
and real age is used as a health marker, so the models must be accurate and fair across sexes,
scanners and populations. iAudit audits that.

| Cohort | Where | Scans | Role |
|---|---|---|---|
| IXI | London, 3 hospitals (Guys, HH, IOP) | 560 | Main audit; Guys+HH train the ridge heads; IOP is the scanner every model fails on |
| eNKI | NKI-Rockland, New York | 39 | Second cohort |
| SALD | Southwest University, China | 200 (age-balanced) | **Held-out test**: no model or head ever saw it |

| Model variant | What it is |
|---|---|
| DeepBrainNet | Published 2D CNN (VGG16-style), 80 axial slices, median of slice predictions |
| BrainAgeNeXt | Published 3D CNN (MedNeXt), ensemble of 5 models, median prediction |
| `... + ridge` heads | CNN frozen; its internal features (512 or 1024 numbers per scan) feed a ridge regression trained on IXI Guys+HH only |
| Linear regression / Random forest / XGBoost | Tabular models on 7 tissue-volume features, IXI only |

**Headline results** (MAE = mean absolute error in years, 95% bootstrap CI):

| | IXI Guys+HH (n=492) | eNKI (n=39) | SALD, held out (n=200) |
|---|---|---|---|
| DeepBrainNet | 4.57 (4.25-4.89) | 6.30 (4.79-7.93) | 5.77 (5.17-6.39) |
| BrainAgeNeXt | 3.38 (3.17-3.61) | 3.69 (2.92-4.55) | **3.39 (3.03-3.76)** |

Caveat that must stay visible on any version of the site: BrainAgeNeXt was trained on openBHB
(which contains IXI) and NKI-RS (the source of eNKI). Only SALD is a clean test. Full write-up:
`../README.md` and `../PROJECT_REPORT.md`.

## 2. What is in this folder

```
website/
  HANDOFF.md                this file
  site/                     the deployable website (static files, no build step)
    index.html              page + all case data inlined
    img/*.png               234 MRI slice images (26 cases x 9 views)
    vercel.json             cache headers for the images
  data/
    cases.json              the 26 demo cases: ages, sex, cohort, every prediction, image paths
    all_predictions.csv     EVERY scan (799) x EVERY model - build anything you like from this
    metrics.json            cohort-level results (MAE + CIs, fairness audits, reliability, uncertainty)
  source/
    template.html           page design + code - edit this, then rebuild
    build_site.py           template + data/cases.json -> site/index.html (stdlib Python only)
    export_all_values.py    results/ -> data/all_predictions.csv + metrics.json (pandas)
    verify_data.py          checks every site number against the results files (pandas)
    export_from_project.py  how cases.json + images were made (needs the MRI volumes - see section 6)
    prep_enki.py            preprocessing used for one eNKI display image (see section 6)
```

## 3. Run, rebuild, deploy

- **Look at it:** open `site/index.html` in a browser, or `npx serve website/site`.
- **Change the design or code:** edit `source/template.html`, then `python website/source/build_site.py`.
- **Prove the numbers:** `python website/source/verify_data.py` (prints "all match"; exits non-zero otherwise).
- **Deploy on Vercel:** Add New > Project > import this GitHub repo > *Root Directory* `website/site` >
  Framework Preset *Other* > Deploy. Or `npx vercel --prod` inside `website/site`.

## 4. Data reference

### data/all_predictions.csv (one row per scan)

| Column | Meaning |
|---|---|
| `scan_id` | `IXI002`, `eNKI_A00028552`, `SALD_031274` |
| `cohort`, `site` | IXI sites: Guys, HH, IOP; else same as cohort |
| `age`, `sex` | true age in years (SALD: whole years), F/M |
| `ethnicity` | IXI self-report; eNKI not recorded; SALD cohort-level |
| `in_brainagenext_training_data` | yes (openBHB) / yes (NKI-RS) / no (held out) |
| `deepbrainnet` | DeepBrainNet, Colab run (IXI Guys+HH, eNKI, SALD) |
| `dbn_original_run` | DeepBrainNet, original laptop run - the only DeepBrainNet values for **IOP** |
| `brainagenext`, `brainagenext_model1..5`, `brainagenext_sd` | ensemble median, each member, members' SD |
| `dbn_gap512_ridge`, `dbn_dense1024_ridge`, `brainagenext_ridge`, `dbn_gap_plus_brainagenext_ridge` | ridge heads; IXI values are 5-fold out-of-fold; eNKI/SALD from heads trained on all IXI Guys+HH; no gap512 head on SALD |
| `linear_regression`, `random_forest`, `xgboost` | tabular models, IXI only (5-fold out-of-fold) |
| `reliability_D` | distance of the scan's tissue volumes from the IXI Guys+HH reference (IXI, eNKI) |

Empty cell = that model was not run on that scan (e.g. BrainAgeNeXt was never run on IOP).
Error = prediction - age. MAE = mean of |error|.

### data/cases.json (what the page renders)

`cases[]`: `id, cohort, site, age, sex, ethnicity, why` (why it was picked), `note` (caveat shown under
the scan), `preds` (model name -> years; `"BrainAgeNeXt models"` is a list of 5), `views`
(`axial`: 7 image paths inferior->superior, `coronal`, `sagittal`). Plus `summary` (cohort MAE + CI),
`heads_ixi`, `heads_sald` for the tables.

### Images

Each case's MRI is the **preprocessed** volume the models received: N4 bias correction, denoising,
skull stripping and affine registration to the cropped MNI152 template (ANTsPyNet), 182x218x182 at
1 mm, intensities scaled 0-1. Shown in neurological view (patient's left on the left). Axial slices
are z = 40, 55, 70, 85, 100, 115, 130; coronal y = 119; sagittal x = 73.

## 5. How the page works

- One self-contained page: data inlined as JSON in `<script id="demo-data">`, plain JavaScript, no framework.
- **Chart**: one row per model on a shared age axis; the black vertical line is the true age; markers are
  colour = model family (DeepBrainNet blue, BrainAgeNeXt orange, combined head green, tabular violet),
  shape = type (circle CNN, diamond ridge head, square tabular). BrainAgeNeXt's row also shows its 5
  members as ticks and their range. Errors of 5 years or more are flagged.
- **Quiz mode**: hides ages and predictions until the viewer guesses.
- **Light and dark themes** from CSS tokens at the top of the `<style>`; the scan viewer is always black.
- **Lesson learned:** inside sandboxed frames `history.replaceState` can throw. The page redraws first and
  updates the URL hash in a `try` block. Keep that order or case switching silently stops working.

## 6. What is not here, and why

- **Raw and preprocessed MRI volumes** (several GB; IXI is CC BY-SA 3.0, eNKI/SALD are under INDI terms).
  Only the rendered PNG slices of the 26 demo cases are included. `export_from_project.py` documents how
  they were produced but cannot run without the volumes.
- **Live prediction on an uploaded scan.** Not possible in a browser: each scan needs ANTs registration
  (~2-3 min CPU), TensorFlow (DeepBrainNet) and PyTorch (BrainAgeNeXt). A "deeper" site that does this
  needs a GPU backend (e.g. a FastAPI service on a GPU VM, or a Hugging Face Space) running the same
  pipeline as `../colab/iaudit_colab_auto.ipynb`. Present any such result as a new run, never mixed with
  the recorded results.
- **Images of one eNKI case** (`eNKI_A00054895`) were re-made on a laptop with the identical pipeline;
  the other 25 come from the recorded runs. All predictions are from the recorded runs.

## 7. Ideas for a deeper version (all doable with the data here)

1. Explore all 799 scans: scatter of predicted vs true age per model and cohort, filter by sex/site/age.
2. Fairness view from `metrics.json`: age-corrected bias by sex and site, per model.
3. The IOP story: every model fails on that scanner (`dbn_original_run` and tabular columns for IOP).
4. Uncertainty: BrainAgeNeXt's 5-member spread (`brainagenext_sd`) against its error.
5. Keep the training-data caveat next to every IXI/eNKI BrainAgeNeXt number.

## 8. Credits and licences

IXI dataset (brain-development.org, CC BY-SA 3.0); NKI-Rockland Sample and SALD via INDI/FCP;
DeepBrainNet via ANTsPyNet; BrainAgeNeXt (La Rosa et al., Imaging Neuroscience 2025). Cite these in any
public version.
