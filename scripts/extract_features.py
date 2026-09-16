"""Stage T1 - extract simple volumetric features from the T1s already used
for the DeepBrainNet audit, for the SAME 560-subject cohort (apples-to-apples
comparison in Stage T5's fairness audit).

Uses antspynet.deep_atropos for a six-tissue segmentation. Label mapping is
NOT hardcoded from memory - it's read from the function's own docstring and
confirmed against a real run before this script trusts it (see the printed
mapping below, and predict_one.py-style sanity checks).

Confirmed six-tissue labels (antspynet.utilities.deep_atropos docstring,
cross-checked against a live run on IXI090):
    0 = background   (excluded from all features)
    1 = CSF
    2 = gray matter
    3 = white matter
    4 = deep gray matter
    5 = brain stem
    6 = cerebellum

Features per subject (13):
    volume_CSF_ml, volume_GM_ml, volume_WM_ml, volume_deepGM_ml,
    volume_brainstem_ml, volume_cerebellum_ml,
    frac_CSF, frac_GM, frac_WM, frac_deepGM, frac_brainstem, frac_cerebellum,
    total_brain_volume_ml
(fractions are each tissue's volume / total_brain_volume_ml, i.e. of labels 1-6 summed)

Checkpointed to results/_features.jsonl after every subject; resumable.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import gc
import json
import time
import traceback
from pathlib import Path

import ants
import numpy as np
import pandas as pd
from antspynet.utilities import deep_atropos

RESULTS = Path(r"D:\Projects\iAuditAge\results")
CKPT_PATH = RESULTS / "_features.jsonl"
OUT_CSV = RESULTS / "features.csv"
DBN_CSV = RESULTS / "predictions_deepbrainnet.csv"

TISSUE_LABELS = {
    1: "CSF",
    2: "GM",
    3: "WM",
    4: "deepGM",
    5: "brainstem",
    6: "cerebellum",
}


def load_checkpoint():
    rows, failed = [], {}
    if not CKPT_PATH.exists():
        return rows, failed
    for line in CKPT_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_failed"):
            failed[rec["subject_id"]] = rec["_error"]
        else:
            rows.append(rec)
    return rows, failed


def append_checkpoint(rec):
    with open(CKPT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def extract_one(filepath):
    t1 = ants.image_read(filepath)
    result = deep_atropos(t1, do_preprocessing=True, verbose=False)
    seg = result["segmentation_image"]
    arr = seg.numpy()
    voxel_vol_ml = float(np.prod(seg.spacing)) / 1000.0

    labels, counts = np.unique(arr, return_counts=True)
    vol_by_label = {int(l): float(c) * voxel_vol_ml for l, c in zip(labels, counts)}

    feats = {}
    total = 0.0
    for lbl, name in TISSUE_LABELS.items():
        v = vol_by_label.get(lbl, 0.0)
        feats[f"volume_{name}_ml"] = v
        total += v
    feats["total_brain_volume_ml"] = total
    for lbl, name in TISSUE_LABELS.items():
        feats[f"frac_{name}"] = feats[f"volume_{name}_ml"] / total if total > 0 else np.nan

    # sanity: every voxel accounted for is either background or one of the six labels
    unexpected = set(int(l) for l in labels) - {0} - set(TISSUE_LABELS)
    if unexpected:
        raise RuntimeError(f"unexpected segmentation labels: {unexpected}")

    # deep_atropos returns 7 probability images (one per label) plus the
    # segmentation - these are large ANTs/ITK-backed images that Python's GC
    # doesn't always reclaim promptly. Free them explicitly; we saw
    # MemoryError on later subjects otherwise (RAM not released between scans).
    del result, t1, seg, arr
    gc.collect()

    return feats


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    dbn = pd.read_csv(DBN_CSV)
    cohort = dbn[["subject_id", "chronological_age", "sex", "site", "ethnicity"]].copy()
    # recover filepath the same way load_demographics does
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from load_demographics import build_cohort as _bc
    full, _ = _bc()
    path_map = dict(zip(full["subject_id"], full["filepath"]))
    cohort["filepath"] = cohort["subject_id"].map(path_map)
    missing_path = cohort["filepath"].isna().sum()
    if missing_path:
        print(f"WARNING: {missing_path} subjects from predictions CSV have no "
              f"matching T1 file on disk right now (skipping those)")
        cohort = cohort.dropna(subset=["filepath"])

    print(f"target cohort (from {DBN_CSV.name}): {len(cohort)} subjects")

    rows, failures = load_checkpoint()
    done = {r["subject_id"] for r in rows} | set(failures)
    if done:
        print(f"resuming: {len(rows)} done, {len(failures)} failed, "
              f"{len(cohort) - len(done)} remaining")

    for i, r in enumerate(cohort.itertuples(index=False), 1):
        if r.subject_id in done:
            continue
        t0 = time.time()
        try:
            feats = extract_one(r.filepath)
            rec = {
                "subject_id": r.subject_id,
                "chronological_age": float(r.chronological_age),
                "sex": r.sex,
                "site": r.site,
                "ethnicity": r.ethnicity,
                **feats,
            }
            rows.append(rec)
            append_checkpoint(rec)
            print(f"[{i}/{len(cohort)}] {r.subject_id}: "
                  f"total_brain={feats['total_brain_volume_ml']:.0f}mL  ({time.time()-t0:.0f}s)")
        except Exception as e:
            failures[r.subject_id] = repr(e)
            append_checkpoint({"subject_id": r.subject_id, "_failed": True, "_error": repr(e)})
            print(f"[{i}/{len(cohort)}] {r.subject_id}: FAILED {e!r}")
            traceback.print_exc()

    if not rows:
        raise SystemExit("no features extracted - nothing to write")

    df = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")

    # ---------------- VERIFY T1 ----------------
    print("\n" + "=" * 60)
    print("VERIFY T1")
    print("=" * 60)
    print(f"n subjects with features = {len(df)}  (target = {len(cohort)})")
    print(f"n failed = {len(failures)}")
    for sid, err in failures.items():
        print(f"    {sid}: {err}")

    feature_cols = [c for c in df.columns if c not in
                    ("subject_id", "chronological_age", "sex", "site", "ethnicity")]
    n_nan = df[feature_cols].isna().sum().sum()
    print(f"\nassert NO NaNs in features: {'PASS' if n_nan == 0 else f'FAIL ({n_nan} NaNs)'}")

    tbv = df["total_brain_volume_ml"]
    print(f"total_brain_volume_ml: min={tbv.min():.1f}  max={tbv.max():.1f}  "
          f"mean={tbv.mean():.1f}")
    plausible = (tbv.min() > 500) and (tbv.max() < 2500)
    print(f"plausible adult range (roughly 1000-1600 mL, allowing margin): "
          f"{'looks reasonable' if plausible else 'CHECK THIS - looks off'}")

    print(f"\nfeature table head:\n{df.head(5).to_string()}")


if __name__ == "__main__":
    main()
