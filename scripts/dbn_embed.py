"""Stage F1 - DeepBrainNet feature extraction for transfer learning.

Replicates antspynet.utilities.brain_age preprocessing exactly, then saves for
every scan: the preprocessed brain volume (so later experiments skip the slow
registration), DeepBrainNet's per-slice 512-d GAP features, per-slice 1024-d
dense_3 features, and per-slice age predictions.

Order: IXI Guys+HH (training pool) -> IXI IOP (held-out site) -> eNKI (external).
Everything is written to E:. Checkpointed per scan; rerun to resume.
"""
import os
import sys
import tempfile
from pathlib import Path

E_TMP = r"E:\iAuditData\tmp"
os.makedirs(E_TMP, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = E_TMP
tempfile.tempdir = E_TMP
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import statistics
import traceback

import ants
import numpy as np
import pandas as pd
import tf_keras as keras
from antspynet.utilities import preprocess_brain_image, set_antsxnet_cache_directory

set_antsxnet_cache_directory(r"E:\iAuditData\keras_cache\ANTsXNet")

sys.path.insert(0, str(Path(__file__).parent))
from load_demographics import build_cohort

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(r"E:\iAuditData\dbn_embed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = r"E:\iAuditData\keras_cache\ANTsXNet\brainAgeDeepBrainNet.h5"
WHICH_SLICES = list(range(45, 125))  # identical to antspynet brain_age
SITE_ORDER = {"Guys": 0, "HH": 0, "IOP": 1}


def build_manifest():
    cohort, _ = build_cohort()
    dbn = pd.read_csv(RESULTS / "predictions_deepbrainnet.csv")
    ixi = dbn.merge(cohort[["subject_id", "filepath"]], on="subject_id", how="inner")
    ixi = ixi.assign(dataset="IXI", scan_id=ixi["subject_id"],
                     original_pred=ixi["predicted_age"])
    ixi = ixi[["scan_id", "dataset", "site", "chronological_age", "sex", "filepath", "original_pred"]]

    enki = pd.read_csv(RESULTS / "enki_features_predictions.csv")
    enki = pd.DataFrame({
        "scan_id": "eNKI_" + enki["participant_id"],
        "dataset": "eNKI",
        "site": "eNKI",
        "chronological_age": enki["chronological_age"],
        "sex": enki["sex"].str[0],
        "filepath": [rf"E:\iAuditData\enki\sub-{p}_T1w.nii.gz" for p in enki["participant_id"]],
        "original_pred": enki["predicted_age"],
    })

    m = pd.concat([ixi, enki], ignore_index=True)
    m["order"] = m["site"].map(SITE_ORDER).fillna(2)
    m = m.sort_values(["order", "scan_id"]).drop(columns="order").reset_index(drop=True)
    missing = [f for f in m["filepath"] if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} scan files missing, e.g. {missing[:3]}")
    return m


def main():
    manifest = build_manifest()
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    counts = manifest.groupby(["dataset", "site"]).size().to_dict()
    print(f"target: {len(manifest)} scans  {counts}", flush=True)

    model = keras.models.load_model(MODEL_PATH)
    gap_layer = model.get_layer("global_average_pooling2d_2")
    dense_layer = model.get_layer("dense_3")
    feat_model = keras.Model(model.input, [gap_layer.output, dense_layer.output, model.output])

    done = sum((OUT_DIR / f"{s}.npz").exists() for s in manifest["scan_id"])
    print(f"already done: {done}", flush=True)

    for i, row in enumerate(manifest.itertuples(), 1):
        npz_path = OUT_DIR / f"{row.scan_id}.npz"
        if npz_path.exists():
            continue
        t0 = time.time()
        try:
            t1 = ants.image_read(row.filepath)
            prep = preprocess_brain_image(
                t1,
                truncate_intensity=(0.01, 0.99),
                brain_extraction_modality="t1",
                template="croppedMni152",
                template_transform_type="antsRegistrationSyNQuickRepro[a]",
                do_bias_correction=True,
                do_denoising=True,
                verbose=False)
            img = prep["preprocessed_image"] * prep["brain_mask"]
            img = (img - img.min()) / (img.max() - img.min())

            prep_path = OUT_DIR / f"{row.scan_id}_prep.nii.gz"
            ants.image_write(img, str(prep_path))

            batch = np.zeros((len(WHICH_SLICES), *img.shape[0:2], 3), dtype=np.float32)
            for j, idx in enumerate(WHICH_SLICES):
                s = ants.slice_image(img, axis=2, idx=idx).numpy()
                batch[j, :, :, 0] = batch[j, :, :, 1] = batch[j, :, :, 2] = s
            gap, dense, slice_pred = feat_model.predict(batch, verbose=0)
            pred = float(statistics.median(slice_pred[:, 0]))

            tmp_path = OUT_DIR / f"{row.scan_id}.tmp.npz"
            np.savez_compressed(tmp_path, gap=gap.astype(np.float16),
                                dense=dense.astype(np.float16),
                                slice_pred=slice_pred[:, 0].astype(np.float32),
                                predicted_age=np.float32(pred))
            os.replace(tmp_path, npz_path)
            print(f"[{i}/{len(manifest)}] {row.scan_id} {row.site}: age={row.chronological_age:.1f} "
                  f"pred={pred:.1f} (orig {row.original_pred:.1f})  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(manifest)}] {row.scan_id}: FAILED {e!r}", flush=True)
            traceback.print_exc()
        finally:
            for f in Path(E_TMP).glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    done = sum((OUT_DIR / f"{s}.npz").exists() for s in manifest["scan_id"])
    print(f"\nfinished: {done}/{len(manifest)} scans have features", flush=True)


if __name__ == "__main__":
    main()
