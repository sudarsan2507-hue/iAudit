"""Check which axis orientation of the dbn_embed preprocessed brains matches
BrainAgeNeXt's expected FSL-MNI152 input, using the official model class and
transforms unchanged. Reports raw (uncorrected) predictions only.
"""
import os
import sys
import tempfile

E_TMP = r"E:\iAuditData\tmp"
os.environ["TEMP"] = os.environ["TMP"] = E_TMP
tempfile.tempdir = E_TMP

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BANG_DIR = Path(r"E:\iAuditData\brainagenext")
sys.path.insert(0, str(BANG_DIR))
from BrainAge_estimation import MedNeXtEncReg, prepare_transforms  # official, unchanged

EMB = Path(r"E:\iAuditData\dbn_embed")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main():
    manifest = pd.read_csv(EMB / "manifest.csv")
    ready = [r for r in manifest.itertuples() if (EMB / f"{r.scan_id}.npz").exists()][:N]
    print(f"testing {len(ready)} scans", flush=True)

    models = []
    for i in range(1, 6):
        m = MedNeXtEncReg()
        m.load_state_dict(torch.load(BANG_DIR / f"BrainAge_{i}.pth", map_location="cpu"))
        m.eval()
        models.append(m)

    tf = prepare_transforms()
    variants = {"as_is": None, "flip_LR": 1, "flip_AP": 2, "flip_SI": 3}
    rows = []
    for r in ready:
        x = tf({"image": str(EMB / f"{r.scan_id}_prep.nii.gz")})["image"]
        x = torch.as_tensor(np.asarray(x))[None]  # (1,1,160,192,160)
        rec = {"scan_id": r.scan_id, "age": r.chronological_age}
        for name, ax in variants.items():
            xi = x if ax is None else torch.flip(x, dims=[ax + 1])
            t0 = time.time()
            with torch.no_grad():
                preds = [float(m(xi)) for m in models]
            rec[name] = float(np.median(preds))
            rec[f"{name}_s"] = time.time() - t0
        rows.append(rec)
        print(r.scan_id, f"age={r.chronological_age:.1f}",
              " ".join(f"{k}={rec[k]:.1f}" for k in variants), f"({rec['as_is_s']:.0f}s/5 models)", flush=True)

    df = pd.DataFrame(rows)
    print("\nMAE (raw, uncorrected) per orientation:")
    for k in variants:
        print(f"  {k:8s} {np.abs(df[k] - df['age']).mean():.2f}y   r={np.corrcoef(df[k], df['age'])[0,1]:.2f}")


if __name__ == "__main__":
    main()
