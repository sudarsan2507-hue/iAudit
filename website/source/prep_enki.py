"""Pick eNKI demo cases (one per age decade + each CNN's worst miss) and preprocess their raw T1s
with the exact antspynet pipeline used for the models (scripts/dbn_embed.py). Images are for display;
predictions come from the Colab run. Output: E:\\iAuditData\\demo\\enki_prep\\<scan_id>_prep.nii.gz
"""
import os
import sys
import tempfile
import time

E_TMP = r"E:\iAuditData\tmp"
os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = E_TMP
tempfile.tempdir = E_TMP
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_from_project import pick  # same case-selection rule as IXI/SALD

OUT = Path(r"E:\iAuditData\demo\enki_prep")
RAW = Path(r"E:\iAuditData\enki")


def chosen():
    t = pd.read_csv(r"D:\Projects\iAuditAge\results\transfer_head_predictions.csv")
    e = t[t.site == "eNKI"].reset_index(drop=True)
    e["e_dbn"], e["e_bang"] = (e.dbn_raw - e.age).abs(), (e.bang_raw - e.age).abs()
    return pick(e, "e_bang", [("e_dbn", "DeepBrainNet's largest miss"), ("e_bang", "BrainAgeNeXt's largest miss")])


def main():
    OUT.mkdir(exist_ok=True)
    picks = chosen()
    print(f"{len(picks)} eNKI cases: {picks}", flush=True)
    if "--list" in sys.argv:
        return
    import ants
    from antspynet.utilities import preprocess_brain_image, set_antsxnet_cache_directory
    set_antsxnet_cache_directory(r"E:\iAuditData\keras_cache\ANTsXNet")
    for i, sid in enumerate(picks, 1):
        dst = OUT / f"{sid}_prep.nii.gz"
        if dst.exists():
            continue
        t0 = time.time()
        raw = RAW / f"sub-{sid.replace('eNKI_', '')}_T1w.nii.gz"
        prep = preprocess_brain_image(ants.image_read(str(raw)), truncate_intensity=(0.01, 0.99),
                                      brain_extraction_modality="t1", template="croppedMni152",
                                      template_transform_type="antsRegistrationSyNQuickRepro[a]",
                                      do_bias_correction=True, do_denoising=True, verbose=False)
        img = prep["preprocessed_image"] * prep["brain_mask"]
        img = (img - img.min()) / (img.max() - img.min())
        ants.image_write(img, str(dst))
        print(f"[{i}/{len(picks)}] {sid} done ({time.time() - t0:.0f}s)", flush=True)
        for f in Path(E_TMP).glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
