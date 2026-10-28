"""Collect real cases for the brain-age demo page: preprocessed MRI slices + every model's prediction.

Everything is taken from the project's actual outputs (no new inference):
  IXI : results/transfer_head_predictions.csv (Colab), results/brainagenext_predictions.csv (5 models),
        results/predictions_{linearregression,randomforest,xgboost}.csv (tabular), results/features.csv (distance D)
        images from E:\\iAuditData\\dbn_embed\\<id>_prep.nii.gz
  SALD: results/sald_predictions.csv; images from the sald_outputs Drive zip (dbn_embed/<id>_prep.nii.gz)
Writes E:\\iAuditData\\demo\\demo_data.json (images inlined, for the Claude Artifact page) and the website copy:
website/data/cases.json + website/site/img/*.png. Needs the preprocessed MRI volumes, which are not in the repo;
the website can be rebuilt without them from cases.json (website/source/build_site.py).
"""
import base64
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pandas as pd

R = Path(r"D:\Projects\iAuditAge\results")
OUT = Path(r"E:\iAuditData\demo")
IXI_PREP = Path(r"E:\iAuditData\dbn_embed")
SALD_ZIP = Path(r"E:\iAuditData\sald_outputs-20261026T200418Z-1-001.zip")
WEBSITE = Path(r"D:\Projects\iAuditAge\website")
ENKI_LAPTOP_IMAGES = {"eNKI_A00054895"}   # the other 7 eNKI images were downloaded from the Colab run's Drive outputs
AXIAL = [40, 55, 70, 85, 100, 115, 130]          # z indices in the 182x218x182 MNI volume
BINS = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]


def png(a):
    lo, hi = 0.0, np.percentile(a[a > 0], 99.5) if (a > 0).any() else 1.0
    buf = io.BytesIO()
    mpimg.imsave(buf, np.clip((a - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def views(vol):
    vol = np.asarray(vol, dtype=np.float32)
    x, y, z = (s // 2 for s in vol.shape)
    return {"axial": [png(np.rot90(vol[:, :, k])) for k in AXIAL],
            "coronal": png(np.rot90(vol[:, y + 10, :])),
            "sagittal": png(np.rot90(vol[x - 18, :, :]))}


def pick(df, err_col, n_extra_tag):
    """one representative scan per age decade (error closest to the decade median) + worst cases."""
    chosen = {}
    for lo, hi in BINS:
        b = df[(df.age >= lo) & (df.age < hi)]
        if len(b):
            i = (b[err_col] - b[err_col].median()).abs().idxmin()
            chosen[b.loc[i, "scan_id"]] = f"Typical {lo}s"
    for col, tag in n_extra_tag:
        i = df[col].idxmax()
        chosen.setdefault(df.loc[i, "scan_id"], tag)
    return chosen


def ixi_cases():
    t = pd.read_csv(R / "transfer_head_predictions.csv")
    b = pd.read_csv(R / "brainagenext_predictions.csv").set_index("scan_id")
    tab = {m: pd.read_csv(R / f"predictions_{m.lower()}.csv").set_index("subject_id")["predicted_age"]
           for m in ["LinearRegression", "RandomForest", "XGBoost"]}
    df = t[t.site.isin(["Guys", "HH"])].copy()
    df = df[[(IXI_PREP / f"{s}_prep.nii.gz").exists() and s in tab["XGBoost"].index for s in df.scan_id]]
    df["e_dbn"], df["e_bang"] = (df.dbn_raw - df.age).abs(), (df.bang_raw - df.age).abs()
    chosen = pick(df.reset_index(drop=True), "e_bang",
                  [("e_dbn", "DeepBrainNet's largest miss"), ("e_bang", "BrainAgeNeXt's largest miss")])
    out = []
    for sid, why in chosen.items():
        r = df[df.scan_id == sid].iloc[0]
        bb = b.loc[sid]
        out.append({
            "id": sid, "cohort": "IXI", "site": r.site, "age": round(float(r.age), 1), "sex": bb.sex,
            "ethnicity": bb.ethnicity, "why": why,
            "note": "IXI is in BrainAgeNeXt's training data (via openBHB). Heads: 5-fold out-of-fold.",
            "preds": {
                "DeepBrainNet": float(r.dbn_raw), "BrainAgeNeXt": float(r.bang_raw),
                "BrainAgeNeXt models": [float(bb[f"pred_model{i}"]) for i in range(1, 6)],
                "DBN GAP-512 + ridge": float(r["pred DBN gap512"]),
                "DBN dense-1024 + ridge": float(r["pred DBN dense1024"]),
                "BrainAgeNeXt + ridge": float(r["pred BANG 512"]),
                "DBN GAP + BrainAgeNeXt + ridge": float(r["pred DBN gap + BANG"]),
                "Linear regression": float(tab["LinearRegression"][sid]),
                "Random forest": float(tab["RandomForest"][sid]), "XGBoost": float(tab["XGBoost"][sid])},
            "_img": IXI_PREP / f"{sid}_prep.nii.gz"})
    return out


def enki_cases():
    t = pd.read_csv(R / "transfer_head_predictions.csv")
    b = pd.read_csv(R / "brainagenext_predictions.csv").set_index("scan_id")
    e = t[t.site == "eNKI"].reset_index(drop=True)
    e["e_dbn"], e["e_bang"] = (e.dbn_raw - e.age).abs(), (e.bang_raw - e.age).abs()
    chosen = pick(e, "e_bang", [("e_dbn", "DeepBrainNet's largest miss"), ("e_bang", "BrainAgeNeXt's largest miss")])
    out = []
    for sid, why in chosen.items():
        r = e[e.scan_id == sid].iloc[0]
        bb = b.loc[sid]
        out.append({
            "id": sid, "cohort": "eNKI", "site": "eNKI", "age": round(float(r.age), 1), "sex": bb.sex,
            "ethnicity": "not recorded", "why": why,
            "note": ("eNKI (NKI-Rockland, New York) comes from NKI-RS, which is in BrainAgeNeXt's training data. "
                     "Heads were trained on IXI only. "
                     + ("This image was re-processed on the laptop with the identical pipeline for display; "
                        "the predictions are from the Colab run." if sid in ENKI_LAPTOP_IMAGES else
                        "Image and predictions are both from the Colab run.")),
            "preds": {
                "DeepBrainNet": float(r.dbn_raw), "BrainAgeNeXt": float(r.bang_raw),
                "BrainAgeNeXt models": [float(bb[f"pred_model{i}"]) for i in range(1, 6)],
                "DBN GAP-512 + ridge": float(r["pred DBN gap512"]),
                "DBN dense-1024 + ridge": float(r["pred DBN dense1024"]),
                "BrainAgeNeXt + ridge": float(r["pred BANG 512"]),
                "DBN GAP + BrainAgeNeXt + ridge": float(r["pred DBN gap + BANG"])},
            "_img": Path(r"E:\iAuditData\demo\enki_prep") / f"{sid}_prep.nii.gz"})
    return out


def sald_cases():
    s = pd.read_csv(R / "sald_predictions.csv")
    s["e_dbn"], s["e_bang"] = (s.dbn_raw - s.age).abs(), (s.bang_raw - s.age).abs()
    chosen = pick(s, "e_bang", [("e_dbn", "DeepBrainNet's largest miss"), ("e_bang", "BrainAgeNeXt's largest miss")])
    out = []
    for sid, why in chosen.items():
        r = s[s.scan_id == sid].iloc[0]
        out.append({
            "id": sid, "cohort": "SALD", "site": "SALD", "age": float(r.age), "sex": r.sex, "ethnicity": "Chinese",
            "why": why, "note": "SALD was never seen by any model or head. Ages are whole years.",
            "preds": {
                "DeepBrainNet": float(r.dbn_raw), "BrainAgeNeXt": float(r.bang_raw),
                "BrainAgeNeXt models": [float(r[f"bang_model{i}"]) for i in range(1, 6)],
                "DBN dense-1024 + ridge": float(r["pred DBN dense1024 + ridge"]),
                "BrainAgeNeXt + ridge": float(r["pred BANG 512 + ridge"]),
                "DBN GAP + BrainAgeNeXt + ridge": float(r["pred DBN gap + BANG + ridge"])},
            "_img": f"sald_outputs/dbn_embed/{sid}_prep.nii.gz"})
    return out


def main():
    OUT.mkdir(exist_ok=True)
    cases = ixi_cases() + enki_cases() + sald_cases()
    zf = zipfile.ZipFile(SALD_ZIP)
    for c in cases:
        p = c.pop("_img")
        if isinstance(p, Path):
            img = nib.load(str(p))
        else:
            tmp = OUT / "tmp_prep.nii.gz"
            tmp.write_bytes(zf.read(p))
            img = nib.load(str(tmp))
        img = nib.as_closest_canonical(img)
        c["views"] = views(img.get_fdata())
        c["preds"] = {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v])
                      for k, v in c["preds"].items()}
        print(f"{c['cohort']:4s} {c['id']:12s} age {c['age']:5.1f}  {c['why']:30s} "
              f"DBN {c['preds']['DeepBrainNet']:5.1f}  BANG {c['preds']['BrainAgeNeXt']:5.1f}")
    (OUT / "tmp_prep.nii.gz").unlink(missing_ok=True)

    summ = pd.read_csv(R / "sald_summary.csv")
    heads = pd.read_csv(R / "transfer_head_mae.csv")
    sald_mae = pd.read_csv(R / "sald_mae.csv")
    data = {"cases": cases,
            "summary": summ[["cohort", "model", "n", "MAE", "ci_low", "ci_high", "mean_error"]].to_dict("records"),
            "heads_ixi": heads[["model", "mae_guys_hh_5fold_oof", "mae_enki"]].to_dict("records"),
            "heads_sald": sald_mae.to_dict("records")}
    (OUT / "demo_data.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"\n{len(cases)} cases, demo_data.json = {(OUT / 'demo_data.json').stat().st_size / 1e6:.2f} MB")
    to_website(data)


def to_website(data):
    """Website copy: images as files in website/site/img/, data in website/data/cases.json (paths, not data URIs)."""
    site_img, data_dir = WEBSITE / "site" / "img", WEBSITE / "data"
    for p in site_img.glob("*.png"):
        p.unlink()
    site_img.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(json.dumps(data))

    def save(uri, name):
        (site_img / name).write_bytes(base64.b64decode(uri.split(",", 1)[1]))
        return f"img/{name}"

    for c in data["cases"]:
        v = c["views"]
        v["axial"] = [save(u, f"{c['id']}_axial{i}.png") for i, u in enumerate(v["axial"])]
        v["coronal"] = save(v["coronal"], f"{c['id']}_coronal.png")
        v["sagittal"] = save(v["sagittal"], f"{c['id']}_sagittal.png")
    (data_dir / "cases.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"website: {len(list(site_img.glob('*.png')))} images, cases.json "
          f"{(data_dir / 'cases.json').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
