"""BrainAgeNeXt fairness audit - the Stage 5 method (fairness_audit.py) applied to
the Colab transfer-learning run.

Input: transfer_head_predictions.csv from the Colab notebook's outputs/ folder
(per-scan dbn_raw, bang_raw and 5-fold out-of-fold head predictions). Sex and
ethnicity are joined from results/predictions_deepbrainnet.csv.

Scope: IXI Guys+HH only. IOP was not part of the Colab run, and eNKI has no
ethnicity field and is a different population. All three models are audited on
the identical subject set so they compare like for like:
  DeepBrainNet (raw), BrainAgeNeXt (raw), BrainAgeNeXt 512 + ridge (out-of-fold).

The age-bias correction, subgroup summaries and statistical tests are imported
from fairness_audit.py unchanged; this script does not touch that audit's outputs.

Usage: python scripts/bang_fairness_audit.py [path/to/transfer_head_predictions.csv]
Writes results/bias_report_brainagenext.csv and results/fig_bang_fairness.png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fairness_audit import (MIN_GROUP_N, RESULTS, add_gap_and_correction, axis_stats,
                            intersectional_summary, subgroup_summary)

DEFAULT_IN = Path(r"E:\iAuditData\colab_outputs\transfer_head_predictions.csv")
MODELS = {"dbn_raw": "DeepBrainNet", "bang_raw": "BrainAgeNeXt", "pred BANG 512": "BrainAgeNeXt+ridge"}
COLORS = {"DeepBrainNet": "#2a78d6", "BrainAgeNeXt": "#eb6834", "BrainAgeNeXt+ridge": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
AXES = ["sex", "site", "ethnicity"]


def load(path):
    tp = pd.read_csv(path)
    missing = [c for c in [*MODELS, "scan_id", "site", "age"] if c not in tp.columns]
    if missing:
        raise SystemExit(f"{path} lacks columns {missing}; has {tp.columns.tolist()}")
    tp = tp[tp["site"].isin(["Guys", "HH"])]
    demo = pd.read_csv(RESULTS / "predictions_deepbrainnet.csv")[["subject_id", "sex", "ethnicity"]]
    tp = tp.merge(demo, left_on="scan_id", right_on="subject_id", how="inner")
    if tp[list(MODELS)].isna().any().any():
        raise SystemExit("missing predictions for some Guys/HH scans - is the Colab run complete?")
    frames = [pd.DataFrame({"subject_id": tp["subject_id"], "chronological_age": tp["age"],
                            "predicted_age": tp[col], "sex": tp["sex"], "site": tp["site"],
                            "ethnicity": tp["ethnicity"], "model_name": name})
              for col, name in MODELS.items()]
    return pd.concat(frames, ignore_index=True)


def audit(df):
    rows = []
    for model in MODELS.values():
        sub = df[df["model_name"] == model]
        print(f"\n=== {model}  (n={len(sub)})  raw MAE {np.abs(sub['gap']).mean():.2f}y")
        for axis in AXES:
            summ = subgroup_summary(sub, axis, model)
            st = axis_stats(sub, axis, model)
            print(f"[{axis}] {st['omnibus_test'] or 'not testable'}"
                  + (f"  p={st['omnibus_p']:.4g}" if st["omnibus_p"] is not None else ""))
            print(summ[["group", "n", "MAE", "mean_corrected_gap", "underpowered"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            for g1, g2, p, md in st["pairwise"]:
                print(f"    {g1} vs {g2}: p={p:.4g}  median diff={md:+.2f}y")
            for r in summ.itertuples():
                rows.append({"model_name": model, "axis": axis, "group": r.group, "n": r.n,
                             "MAE_corrected": r.MAE, "mean_corrected_gap": r.mean_corrected_gap,
                             "underpowered": r.underpowered, "omnibus_test": st["omnibus_test"],
                             "omnibus_p": st["omnibus_p"]})
        inter = intersectional_summary(sub, model)
        print("[sex x site]")
        print(inter[["sex", "site", "n", "MAE", "mean_corrected_gap"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return pd.DataFrame(rows)


def plot(rep):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, axis in zip(axes, ["sex", "site"]):
        ax.set_facecolor(SURFACE)
        part = rep[(rep["axis"] == axis) & (rep["n"] >= MIN_GROUP_N)]
        groups = sorted(part["group"].unique())
        w = 0.26
        for i, model in enumerate(MODELS.values()):
            vals = [part[(part["model_name"] == model) & (part["group"] == g)]["mean_corrected_gap"].item()
                    for g in groups]
            xs = np.arange(len(groups)) + (i - 1) * (w + 0.02)
            ax.bar(xs, vals, w, color=COLORS[model], label=model, zorder=3)
            for x, v in zip(xs, vals):   # value labels: the aqua bar is below 3:1 contrast on the surface
                ax.text(x, v + (0.04 if v >= 0 else -0.04), f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7, color=INK2, zorder=4)
        ax.axhline(0, color=INK2, lw=0.8, zorder=2)
        ax.set_xticks(range(len(groups)), groups, color=INK)
        ax.set_title(f"by {axis}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("mean age-corrected gap (years)", fontsize=9, color=INK2)
    fig.suptitle("Subgroup bias after age correction - IXI Guys+HH, same 492 subjects for every model",
                 fontsize=10.5, color=INK, x=0.01, ha="left", y=0.98)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, ncol=3,
               loc="upper left", bbox_to_anchor=(0.01, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out = RESULTS / "fig_bang_fairness.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    if not path.exists():
        raise SystemExit(f"{path} not found - download outputs/transfer_head_predictions.csv from Google Drive")
    df = add_gap_and_correction(load(path))
    rep = audit(df)
    out = RESULTS / "bias_report_brainagenext.csv"
    rep.to_csv(out, index=False)
    print("\nwrote", out)
    plot(rep)


if __name__ == "__main__":
    main()
