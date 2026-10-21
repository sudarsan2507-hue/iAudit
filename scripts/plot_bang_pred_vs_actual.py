"""Predicted vs actual age, DeepBrainNet next to BrainAgeNeXt, from the Colab export.

Input: transfer_head_predictions.csv (default path below). Writes results/fig_bang_pred_vs_actual.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_IN = RESULTS / "transfer_head_predictions.csv"   # copied from the Colab export
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COHORTS = {"IXI Guys+HH": ("#2a78d6", ["Guys", "HH"]), "eNKI": ("#eb6834", ["eNKI"])}
PANELS = {"dbn_raw": "DeepBrainNet", "bang_raw": "BrainAgeNeXt \u2020"}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    lim = (15, 95)
    for ax, (col, title) in zip(axes, PANELS.items()):
        ax.set_facecolor(SURFACE)
        ax.plot(lim, lim, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        stats_txt = []
        for name, (color, sites) in COHORTS.items():
            sub = df[df["site"].isin(sites)]
            ax.scatter(sub["age"], sub[col], s=14, color=color, alpha=0.55, linewidths=0,
                       label=f"{name} (n={len(sub)})", zorder=3)
            mae = np.abs(sub[col] - sub["age"]).mean()
            r = np.corrcoef(sub["age"], sub[col])[0, 1]
            stats_txt.append(f"{name}: MAE {mae:.2f} y, r = {r:.3f}")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.text(0.03, 0.97, "\n".join(stats_txt), transform=ax.transAxes, va="top", fontsize=8,
                color=INK2, linespacing=1.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Chronological age (years)", fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Predicted age (years)", fontsize=9, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, ncol=2,
               loc="upper left", bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("Predicted vs actual age (dashed = perfect prediction)", fontsize=11, color=INK,
                 x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.01, "\u2020 BrainAgeNeXt's training data includes IXI (via openBHB) and NKI-RS.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    out = RESULTS / "fig_bang_pred_vs_actual.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
