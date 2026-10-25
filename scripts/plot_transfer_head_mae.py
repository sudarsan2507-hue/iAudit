"""Plot the transfer-learning MAE comparison (Colab run, colab/iaudit_colab_auto.ipynb).

Reads results/transfer_head_mae.csv (copied from the notebook's stage-3 output)
and writes results/fig_transfer_head_mae.png. Models whose pretraining data
overlaps the test cohorts are marked with a dagger so the sub-4y numbers are
never shown without that caveat.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"mae_guys_hh_5fold_oof": ("IXI Guys+HH (5-fold out-of-fold, n=492)", "#2a78d6"),
          "mae_enki": ("eNKI (not used to train the heads, n=39)", "#eb6834")}
TARGET = 4.0


def main():
    df = pd.read_csv(RESULTS / "transfer_head_mae.csv").iloc[::-1].reset_index(drop=True)
    labels = [m + ("  †" if "BrainAgeNeXt" in m else "") for m in df["model"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    h, gap = 0.36, 0.04
    for k, (col, (name, color)) in enumerate(SERIES.items()):
        ys = [i + (0.5 - k) * (h + gap) for i in range(len(df))]   # Guys+HH on top within each group
        ax.barh(ys, df[col], height=h, color=color, label=name, zorder=3)
        for y, v in zip(ys, df[col]):
            ax.text(v + 0.06, y, f"{v:.2f}", va="center", fontsize=8, color=INK2, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))

    ax.axvline(TARGET, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(TARGET + 0.05, -0.75, "target: MAE < 4 y", fontsize=8, color=INK2, va="center")
    ax.set_yticks(range(len(df)), labels, fontsize=9, color=INK)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(-1.0, len(df) - 0.45)
    ax.set_xlabel("Mean absolute error (years)", fontsize=9, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False,
              labelcolor=INK, handlelength=1.2, borderaxespad=0.2)
    fig.suptitle("Transfer-learning heads on frozen brain-age CNN features", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.text(0.01, 0.01, "† BrainAgeNeXt was pretrained on openBHB (includes IXI) and NKI-RS, so its "
             "sub-4 y errors are not a held-out result.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = RESULTS / "fig_transfer_head_mae.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
