"""BrainAgeNeXt vs DeepBrainNet: how much does each model's error grow from IXI to eNKI?

Slope chart from results/transfer_head_mae.csv (raw model rows only, no heads).
Writes results/fig_bang_vs_dbn.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {"DeepBrainNet raw": ("DeepBrainNet", "#2a78d6"),
          "BrainAgeNeXt raw": ("BrainAgeNeXt †", "#eb6834")}


def main():
    df = pd.read_csv(RESULTS / "transfer_head_mae.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [0, 1]
    for key, (name, color) in MODELS.items():
        ys = [df.loc[key, "mae_guys_hh_5fold_oof"], df.loc[key, "mae_enki"]]
        ax.plot(xs, ys, color=color, lw=2, zorder=3)
        ax.scatter(xs, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(-0.06, ys[0], f"{ys[0]:.2f}", ha="right", va="center", fontsize=9, color=INK2)
        ax.text(1.06, ys[1], f"{ys[1]:.2f}", ha="left", va="center", fontsize=9, color=INK2)
        ax.text(1.32, ys[1], f"{name}\n+{ys[1] - ys[0]:.2f} y", ha="left", va="center",
                fontsize=9, color=INK, linespacing=1.3)
    ax.axhline(4.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.45, 4.05, "4 y target", fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(xs, ["IXI Guys+HH\n(n=492)", "eNKI\n(n=39)"], fontsize=9, color=INK)
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(2.5, 7.0)
    ax.set_ylabel("Mean absolute error (years)", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("Error growth from IXI to a second cohort (eNKI)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.text(0.02, 0.015, "† BrainAgeNeXt's training data includes IXI (via openBHB) and NKI-RS,\n"
             "  so its small increase may reflect memory rather than robustness.",
             fontsize=7.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    out = RESULTS / "fig_bang_vs_dbn.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
