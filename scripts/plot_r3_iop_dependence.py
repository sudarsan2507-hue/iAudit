"""Show that the pooled R3 error-vs-distance effect comes from the IOP scans.

Reads results/pooled_error_vs_D.csv (written by stage_r3_pooled.py) and draws the same
scatter twice: all 583 scans, and with the 66 IOP scans removed. Each panel reports the
age-controlled partial Spearman r, computed exactly as stage_r3_pooled.py does.
Writes results/fig_r3_iop_dependence.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = {"IXI-reference (Guys+HH)": ("IXI Guys+HH", "#2a78d6"),
          "eNKI": ("eNKI", "#eb6834"),
          "IXI-IOP": ("IXI IOP", "#1baf7a")}


def partial_spearman(s):
    a, d, e = s["chronological_age"].to_numpy(), s["D"].to_numpy(), s["abs_error"].to_numpy()
    rd = d - np.polyval(np.polyfit(a, d, 1), a)
    re = e - np.polyval(np.polyfit(a, e, 1), a)
    return stats.spearmanr(rd, re)


def main():
    df = pd.read_csv(RESULTS / "pooled_error_vs_D.csv")
    panels = [("All scans", df), ("Without IOP", df[df["dataset"] != "IXI-IOP"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for key, (label, color) in GROUPS.items():
            g = sub[sub["dataset"] == key]
            if len(g):
                ax.scatter(g["D"], g["abs_error"], s=14, color=color, alpha=0.6, linewidths=0,
                           label=f"{label} (n={len(g)})", zorder=3)
        r, p = partial_spearman(sub)
        ax.set_title(f"{title} (n={len(sub)})", fontsize=10.5, color=INK, loc="left")
        ax.text(0.97, 0.97, f"age-controlled partial r = {r:+.3f}\np = {p:.2g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
        ax.set_xlabel("Distance D from the IXI Guys+HH reference", fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("|DeepBrainNet error| (years)", fontsize=9, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, ncol=3,
               loc="upper left", bbox_to_anchor=(0.01, 0.92))
    fig.suptitle("The pooled error-vs-distance effect (Stage R3) comes from the IOP scans",
                 fontsize=11, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = RESULTS / "fig_r3_iop_dependence.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
