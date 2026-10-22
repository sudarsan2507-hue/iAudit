"""BrainAgeNeXt: does disagreement between its 5 models predict the error?

Reads results/brainagenext_predictions.csv and results/brainagenext_uncertainty.csv
(both from export_brainagenext_data.py). Writes results/fig_bang_uncertainty.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = {"IXI": ("IXI Guys+HH", "#2a78d6"), "eNKI": ("eNKI", "#eb6834")}


def main():
    df = pd.read_csv(RESULTS / "brainagenext_predictions.csv")
    df["abs_error"] = (df["predicted_age"] - df["chronological_age"]).abs()
    unc = pd.read_csv(RESULTS / "brainagenext_uncertainty.csv").set_index("scope")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ds, (label, color)) in zip(axes, PANELS.items()):
        ax.set_facecolor(SURFACE)
        sub = df[df["dataset"] == ds]
        ax.scatter(sub["ensemble_sd"], sub["abs_error"], s=14, color=color, alpha=0.6, linewidths=0, zorder=3)
        u = unc.loc["IXI Guys+HH" if ds == "IXI" else "eNKI"]
        ax.set_title(f"{label} (n={len(sub)})", fontsize=10.5, color=INK, loc="left")
        ax.text(0.97, 0.97, f"age-controlled partial r = {u['partial_spearman_r_sd_vs_abs_error']:+.3f}\n"
                f"p = {u['p']:.2g}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
                linespacing=1.5, bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
        ax.set_xlabel("Disagreement between the 5 models (SD, years)", fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("|BrainAgeNeXt error| (years)", fontsize=9, color=INK2)
    fig.suptitle("BrainAgeNeXt ensemble disagreement vs error: weak signal on IXI, none on eNKI",
                 fontsize=11, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS / "fig_bang_uncertainty.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
