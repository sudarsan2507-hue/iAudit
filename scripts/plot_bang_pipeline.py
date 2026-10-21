"""Diagram of how BrainAgeNeXt and DeepBrainNet were run on Colab (colab/iaudit_colab_auto.ipynb).

Writes results/fig_bang_pipeline.png. Every box describes code that exists in the notebook.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
NEUTRAL, DBN, BANG = "#8a8985", "#2a78d6", "#eb6834"


def box(ax, x0, y0, x1, y1, title, body, edge):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=SURFACE, ec=edge, lw=1.6, zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, y1 - 2.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK, zorder=3)
    ax.text(cx, y1 - 5.6, body, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.45, zorder=3)


def arrow(ax, p, q, color=NEUTRAL):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.4, color=color,
                                 shrinkA=0, shrinkB=0, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 54)
    ax.axis("off")

    box(ax, 2, 22, 21, 36, "T1 MRI scan", "IXI (NITRC), eNKI (S3)\nMD5-checked against\nthe laptop copy", NEUTRAL)
    box(ax, 25, 20, 53, 38, "Shared preprocessing", "ANTsPyNet: N4 bias correction,\ndenoising, brain extraction,\n"
        "affine registration to MNI152\nsaved once as *_prep.nii.gz", NEUTRAL)
    box(ax, 58, 39, 101, 53, "DeepBrainNet  (TensorFlow)", "80 axial slices → VGG16 2D CNN\n"
        "age = median of slice predictions\nfeatures: GAP-512, dense-1024", DBN)
    box(ax, 58, 2, 81, 18, "BrainAgeNeXt input", "1 mm, crop/pad 160×192×160,\nz-normalise, orientation flip\n"
        "(best of 4, locked on\n20 Guys+HH scans)", BANG)
    box(ax, 85, 2, 107, 18, "5-model ensemble", "5 × MedNeXt 3D encoder\n512-d pooled features each\n"
        "age = median of 5\n(PyTorch, ~5 s/scan on T4)", BANG)
    box(ax, 111, 20, 128, 38, "Ridge head", "trained on IXI\nGuys+HH only\n5-fold out-of-fold\n→ MAE on IXI\nand eNKI", NEUTRAL)

    arrow(ax, (21, 29), (25, 29))
    arrow(ax, (53, 33), (58, 46), DBN)
    arrow(ax, (53, 25), (58, 10), BANG)
    arrow(ax, (81, 10), (85, 10), BANG)
    arrow(ax, (101, 46), (111, 33), DBN)
    arrow(ax, (107, 10), (111, 25), BANG)

    ax.text(2, 13, "Safety checks (run stops itself and frees the runtime on failure):", fontsize=8,
            fontweight="bold", color=INK, va="top")
    ax.text(2, 9.6, "- each session re-runs IXI002 and must match the laptop\n"
            "  (DeepBrainNet within 1 y, BrainAgeNeXt within 1.5 y)\n"
            "- drift and failure watchdogs; every saved file re-checked",
            fontsize=7.8, color=INK2, va="top", linespacing=1.45)
    fig.suptitle("How BrainAgeNeXt and DeepBrainNet were run (colab/iaudit_colab_auto.ipynb)",
                 fontsize=11, color=INK, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS / "fig_bang_pipeline.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
