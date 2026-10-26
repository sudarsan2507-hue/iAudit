"""SALD held-out test: analysis of colab/iaudit_colab_sald.ipynb outputs.

SALD was only a test set in the BrainAgeNeXt paper, so these are clean held-out numbers.

Inputs (copied into results/ from the notebook's sald_outputs/):
  results/sald_predictions.csv    per-scan predictions, both CNNs + IXI-trained heads
  results/sald_mae.csv            notebook's own summary (re-checked here)
  results/transfer_head_predictions.csv, results/brainagenext_predictions.csv  (IXI/eNKI, for comparison)

Writes:
  results/sald_summary.csv        MAE with bootstrap 95% CIs per model x cohort
  results/sald_sex_audit.csv      Stage 5 audit (age-corrected gap by sex) on SALD
  results/fig_mae_by_cohort.png   MAE +/- 95% CI, IXI / eNKI / SALD
  results/fig_sald_age_bins.png   MAE by age decade on SALD
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fairness_audit import add_gap_and_correction, axis_stats, subgroup_summary

RESULTS = Path(__file__).resolve().parent.parent / "results"
B, SEED = 5000, 0
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {"DeepBrainNet": "#2a78d6", "BrainAgeNeXt": "#eb6834"}


def boot_mae(err, rng):
    err = np.abs(np.asarray(err))
    idx = rng.integers(0, len(err), (B, len(err)))
    m = err[idx].mean(1)
    return err.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    rng = np.random.default_rng(SEED)
    sald = pd.read_csv(RESULTS / "sald_predictions.csv")
    nb = pd.read_csv(RESULTS / "sald_mae.csv").set_index("model")

    # 1. re-check the notebook's numbers from the per-scan file
    for col, name in [("dbn_raw", "DeepBrainNet raw"), ("bang_raw", "BrainAgeNeXt raw"),
                      ("pred BANG 512 + ridge", "BANG 512 + ridge (trained on IXI)")]:
        mine = np.abs(sald[col] - sald["age"]).mean()
        assert abs(mine - nb.loc[name, "MAE"]) < 1e-3, (name, mine, nb.loc[name, "MAE"])
    print(f"re-check: per-scan MAEs match the notebook summary (n={len(sald)})")

    # 2. MAE with bootstrap CIs, all cohorts
    ixi = pd.read_csv(RESULTS / "transfer_head_predictions.csv")
    cohorts = {"IXI Guys+HH": ixi[ixi["site"].isin(["Guys", "HH"])], "eNKI": ixi[ixi["site"] == "eNKI"],
               "SALD": sald}
    rows = []
    for cohort, df in cohorts.items():
        for col, model in [("dbn_raw", "DeepBrainNet"), ("bang_raw", "BrainAgeNeXt")]:
            err = df[col] - df["age"]
            mae, lo, hi = boot_mae(err, rng)
            rows.append({"cohort": cohort, "model": model, "n": len(df), "MAE": mae, "ci_low": lo, "ci_high": hi,
                         "mean_error": err.mean(), "r": np.corrcoef(df[col], df["age"])[0, 1],
                         "in_bang_training_data": cohort != "SALD"})
    summ = pd.DataFrame(rows)
    summ.round(3).to_csv(RESULTS / "sald_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))

    # paired bootstrap: BrainAgeNeXt minus DeepBrainNet MAE on SALD
    a, d = np.abs(sald["bang_raw"] - sald["age"]).to_numpy(), np.abs(sald["dbn_raw"] - sald["age"]).to_numpy()
    idx = rng.integers(0, len(a), (B, len(a)))
    diff = a[idx].mean(1) - d[idx].mean(1)
    print(f"\nSALD MAE difference BrainAgeNeXt - DeepBrainNet: {a.mean() - d.mean():+.2f} y "
          f"(95% CI {np.percentile(diff, 2.5):+.2f} to {np.percentile(diff, 97.5):+.2f})")
    bang_sald = summ[(summ.cohort == "SALD") & (summ.model == "BrainAgeNeXt")].iloc[0]
    print(f"BrainAgeNeXt SALD MAE {bang_sald.MAE:.2f} y, 95% CI {bang_sald.ci_low:.2f}-{bang_sald.ci_high:.2f}: "
          f"{'entirely below' if bang_sald.ci_high < 4 else 'NOT entirely below'} 4 y")

    # 3. sex audit on SALD (Stage 5 method: pooled age-bias correction, Mann-Whitney)
    long = pd.concat([pd.DataFrame({"subject_id": sald["scan_id"], "chronological_age": sald["age"],
                                    "predicted_age": sald[col], "sex": sald["sex"], "site": "SALD",
                                    "ethnicity": "n/a", "model_name": name})
                      for col, name in [("dbn_raw", "DeepBrainNet"), ("bang_raw", "BrainAgeNeXt")]])
    long = add_gap_and_correction(long.reset_index(drop=True))
    sex_rows = []
    for name in ["DeepBrainNet", "BrainAgeNeXt"]:
        sub = long[long["model_name"] == name]
        st = axis_stats(sub, "sex", name)
        for r in subgroup_summary(sub, "sex", name).itertuples():
            sex_rows.append({"model": name, "sex": r.group, "n": r.n, "MAE_corrected": r.MAE,
                             "mean_corrected_gap": r.mean_corrected_gap, "mann_whitney_p": st["omnibus_p"]})
    sex = pd.DataFrame(sex_rows)
    sex.round(4).to_csv(RESULTS / "sald_sex_audit.csv", index=False)
    print("\nSALD sex audit (age-corrected gap):\n" + sex.round(3).to_string(index=False))

    # 4. figures
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.arange(len(cohorts))
    for k, model in enumerate(MODEL_COLORS):
        s = summ[summ["model"] == model].set_index("cohort").loc[list(cohorts)]
        x = xs + (k - 0.5) * 0.22
        ax.errorbar(x, s["MAE"], yerr=[s["MAE"] - s["ci_low"], s["ci_high"] - s["MAE"]], fmt="o", ms=8,
                    color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model], elinewidth=2, capsize=4, label=model,
                    zorder=3)
        for xi, v in zip(x, s["MAE"]):
            ax.text(xi + 0.06, v, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.axhline(4, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(cohorts) - 0.45, 4.05, "4 y target", fontsize=8, color=INK2, va="bottom", ha="right")
    labels = [f"{c}\n(n={len(df)})" + ("" if c == "SALD" else "\n† in BrainAgeNeXt training")
              for c, df in cohorts.items()]
    ax.set_xticks(xs, labels, fontsize=8.5, color=INK)
    ax.set_ylabel("MAE, years (95% bootstrap CI)", fontsize=9, color=INK2)
    ax.set_ylim(2, 8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("MAE by cohort - SALD is the clean held-out test for BrainAgeNeXt", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "fig_mae_by_cohort.png", facecolor=SURFACE)

    bins = pd.cut(sald["age"], [19, 30, 40, 50, 60, 70, 81], right=False)
    labels = ["20s", "30s", "40s", "50s", "60s", "70s"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.38
    for k, (col, model) in enumerate([("dbn_raw", "DeepBrainNet"), ("bang_raw", "BrainAgeNeXt")]):
        m = (sald[col] - sald["age"]).abs().groupby(bins, observed=True).mean().to_numpy()
        x = np.arange(len(m)) + (k - 0.5) * (w + 0.03)
        ax.bar(x, m, w, color=MODEL_COLORS[model], label=model, zorder=3)
        for xi, v in zip(x, m):
            ax.text(xi, v + 0.12, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
    ax.axhline(4, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    n_bin = bins.value_counts(sort=False).to_numpy()
    ax.set_xticks(range(len(labels)), [f"{lab}\n(n={n})" for lab, n in zip(labels, n_bin)], fontsize=8.5,
                  color=INK)
    ax.set_ylabel("MAE (years)", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("SALD: MAE by age decade (dashed = 4 y target)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "fig_sald_age_bins.png", facecolor=SURFACE)
    print("\nwrote sald_summary.csv, sald_sex_audit.csv, fig_mae_by_cohort.png, fig_sald_age_bins.png")


if __name__ == "__main__":
    main()
