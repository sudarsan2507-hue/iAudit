"""Stage 5 - fairness audit of brain-age predictions.

Loads every results/predictions_*.csv, computes an age-bias-corrected
prediction gap per model, stratifies it by sex / site / ethnicity (and their
intersection), and reports whether disparities across subgroups are
statistically real or plausibly chance.
"""

import itertools
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(r"D:\Projects\iAuditAge\results")
MIN_GROUP_N = 10  # below this, a subgroup is "underpowered" - flagged, not hidden


# --------------------------------------------------------------------------
# 1. Load + pool every model's predictions
# --------------------------------------------------------------------------
def load_all_predictions():
    frames = []
    for f in sorted(RESULTS.glob("predictions_*.csv")):
        df = pd.read_csv(f)
        frames.append(df)
        print(f"loaded {f.name}: {len(df)} rows, model(s) = {df['model_name'].unique().tolist()}")
    if not frames:
        raise SystemExit(f"no results/predictions_*.csv files found in {RESULTS}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 2-3. Gap + age-bias correction
# --------------------------------------------------------------------------
def add_gap_and_correction(df):
    """Fit ONE linear model gap ~ chronological_age on the POOLED cohort per
    model (never per subgroup - fitting per subgroup would absorb the very
    bias we're auditing). corrected_gap = gap - (a*age + b).

    This mirrors the de Lange & Cole (2020) age-bias-correction approach for
    brain-age gap estimates. Fitting the correction on the same sample being
    audited is standard practice in this literature but is a real limitation:
    it removes the population-average age trend for THIS cohort, not an
    independent reference trend, so any age-correlated subgroup imbalance in
    the cohort composition could partially confound the "corrected" result.
    """
    df = df.copy()
    df["gap"] = df["predicted_age"] - df["chronological_age"]

    df["corrected_gap"] = np.nan
    df["fit_slope"] = np.nan
    df["fit_intercept"] = np.nan
    for model, sub in df.groupby("model_name"):
        slope, intercept, r, p, se = stats.linregress(sub["chronological_age"], sub["gap"])
        pred_gap = slope * sub["chronological_age"] + intercept
        df.loc[sub.index, "corrected_gap"] = sub["gap"] - pred_gap
        df.loc[sub.index, "fit_slope"] = slope
        df.loc[sub.index, "fit_intercept"] = intercept
        print(f"[{model}] age-bias fit: gap = {slope:.4f}*age + {intercept:.4f}  (r={r:.3f})")
    return df


# --------------------------------------------------------------------------
# 4-5. Stratify + per-subgroup summary
# --------------------------------------------------------------------------
def subgroup_summary(df, axis, model):
    """Return a DataFrame of n, MAE, mean_gap per subgroup for one axis."""
    rows = []
    for grp, sub in df.groupby(axis):
        rows.append({
            "model_name": model,
            "axis": axis,
            "group": grp,
            "n": len(sub),
            "MAE": sub["corrected_gap"].abs().mean(),
            "mean_corrected_gap": sub["corrected_gap"].mean(),
            "underpowered": len(sub) < MIN_GROUP_N,
        })
    return pd.DataFrame(rows)


def usable_axis(df, axis):
    """An axis is usable if >=2 groups have n >= MIN_GROUP_N."""
    counts = df[axis].value_counts()
    usable_groups = counts[counts >= MIN_GROUP_N]
    return len(usable_groups) >= 2, usable_groups


# --------------------------------------------------------------------------
# 6. Stats: Mann-Whitney (2 groups) / Kruskal-Wallis (3+) + post-hoc
# --------------------------------------------------------------------------
def axis_stats(df, axis, model):
    """Omnibus test + Bonferroni-corrected pairwise post-hoc with effect
    size (median difference). Only groups with n >= MIN_GROUP_N are tested."""
    counts = df[axis].value_counts()
    groups = counts[counts >= MIN_GROUP_N].index.tolist()
    samples = {g: df.loc[df[axis] == g, "corrected_gap"].values for g in groups}

    out = {"axis": axis, "model_name": model, "groups_tested": groups}

    if len(groups) < 2:
        out["omnibus_test"] = None
        out["omnibus_p"] = None
        out["pairwise"] = []
        return out

    if len(groups) == 2:
        g1, g2 = groups
        stat, p = stats.mannwhitneyu(samples[g1], samples[g2], alternative="two-sided")
        out["omnibus_test"] = "Mann-Whitney U"
        out["omnibus_p"] = p
        pairwise = [(g1, g2, p, np.median(samples[g1]) - np.median(samples[g2]))]
    else:
        stat, p = stats.kruskal(*samples.values())
        out["omnibus_test"] = "Kruskal-Wallis"
        out["omnibus_p"] = p
        pairs = list(itertools.combinations(groups, 2))
        bonf = len(pairs)
        pairwise = []
        for g1, g2 in pairs:
            _, p_raw = stats.mannwhitneyu(samples[g1], samples[g2], alternative="two-sided")
            p_adj = min(p_raw * bonf, 1.0)
            med_diff = np.median(samples[g1]) - np.median(samples[g2])
            pairwise.append((g1, g2, p_adj, med_diff))

    out["pairwise"] = pairwise
    return out


# --------------------------------------------------------------------------
# 7. Intersectional: sex x site
# --------------------------------------------------------------------------
def intersectional_summary(df, model):
    rows = []
    for (sex, site), sub in df.groupby(["sex", "site"]):
        rows.append({
            "model_name": model,
            "sex": sex,
            "site": site,
            "n": len(sub),
            "MAE": sub["corrected_gap"].abs().mean(),
            "mean_corrected_gap": sub["corrected_gap"].mean(),
            "underpowered": len(sub) < MIN_GROUP_N,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9. Plots
# --------------------------------------------------------------------------
def make_plots(df, models):
    for model in models:
        sub = df[df["model_name"] == model]
        tag = model.replace(" ", "_")

        # boxplot corrected_gap by sex
        fig, ax = plt.subplots(figsize=(5, 5))
        groups = [g["corrected_gap"].values for _, g in sub.groupby("sex")]
        labels = sub["sex"].unique().tolist()
        labels = sorted(sub["sex"].dropna().unique().tolist())
        groups = [sub.loc[sub["sex"] == l, "corrected_gap"].values for l in labels]
        ax.boxplot(groups, tick_labels=labels)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_ylabel("corrected brain-age gap (years)")
        ax.set_title(f"{model}: corrected gap by sex")
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_boxplot_sex_{tag}.png", dpi=150)
        plt.close(fig)

        # MAE by site (bar)
        site_summary = subgroup_summary(sub, "site", model)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(site_summary["group"].astype(str), site_summary["MAE"])
        ax.set_ylabel("MAE of corrected gap (years)")
        ax.set_title(f"{model}: MAE by site")
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_mae_by_site_{tag}.png", dpi=150)
        plt.close(fig)

        # Bland-Altman: gap vs age, before and after correction
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, col, title in [
            (axes[0], "gap", "before correction"),
            (axes[1], "corrected_gap", "after correction"),
        ]:
            ax.scatter(sub["chronological_age"], sub[col], s=10, alpha=0.4)
            slope, intercept, r, p, se = stats.linregress(sub["chronological_age"], sub[col])
            xs = np.array([sub["chronological_age"].min(), sub["chronological_age"].max()])
            ax.plot(xs, slope * xs + intercept, color="red", lw=1.5,
                     label=f"slope={slope:.3f}")
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_xlabel("chronological age")
            ax.set_ylabel("gap (years)")
            ax.set_title(f"{model}: {title}")
            ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_bland_altman_{tag}.png", dpi=150)
        plt.close(fig)

    # multi-model subgroup-bias comparison (still produced with 1 model;
    # structure is ready for when a second model is added)
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        sub = df[df["model_name"] == model]
        sex_summary = subgroup_summary(sub, "sex", model).sort_values("group")
        x = np.arange(len(sex_summary))
        ax.bar(x + i * width, sex_summary["mean_corrected_gap"], width, label=model)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(sex_summary["group"])
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("mean corrected gap (years)")
    ax.set_title("Multi-model subgroup bias comparison (by sex)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_multimodel_subgroup_comparison.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    raw = load_all_predictions()
    df = add_gap_and_correction(raw)
    models = sorted(df["model_name"].unique())
    print(f"\nmodels found: {models}")

    bias_rows = []
    checklist_lines = ["# Fairness Checklist\n"]

    print("\n" + "=" * 70)
    print("PER-SUBGROUP SUMMARIES (corrected_gap: n, MAE, mean)")
    print("=" * 70)

    for model in models:
        sub = df[df["model_name"] == model]
        print(f"\n--- model: {model} (n={len(sub)}) ---")

        for axis in ["sex", "site", "ethnicity"]:
            usable, counts = usable_axis(sub, axis)
            if not usable:
                msg = f"  [{axis}] SKIPPED - fewer than 2 groups with n>={MIN_GROUP_N} (counts: {dict(counts)})"
                print(msg)
                checklist_lines.append(f"- **{axis}** ({model}): skipped - {msg.strip()}")
                continue

            summary = subgroup_summary(sub[sub[axis].isin(counts.index)], axis, model)
            print(f"\n  [{axis}]")
            print(summary[["group", "n", "MAE", "mean_corrected_gap", "underpowered"]]
                  .to_string(index=False))
            for _, r in summary.iterrows():
                bias_rows.append({
                    "model_name": model, "axis": axis, "group": r["group"],
                    "n": r["n"], "MAE": r["MAE"], "mean_gap": r["mean_corrected_gap"],
                    "p_value": None, "test": None, "underpowered": r["underpowered"],
                })

            st = axis_stats(sub, axis, model)
            if st["omnibus_test"]:
                print(f"    omnibus: {st['omnibus_test']}  p = {st['omnibus_p']:.4g}")
                for g1, g2, p_adj, med_diff in st["pairwise"]:
                    sig = "**SIGNIFICANT**" if p_adj < 0.05 else "n.s."
                    print(f"    {g1} vs {g2}: median diff = {med_diff:+.2f}y, "
                          f"p(Bonferroni) = {p_adj:.4g}  [{sig}]")
                    bias_rows.append({
                        "model_name": model, "axis": axis,
                        "group": f"{g1} vs {g2}", "n": None, "MAE": None,
                        "mean_gap": med_diff, "p_value": p_adj,
                        "test": f"pairwise post-hoc ({st['omnibus_test']})",
                        "underpowered": False,
                    })
                checklist_lines.append(
                    f"- **{axis}** ({model}): {st['omnibus_test']} p={st['omnibus_p']:.4g}; "
                    + "; ".join(f"{g1} vs {g2} diff={d:+.2f}y p={p:.4g}"
                                for g1, g2, p, d in st["pairwise"])
                )

    print("\n" + "=" * 70)
    print("INTERSECTIONAL: sex x site")
    print("=" * 70)
    for model in models:
        sub = df[df["model_name"] == model]
        inter = intersectional_summary(sub, model)
        print(f"\n--- model: {model} ---")
        print(inter.to_string(index=False))
        for _, r in inter.iterrows():
            flag = " (UNDERPOWERED, n<10)" if r["underpowered"] else ""
            bias_rows.append({
                "model_name": model, "axis": "sex_x_site",
                "group": f"{r['sex']}/{r['site']}{flag}", "n": r["n"],
                "MAE": r["MAE"], "mean_gap": r["mean_corrected_gap"],
                "p_value": None, "test": None, "underpowered": r["underpowered"],
            })

    print("\n" + "=" * 70)
    print("MULTI-MODEL COMPARISON")
    print("=" * 70)
    if len(models) == 1:
        print(f"Only one model ({models[0]}) has results this session - "
              f"Stage 4B (second model) was skipped. No cross-model comparison possible.")
        checklist_lines.append(
            f"\n**Multi-model comparison:** not available - only {models[0]} was run this session.")
    else:
        for axis in ["sex", "site"]:
            print(f"\n[{axis}] mean_corrected_gap by model:")
            piv = df.pivot_table(index=axis, columns="model_name",
                                  values="corrected_gap", aggfunc="mean")
            print(piv.to_string())

    make_plots(df, models)

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(RESULTS / "bias_report.csv", index=False)
    print(f"\nwrote {RESULTS / 'bias_report.csv'}  ({len(bias_df)} rows)")

    (RESULTS / "fairness_checklist.md").write_text("\n".join(checklist_lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'fairness_checklist.md'}")

    # ---------------- VERIFY 5 ----------------
    print("\n" + "=" * 70)
    print("VERIFY 5 - SANITY CHECKS")
    print("=" * 70)
    all_pass = True
    for model in models:
        sub = df[df["model_name"] == model]
        print(f"\n--- model: {model} ---")

        r_corr, _ = stats.pearsonr(sub["corrected_gap"], sub["chronological_age"])
        ok = abs(r_corr) < 0.10
        all_pass &= ok
        print(f"  |r(corrected_gap, age)| < 0.10 : {'PASS' if ok else 'FAIL'}  (r={r_corr:.4f})")

        slope_before, *_ = stats.linregress(sub["chronological_age"], sub["gap"])
        slope_after, *_ = stats.linregress(sub["chronological_age"], sub["corrected_gap"])
        neg_ok = slope_before < 0
        flat_ok = abs(slope_after) < 0.05
        all_pass &= neg_ok and flat_ok
        print(f"  slope before correction (expect negative): {slope_before:.4f}  "
              f"{'PASS' if neg_ok else 'FAIL'}")
        print(f"  slope after correction (expect ~flat)    : {slope_after:.4f}  "
              f"{'PASS' if flat_ok else 'FAIL'}")

        mean_gap = sub["corrected_gap"].mean()
        mean_ok = abs(mean_gap) < 0.2
        all_pass &= mean_ok
        print(f"  mean corrected_gap ~ 0 (+/-0.2): {mean_gap:+.4f}  "
              f"{'PASS' if mean_ok else 'FAIL'}")

        n_check = bias_df[(bias_df["model_name"] == model) & (bias_df["axis"] == "sex")]["n"].sum()
        n_ok = n_check == len(sub)
        all_pass &= n_ok
        print(f"  subgroup n's (sex) sum to model total ({len(sub)}): {n_check}  "
              f"{'PASS' if n_ok else 'FAIL'}")

    print("\n" + "=" * 70)
    print("ALL VERIFY 5 CHECKS PASSED" if all_pass else "SOME VERIFY 5 CHECKS FAILED - do not trust bias findings until fixed")
    print("=" * 70)


if __name__ == "__main__":
    main()
