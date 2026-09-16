"""Stage T4 - cross-model comparison on the common cohort (subjects present
in every one of the 4 models: the 544 with tabular features, intersected
with the 560 DeepBrainNet subjects - features.csv subjects are a subset of
DeepBrainNet's, so the common set is exactly the 544).

Per the T4 addendum: report RAW gap, corrected gap, and MAE side by side per
site/model, and decide the IOP claim on MAE (absolute, comparable across
models) rather than on corrected-gap sign (relative to each model's own
mean, not directly comparable across models of very different accuracy).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(r"D:\Projects\iAuditAge\results")
MODELS = ["DeepBrainNet", "LinearRegression", "RandomForest", "XGBoost"]


def load_common_cohort():
    feat = pd.read_csv(RESULTS / "features.csv")
    common_ids = set(feat["subject_id"])

    frames = []
    for model in MODELS:
        fname = f"predictions_{model.lower()}.csv"
        df = pd.read_csv(RESULTS / fname)
        df = df[df["subject_id"].isin(common_ids)].copy()
        assert len(df) == len(common_ids), (
            f"{model}: expected {len(common_ids)} common subjects, got {len(df)}"
        )
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df, common_ids


def add_gap_and_correction(df):
    """Same pooled-fit approach as fairness_audit.py, refit here on the
    common-cohort subset so corrected_gap is consistent within this script."""
    df = df.copy()
    df["gap"] = df["predicted_age"] - df["chronological_age"]
    df["corrected_gap"] = np.nan
    for model, sub in df.groupby("model_name"):
        slope, intercept, r, p, se = stats.linregress(sub["chronological_age"], sub["gap"])
        pred_gap = slope * sub["chronological_age"] + intercept
        df.loc[sub.index, "corrected_gap"] = sub["gap"] - pred_gap
    return df


def build_table(df, axis, value_col, agg):
    """rows = subgroup, cols = model, cells = agg(value_col)."""
    pivot = df.groupby([axis, "model_name"])[value_col].agg(agg).unstack("model_name")
    pivot = pivot[MODELS]
    return pivot


def main():
    df, common_ids = load_common_cohort()
    print(f"common-set size (present in all 4 models): {len(common_ids)}")

    df = add_gap_and_correction(df)
    df["abs_err"] = (df["predicted_age"] - df["chronological_age"]).abs()

    SEX_ORDER = ["Female", "Male"]
    SITE_ORDER = ["Guys", "HH", "IOP"]
    sex_map = {"F": "Female", "M": "Male"}
    df["sex_label"] = df["sex"].map(sex_map)

    # ---------------- Table A: mean corrected gap ----------------
    site_corr = build_table(df, "site", "corrected_gap", "mean").reindex(SITE_ORDER)
    sex_corr = build_table(df, "sex_label", "corrected_gap", "mean").reindex(SEX_ORDER)

    # ---------------- Table: RAW mean gap ----------------
    site_raw = build_table(df, "site", "gap", "mean").reindex(SITE_ORDER)
    sex_raw = build_table(df, "sex_label", "gap", "mean").reindex(SEX_ORDER)

    # ---------------- Table B: MAE ----------------
    site_mae = build_table(df, "site", "abs_err", "mean").reindex(SITE_ORDER)
    sex_mae = build_table(df, "sex_label", "abs_err", "mean").reindex(SEX_ORDER)

    print("\n" + "=" * 70)
    print("TABLE A1 - RAW mean gap (predicted - chronological, NO correction), by SITE")
    print("=" * 70)
    print(site_raw.round(3).to_string())

    print("\n" + "=" * 70)
    print("TABLE A2 - corrected mean gap (age-bias-corrected), by SITE")
    print("=" * 70)
    print(site_corr.round(3).to_string())

    print("\n" + "=" * 70)
    print("TABLE B - MAE (uncorrected absolute error), by SITE")
    print("=" * 70)
    print(site_mae.round(3).to_string())

    print("\n" + "=" * 70)
    print("TABLE C1 - RAW mean gap, by SEX")
    print("=" * 70)
    print(sex_raw.round(3).to_string())

    print("\n" + "=" * 70)
    print("TABLE C2 - corrected mean gap, by SEX")
    print("=" * 70)
    print(sex_corr.round(3).to_string())

    print("\n" + "=" * 70)
    print("TABLE D - MAE, by SEX")
    print("=" * 70)
    print(sex_mae.round(3).to_string())

    # ---------------- IOP claim: decide on MAE, not corrected-gap sign ----------------
    print("\n" + "=" * 70)
    print("Q1: IS IOP ANOMALOUS TO EVERY MODEL? (decided on MAE, the absolute,")
    print("    cross-model-comparable metric - NOT on corrected-gap sign)")
    print("=" * 70)
    iop_worst_for = []
    for model in MODELS:
        site_mae_this_model = site_mae[model]
        worst_site = site_mae_this_model.idxmax()
        is_iop_worst = worst_site == "IOP"
        iop_worst_for.append(is_iop_worst)
        print(f"  {model:18s} MAE by site: " +
              "  ".join(f"{s}={site_mae_this_model[s]:.2f}y" for s in SITE_ORDER) +
              f"   -> worst site = {worst_site}" + ("  [IOP worst]" if is_iop_worst else ""))

    all_iop_worst = all(iop_worst_for)
    print()
    if all_iop_worst:
        print("ANSWER: IOP has the WORST MAE for ALL 4 models.")
        print("-> IOP is anomalous to every architecture tested here. This supports a")
        print("   scanner/acquisition artifact that is ARCHITECTURE-INDEPENDENT - a")
        print("   defensible, strong claim, regardless of corrected-gap sign.")
    else:
        n_worst = sum(iop_worst_for)
        print(f"ANSWER: IOP has the worst MAE for {n_worst}/4 models "
              f"({', '.join(m for m, f in zip(MODELS, iop_worst_for) if f)}).")
        print("-> The IOP accuracy problem is NOT uniform across all 4 architectures -")
        print("   it is model/architecture-specific for at least one model family.")

    # ---------------- Q2: sex split ----------------
    print("\n" + "=" * 70)
    print("Q2: DOES THE MALE-OLDER / FEMALE-YOUNGER SPLIT REPLICATE?")
    print("=" * 70)
    print("RAW gap sign by sex (this is the metric NOT confounded by each model's")
    print("own age-correction baseline):")
    print(sex_raw.round(3).to_string())
    male_raw_signs = np.sign(sex_raw.loc["Male"])
    female_raw_signs = np.sign(sex_raw.loc["Female"])
    uniform_male_positive = (male_raw_signs > 0).all()
    uniform_female_negative = (female_raw_signs < 0).all()
    if uniform_male_positive and uniform_female_negative:
        print("\nANSWER: YES - every model's RAW gap is positive for Male and negative for")
        print("Female. The male-older/female-younger directional split REPLICATES across")
        print("all 4 architectures on raw (uncorrected) predictions.")
    else:
        print("\nANSWER: The RAW-gap sex split does NOT replicate uniformly across all 4")
        print("models (see signs above) - it is not a simple universal offset.")

    # ---------------- Q3: interpretation - explain the sign flip, don't force a story ----------------
    print("\n" + "=" * 70)
    print("Q3: INTERPRETATION - why corrected-gap SIGN differs by model family")
    print("=" * 70)
    print("The corrected_gap sign flip seen earlier between DeepBrainNet and the")
    print("tabular models (e.g. IOP: DeepBrainNet +4.89 vs tabular models all")
    print("negative) is a baseline artifact of age-bias correction, NOT evidence")
    print("that the models disagree about which subjects are hard:")
    print()
    print("  - corrected_gap is defined RELATIVE TO EACH MODEL'S OWN mean age-trend")
    print("    fit on its own predictions. A highly accurate model (DeepBrainNet,")
    print("    MAE~5.5y) and much weaker models (tabular, MAE~7-9y) have very")
    print("    different own-mean baselines, so a subgroup's corrected_gap SIGN is")
    print("    not directly comparable across models of different accuracy.")
    print("  - RAW gap and MAE are absolute (not baseline-relative) and ARE directly")
    print("    comparable across models - that is why Q1/Q2 above are decided on")
    print("    MAE and raw gap, not on corrected-gap sign.")
    print("  - A plausible mechanistic reason IOP is hard for the volume-based")
    print("    tabular models specifically: deep_atropos segmentation quality can")
    print("    be sensitive to scanner/acquisition-protocol differences (IOP is a")
    print("    different scanner site than Guys/HH), so systematically distorted")
    print("    tissue-volume features for IOP subjects would degrade the tabular")
    print("    models' predictions for that site in a way not necessarily matching")
    print("    DeepBrainNet's own (differently-sourced) failure mode on the same")
    print("    scans.")
    print()
    print(f"Final interpretation depends on the Q1 answer above: "
          f"{'IOP is the worst site for ALL 4 models -> uniform, architecture-independent scanner/acquisition artifact.' if all_iop_worst else 'IOP MAE degradation is not uniform across all 4 -> at least partly architecture/model-specific.'}")

    # ---------------- plots ----------------
    def grouped_bar(pivot, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(8, 5))
        n_groups = len(pivot.index)
        n_models = len(MODELS)
        width = 0.8 / n_models
        x = np.arange(n_groups)
        for i, model in enumerate(MODELS):
            ax.bar(x + i * width, pivot[model].values, width, label=model)
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(pivot.index)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=120)
        plt.close(fig)
        print(f"wrote {fname}")

    grouped_bar(site_corr, "Corrected mean gap by site, per model", "corrected gap (years)",
                "fig_crossmodel_site_bias.png")
    grouped_bar(sex_corr, "Corrected mean gap by sex, per model", "corrected gap (years)",
                "fig_crossmodel_sex_bias.png")

    return {
        "common_n": len(common_ids),
        "site_raw": site_raw, "site_corr": site_corr, "site_mae": site_mae,
        "sex_raw": sex_raw, "sex_corr": sex_corr, "sex_mae": sex_mae,
        "all_iop_worst": all_iop_worst,
    }


if __name__ == "__main__":
    main()
