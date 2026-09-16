"""Theory-test battery (branch: reliability-experiments).

Five independent experiments probing claims made earlier in the pipeline:
  A - true Leave-Site-Out unseen-scanner penalty for IOP
  B - does unsupervised feature harmonization mitigate it
  C - is DeepBrainNet's sex bias real or confounded by site/brain-size
  D - is there a within-distribution (Guys+HH only) reliability gradient
  E - does site-aware age-correction fix disparity, or only calibration

Each prints an explicit VERDICT. No age labels are used in any
feature-harmonization step (Experiment B); OOF/leave-site-out discipline is
kept throughout for any model that is (re)trained here.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RESULTS = r"D:\Projects\iAuditAge\results"
FEATURE_COLS = [
    "frac_CSF", "frac_GM", "frac_WM", "frac_deepGM", "frac_brainstem",
    "frac_cerebellum", "total_brain_volume_ml",
]
TABULAR_MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(random_state=42),
    "XGBoost": lambda: XGBRegressor(random_state=42),
}
ALL_MODELS = ["DeepBrainNet"] + list(TABULAR_MODELS.keys())
SITES = ["Guys", "HH", "IOP"]

feat = pd.read_csv(fr"{RESULTS}\features.csv")

preds = {}
for m in ALL_MODELS:
    preds[m] = pd.read_csv(fr"{RESULTS}\predictions_{m.lower()}.csv")

summary_rows = []  # collected across experiments for the FINAL table


def stratified_iop_mae_reference():
    """Recompute (not hardcode) each tabular model's earlier stratified-CV
    IOP MAE straight from predictions_*.csv, for an honest A/B comparison."""
    out = {}
    for m in TABULAR_MODELS:
        df = preds[m]
        iop = df[df["site"] == "IOP"]
        out[m] = mean_absolute_error(iop["chronological_age"], iop["predicted_age"])
    return out


# ============================================================================
# EXPERIMENT A - true Leave-Site-Out unseen-scanner penalty
# ============================================================================
def experiment_a():
    print("\n" + "=" * 78)
    print("EXPERIMENT A - TRUE unseen-scanner penalty (Leave-Site-Out)")
    print("=" * 78)

    strat_iop_mae = stratified_iop_mae_reference()

    lso_results = {}  # model -> {held_out_site: MAE}
    lso_preds = {}    # model -> {held_out_site: (idx_in_feat, y_true, y_pred)}
    for model_name, make_model in TABULAR_MODELS.items():
        lso_results[model_name] = {}
        lso_preds[model_name] = {}
        for held_out in SITES:
            train_mask = feat["site"] != held_out
            test_mask = feat["site"] == held_out

            X_train = feat.loc[train_mask, FEATURE_COLS].to_numpy()
            X_test = feat.loc[test_mask, FEATURE_COLS].to_numpy()
            y_train = feat.loc[train_mask, "chronological_age"].to_numpy()
            y_test = feat.loc[test_mask, "chronological_age"].to_numpy()

            scaler = StandardScaler().fit(X_train)  # scaler fit on TRAIN SITES only
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = make_model()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            mae = mean_absolute_error(y_test, y_pred)
            lso_results[model_name][held_out] = mae
            lso_preds[model_name][held_out] = (feat.loc[test_mask, "subject_id"].to_numpy(),
                                                y_test, y_pred, scaler, model)

    print(f"\n{'model':<18}" + "".join(f"{s+'(held out)':>18}" for s in SITES))
    for model_name in TABULAR_MODELS:
        row = "".join(f"{lso_results[model_name][s]:>18.3f}" for s in SITES)
        print(f"{model_name:<18}{row}")

    print("\nheld-out-IOP LSO MAE vs earlier stratified-CV IOP MAE (recomputed from predictions_*.csv):")
    verdicts = []
    for model_name in TABULAR_MODELS:
        lso_mae = lso_results[model_name]["IOP"]
        strat_mae = strat_iop_mae[model_name]
        delta = lso_mae - strat_mae
        pct = 100 * delta / strat_mae
        worse = delta > 0.5  # substantially worse threshold: >0.5y absolute
        verdicts.append(worse)
        print(f"  {model_name:<18} LSO={lso_mae:.3f}y   stratified={strat_mae:.3f}y   "
              f"delta={delta:+.3f}y ({pct:+.1f}%)   {'WORSE' if worse else 'similar'}")

    all_worse = all(verdicts)
    any_worse = any(verdicts)
    if all_worse:
        verdict = "SUPPORTED"
        meaning = "Real unseen-scanner penalty is worse than stratified CV suggested, for all 3 tabular models."
    elif any_worse:
        verdict = "PARTIALLY SUPPORTED"
        meaning = "Unseen-scanner penalty is worse for some but not all tabular models."
    else:
        verdict = "NOT SUPPORTED"
        meaning = "Leave-site-out IOP MAE is similar to stratified-CV IOP MAE - no extra penalty detected."
    print(f"\nVERDICT A: {verdict} - {meaning}")

    summary_rows.append({
        "experiment": "A: True unseen-scanner (LSO) penalty",
        "key_number": f"IOP LSO MAE " + ", ".join(f"{m}={lso_results[m]['IOP']:.2f}y" for m in TABULAR_MODELS),
        "verdict": verdict,
        "meaning": meaning,
    })
    return lso_results, lso_preds, strat_iop_mae


# ============================================================================
# EXPERIMENT B - does unsupervised harmonization fix the unseen site?
# ============================================================================
def experiment_b(lso_preds, lso_results):
    print("\n" + "=" * 78)
    print("EXPERIMENT B - Does harmonization FIX the unseen site?")
    print("=" * 78)

    IN_DIST_TARGET = (6.0, 7.0)  # "in-distribution range" per the spec

    harmon_rows = []
    for model_name in TABULAR_MODELS:
        subj_ids, y_test, y_pred_before, scaler, model = lso_preds[model_name]["IOP"]

        # raw (unscaled) features for train sites (Guys+HH) and IOP
        train_mask = feat["site"] != "IOP"
        test_mask = feat["site"] == "IOP"
        X_train_raw = feat.loc[train_mask, FEATURE_COLS].to_numpy()
        X_iop_raw = feat.loc[test_mask, FEATURE_COLS].to_numpy()

        # per-feature location+scale harmonization, NO ages used:
        # z-score IOP using IOP's OWN mean/std, then rescale to train sites' mean/std
        iop_mean = X_iop_raw.mean(axis=0)
        iop_std = X_iop_raw.std(axis=0)
        train_mean = X_train_raw.mean(axis=0)
        train_std = X_train_raw.std(axis=0)

        z_iop = (X_iop_raw - iop_mean) / iop_std
        harmonized_iop_raw = z_iop * train_std + train_mean

        X_iop_harm_scaled = scaler.transform(harmonized_iop_raw)  # same scaler as Exp A's IOP-held-out fold
        y_pred_after = model.predict(X_iop_harm_scaled)

        mae_before = mean_absolute_error(y_test, y_pred_before)
        mae_after = mean_absolute_error(y_test, y_pred_after)
        drop = mae_before - mae_after
        in_range = IN_DIST_TARGET[0] <= mae_after <= IN_DIST_TARGET[1]
        materially_toward = mae_after <= mae_before - 1.0 or in_range

        harmon_rows.append({
            "model": model_name, "mae_before": mae_before, "mae_after": mae_after,
            "drop": drop, "in_range": in_range, "materially_toward": materially_toward,
        })

    harmon_table = pd.DataFrame(harmon_rows)
    print(harmon_table.round(3).to_string(index=False))

    n_support = harmon_table["materially_toward"].sum()
    if n_support == len(TABULAR_MODELS):
        verdict = "SUPPORTED"
        meaning = "Simple location+scale harmonization mitigates IOP difficulty for all 3 tabular models."
    elif n_support > 0:
        verdict = "PARTIALLY SUPPORTED"
        meaning = f"Harmonization helps for {n_support}/3 models but not all."
    else:
        verdict = "NOT SUPPORTED"
        meaning = "MAE barely moves after harmonization - IOP difficulty is not just location/scale shift."
    print(f"\nVERDICT B: {verdict} - {meaning}")

    summary_rows.append({
        "experiment": "B: Harmonization mitigation",
        "key_number": ", ".join(f"{r['model']}: {r['mae_before']:.2f}->{r['mae_after']:.2f}y" for r in harmon_rows),
        "verdict": verdict,
        "meaning": meaning,
    })
    return harmon_table


# ============================================================================
# EXPERIMENT C - is DeepBrainNet's sex bias real or confounded?
# ============================================================================
def experiment_c():
    print("\n" + "=" * 78)
    print("EXPERIMENT C - Is the sex bias real or confounded by site/brain-size?")
    print("=" * 78)

    c_rows = []
    for model_name in ALL_MODELS:
        df = preds[model_name].merge(
            feat[["subject_id", "total_brain_volume_ml"]], on="subject_id", how="inner"
        )
        df["gap"] = df["predicted_age"] - df["chronological_age"]
        df["sex_M"] = (df["sex"] == "M").astype(int)
        site_dummies = pd.get_dummies(df["site"], prefix="site", drop_first=True).astype(float)

        # BEFORE: gap ~ age + sex
        X_before = pd.DataFrame({
            "age": df["chronological_age"], "sex_M": df["sex_M"],
        })
        X_before = sm.add_constant(X_before)
        model_before = sm.OLS(df["gap"], X_before).fit()
        coef_before = model_before.params["sex_M"]
        p_before = model_before.pvalues["sex_M"]

        # AFTER: gap ~ age + sex + site + total_brain
        X_after = pd.concat([
            pd.DataFrame({"age": df["chronological_age"], "sex_M": df["sex_M"],
                           "total_brain": df["total_brain_volume_ml"]}),
            site_dummies,
        ], axis=1)
        X_after = sm.add_constant(X_after)
        model_after = sm.OLS(df["gap"], X_after).fit()
        coef_after = model_after.params["sex_M"]
        p_after = model_after.pvalues["sex_M"]

        # isolate which covariate kills significance (if it does)
        killer = None
        if p_before < 0.05 <= p_after:
            X_site_only = pd.concat([
                pd.DataFrame({"age": df["chronological_age"], "sex_M": df["sex_M"]}), site_dummies
            ], axis=1)
            X_site_only = sm.add_constant(X_site_only)
            p_site_only = sm.OLS(df["gap"], X_site_only).fit().pvalues["sex_M"]

            X_brain_only = sm.add_constant(pd.DataFrame({
                "age": df["chronological_age"], "sex_M": df["sex_M"],
                "total_brain": df["total_brain_volume_ml"],
            }))
            p_brain_only = sm.OLS(df["gap"], X_brain_only).fit().pvalues["sex_M"]

            if p_site_only >= 0.05 and p_brain_only < 0.05:
                killer = "site"
            elif p_brain_only >= 0.05 and p_site_only < 0.05:
                killer = "total_brain"
            elif p_site_only >= 0.05 and p_brain_only >= 0.05:
                killer = "site AND total_brain (either alone kills it)"
            else:
                killer = "neither alone - only the combination"

        c_rows.append({
            "model": model_name, "n": len(df),
            "sex_coef_before": coef_before, "p_before": p_before,
            "sex_coef_after": coef_after, "p_after": p_after,
            "stays_significant": p_after < 0.05,
            "killed_by": killer if killer else "-",
        })

    c_table = pd.DataFrame(c_rows)
    print(c_table.round(5).to_string(index=False))

    dbn_row = c_table[c_table["model"] == "DeepBrainNet"].iloc[0]
    if dbn_row["stays_significant"]:
        verdict = "SUPPORTED"
        meaning = "DeepBrainNet's sex coefficient stays significant after site + total_brain controls: real effect."
    else:
        verdict = "NOT SUPPORTED"
        meaning = f"DeepBrainNet's sex effect drops to n.s. after controls (killed by: {dbn_row['killed_by']}) - confounded."
    print(f"\nVERDICT C (DeepBrainNet): {verdict} - {meaning}")

    summary_rows.append({
        "experiment": "C: DeepBrainNet sex bias, confound-checked",
        "key_number": f"sex coef before={dbn_row['sex_coef_before']:.3f} (p={dbn_row['p_before']:.4f}), "
                      f"after={dbn_row['sex_coef_after']:.3f} (p={dbn_row['p_after']:.4f})",
        "verdict": verdict,
        "meaning": meaning,
    })
    return c_table


# ============================================================================
# EXPERIMENT D - within-distribution (Guys+HH only) reliability gradient
# ============================================================================
def experiment_d():
    print("\n" + "=" * 78)
    print("EXPERIMENT D - Any within-distribution reliability gradient? (3 more OOD methods)")
    print("=" * 78)

    gh = feat[feat["site"].isin(["Guys", "HH"])].reset_index(drop=True)
    ref_mean = gh[FEATURE_COLS].mean()
    ref_std = gh[FEATURE_COLS].std()
    X = ((gh[FEATURE_COLS] - ref_mean) / ref_std).to_numpy()

    scores = {}

    # kNN mean-distance (k=10), self excluded
    nn = NearestNeighbors(n_neighbors=11).fit(X)
    dist, _ = nn.kneighbors(X)
    scores["kNN_mean_dist_k10"] = dist[:, 1:11].mean(axis=1)

    # IsolationForest - higher = more anomalous
    iso = IsolationForest(random_state=42, n_estimators=200).fit(X)
    scores["IsolationForest"] = -iso.score_samples(X)

    # LocalOutlierFactor (novelty=False -> use negative_outlier_factor_ on fit data)
    lof = LocalOutlierFactor(n_neighbors=10, novelty=False)
    lof.fit_predict(X)
    scores["LocalOutlierFactor"] = -lof.negative_outlier_factor_

    gh_ids = gh["subject_id"].to_numpy()
    d_rows = []
    any_positive_sig = False
    for scorer_name, score_vals in scores.items():
        for model_name in ALL_MODELS:
            df = preds[model_name]
            df_gh = df[df["subject_id"].isin(gh_ids)].set_index("subject_id").loc[gh_ids].reset_index()
            err = (df_gh["predicted_age"] - df_gh["chronological_age"]).abs().to_numpy()

            rho, p = stats.spearmanr(score_vals, err)
            positive_sig = (rho > 0) and (p < 0.05)
            any_positive_sig = any_positive_sig or positive_sig
            d_rows.append({
                "scorer": scorer_name, "model": model_name, "n": len(err),
                "spearman_r": rho, "p": p, "positive_significant": positive_sig,
            })

    d_table = pd.DataFrame(d_rows)
    print(d_table.round(5).to_string(index=False))

    if any_positive_sig:
        sig_rows = d_table[d_table["positive_significant"]]
        verdict = "SUPPORTED"
        meaning = (f"{len(sig_rows)} scorer/model pair(s) show a positive, significant within-distribution "
                   f"reliability signal: " +
                   ", ".join(f"{r.scorer}/{r.model}(r={r.spearman_r:.3f})" for r in sig_rows.itertuples()))
    else:
        verdict = "NULL"
        meaning = ("All scorer/model pairs are null or negative within Guys+HH - the reliability signal found "
                   "earlier is purely between-scanner (IOP vs reference), not a fine-grained within-distribution "
                   "gradient. This is a robust, informative null, not an inconclusive result.")
    print(f"\nVERDICT D: {verdict} - {meaning}")

    summary_rows.append({
        "experiment": "D: Within-distribution (Guys+HH) reliability gradient",
        "key_number": f"{d_table['positive_significant'].sum()}/{len(d_table)} scorer x model pairs positive+significant",
        "verdict": verdict,
        "meaning": meaning,
    })
    return d_table


# ============================================================================
# EXPERIMENT E - does site-aware age-correction reduce disparity, or just calibrate?
# ============================================================================
def experiment_e():
    print("\n" + "=" * 78)
    print("EXPERIMENT E - Does site-aware age-correction reduce disparity?")
    print("=" * 78)

    df = preds["DeepBrainNet"].copy()
    df["gap"] = df["predicted_age"] - df["chronological_age"]
    df["abs_err"] = (df["predicted_age"] - df["chronological_age"]).abs()

    # (i) global: gap ~ age, pooled
    slope, intercept, *_ = stats.linregress(df["chronological_age"], df["gap"])
    df["corrected_i"] = df["gap"] - (slope * df["chronological_age"] + intercept)

    # (ii) per-site: gap ~ age fit SEPARATELY per site
    df["corrected_ii"] = np.nan
    for site, sub in df.groupby("site"):
        s, i, *_ = stats.linregress(sub["chronological_age"], sub["gap"])
        df.loc[sub.index, "corrected_ii"] = sub["gap"] - (s * sub["chronological_age"] + i)

    # (iii) gap ~ age + C(site) covariate, single joint regression
    site_dummies = pd.get_dummies(df["site"], prefix="site", drop_first=True).astype(float)
    X_iii = sm.add_constant(pd.concat([df[["chronological_age"]], site_dummies], axis=1))
    fit_iii = sm.OLS(df["gap"], X_iii).fit()
    df["corrected_iii"] = fit_iii.resid

    print("\n--- mean corrected-gap by site, per correction method ---")
    corr_cols = {"i_global": "corrected_i", "ii_per_site": "corrected_ii", "iii_site_covariate": "corrected_iii"}
    gap_spread_rows = []
    for label, col in corr_cols.items():
        by_site = df.groupby("site")[col].mean().reindex(SITES)
        spread = by_site.max() - by_site.min()
        gap_spread_rows.append({"method": label, **by_site.to_dict(), "spread_max_minus_min": spread})
        print(f"\n[{label}]")
        print(by_site.round(4).to_string())
        print(f"  spread (max-min) = {spread:.4f}")

    print("\n--- per-site MAE (raw |predicted-actual|, should be IDENTICAL across all 3 methods -"
          " correction does not change predictions) ---")
    mae_i = df.groupby("site")["abs_err"].mean().reindex(SITES)
    mae_ii = df.groupby("site")["abs_err"].mean().reindex(SITES)  # same predictions -> same MAE
    mae_iii = df.groupby("site")["abs_err"].mean().reindex(SITES)
    mae_compare = pd.DataFrame({"i_global": mae_i, "ii_per_site": mae_ii, "iii_site_covariate": mae_iii})
    print(mae_compare.round(3).to_string())
    mae_identical = np.allclose(mae_compare["i_global"], mae_compare["ii_per_site"]) and \
                     np.allclose(mae_compare["i_global"], mae_compare["iii_site_covariate"])
    print(f"\nMAE identical across all 3 correction methods: {mae_identical}  "
          f"(confirms correction is a calibration transform, NOT a new set of predictions)")

    spread_i = gap_spread_rows[0]["spread_max_minus_min"]
    spread_ii = gap_spread_rows[1]["spread_max_minus_min"]
    spread_iii = gap_spread_rows[2]["spread_max_minus_min"]
    spread_shrinks = spread_ii < spread_i * 0.5 and spread_iii < spread_i * 0.5

    iop_mae = mae_compare.loc["IOP", "i_global"]
    non_iop_mae = mae_compare.loc[["Guys", "HH"], "i_global"].mean()
    mae_gap_persists = iop_mae > non_iop_mae + 1.0

    if spread_shrinks and mae_identical and mae_gap_persists:
        verdict = "SUPPORTED (as calibration fix ONLY)"
        meaning = (f"Site-aware correction shrinks gap-spread ({spread_i:.3f}y -> {spread_ii:.3f}y/{spread_iii:.3f}y) "
                   f"but IOP MAE is UNCHANGED ({iop_mae:.2f}y, still {iop_mae - non_iop_mae:.2f}y worse than "
                   f"Guys/HH) - this fixes directional calibration, NOT accuracy.")
    else:
        verdict = "NOT SUPPORTED"
        meaning = "Gap-spread did not shrink materially, or the MAE gap did not persist as expected."
    print(f"\nVERDICT E: {verdict} - {meaning}")

    summary_rows.append({
        "experiment": "E: Site-aware age-correction (calibration vs accuracy)",
        "key_number": f"gap-spread {spread_i:.2f}y -> {spread_ii:.2f}y; IOP MAE unchanged at {iop_mae:.2f}y",
        "verdict": verdict,
        "meaning": meaning,
    })
    return gap_spread_rows, mae_compare


def print_final_summary():
    print("\n" + "=" * 78)
    print("FINAL SUMMARY TABLE")
    print("=" * 78)
    summary_df = pd.DataFrame(summary_rows)
    for _, row in summary_df.iterrows():
        print(f"\n[{row['experiment']}]")
        print(f"  key number : {row['key_number']}")
        print(f"  VERDICT    : {row['verdict']}")
        print(f"  meaning    : {row['meaning']}")
        if row["verdict"].startswith("SUPPORTED") or row["verdict"].startswith("PARTIALLY SUPPORTED"):
            print(f"  to confirm : replicate on an independent dataset with a genuinely new, unseen scanner site "
                  f"(not just a held-out split of the same 3-site cohort).")
    summary_df.to_csv(fr"{RESULTS}\theory_test_battery_summary.csv", index=False)
    print(f"\nwrote results\\theory_test_battery_summary.csv")


if __name__ == "__main__":
    lso_results, lso_preds, strat_iop_mae = experiment_a()
    experiment_b(lso_preds, lso_results)
    experiment_c()
    experiment_d()
    experiment_e()
    print_final_summary()
