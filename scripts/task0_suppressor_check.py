"""TASK 0 - Experiment C follow-up: which covariate is the suppressor that
makes DeepBrainNet's sex coefficient grow once controlled for?
"""
import pandas as pd
import statsmodels.api as sm

RESULTS = r"D:\Projects\iAuditAge\results"

feat = pd.read_csv(fr"{RESULTS}\features.csv")
dbn = pd.read_csv(fr"{RESULTS}\predictions_deepbrainnet.csv")

df = dbn.merge(feat[["subject_id", "total_brain_volume_ml"]], on="subject_id", how="inner")
df["gap"] = df["predicted_age"] - df["chronological_age"]
df["sex_M"] = (df["sex"] == "M").astype(int)
site_dummies = pd.get_dummies(df["site"], prefix="site", drop_first=True).astype(float)

print("=" * 78)
print("TASK 0 - Experiment C follow-up: incremental sex-coefficient suppressor check")
print(f"(DeepBrainNet, n={len(df)})")
print("=" * 78)

steps = {}

# (i) + C(sex) only
X_i = sm.add_constant(pd.DataFrame({"age": df["chronological_age"], "sex_M": df["sex_M"]}))
fit_i = sm.OLS(df["gap"], X_i).fit()
steps["(i) age + sex"] = fit_i

# (ii) + C(sex) + total_brain
X_ii = sm.add_constant(pd.DataFrame({
    "age": df["chronological_age"], "sex_M": df["sex_M"], "total_brain": df["total_brain_volume_ml"],
}))
fit_ii = sm.OLS(df["gap"], X_ii).fit()
steps["(ii) age + sex + total_brain"] = fit_ii

# (iii) + C(sex) + C(site)
X_iii = sm.add_constant(pd.concat([
    pd.DataFrame({"age": df["chronological_age"], "sex_M": df["sex_M"]}), site_dummies], axis=1))
fit_iii = sm.OLS(df["gap"], X_iii).fit()
steps["(iii) age + sex + site"] = fit_iii

# (iv) + C(sex) + total_brain + C(site)
X_iv = sm.add_constant(pd.concat([
    pd.DataFrame({"age": df["chronological_age"], "sex_M": df["sex_M"],
                  "total_brain": df["total_brain_volume_ml"]}), site_dummies], axis=1))
fit_iv = sm.OLS(df["gap"], X_iv).fit()
steps["(iv) age + sex + total_brain + site"] = fit_iv

print(f"\n{'step':<38}{'sex_coef':>12}{'p':>12}")
for label, fit in steps.items():
    print(f"{label:<38}{fit.params['sex_M']:>12.4f}{fit.pvalues['sex_M']:>12.6f}")

coef_i = fit_i.params["sex_M"]
coef_ii = fit_ii.params["sex_M"]
coef_iii = fit_iii.params["sex_M"]
coef_iv = fit_iv.params["sex_M"]

grown_by_brain = coef_ii > coef_i * 1.1  # brain-size control grows it
grown_by_site = coef_iii > coef_i * 1.1  # site control grows it

print("\n--- suppressor identification ---")
print(f"baseline (i) sex_coef = {coef_i:.4f}")
print(f"adding total_brain alone (ii): {coef_ii:.4f}  (grew: {grown_by_brain})")
print(f"adding site alone (iii): {coef_iii:.4f}  (grew: {grown_by_site})")
print(f"adding both (iv): {coef_iv:.4f}")

if grown_by_brain and not grown_by_site:
    suppressor = "total_brain"
    verdict_line = ("VERDICT C-followup: suppressor = total_brain. Sex effect was masked by "
                     "brain-size correlation (males have larger brains on average, and total_brain "
                     "correlates with the gap in a way that offsets the raw sex effect); controlling "
                     "for total_brain unmasks the true sex effect. Clean mechanistic claim.")
elif grown_by_site and not grown_by_brain:
    suppressor = "site"
    verdict_line = ("VERDICT C-followup: suppressor = site. The sex effect is tangled up with site "
                     "composition (uneven sex ratios across sites with different biases) - treat this "
                     "mechanistic claim CAUTIOUSLY, it is less clean than a pure brain-size story.")
elif grown_by_brain and grown_by_site:
    suppressor = "both total_brain and site"
    verdict_line = ("VERDICT C-followup: BOTH total_brain and site act as suppressors individually. "
                     "The story is not attributable to a single clean mechanism - treat cautiously and "
                     "report both.")
else:
    suppressor = "neither individually"
    verdict_line = ("VERDICT C-followup: neither total_brain nor site alone materially grows the "
                     "coefficient - the earlier growth (Experiment C) required the FULL joint model. "
                     "No single clean suppressor identified.")

print(f"\n{verdict_line}")
