"""Stage 3 - pilot batch. Runs DeepBrainNet over the downloaded T1s and writes
results/predictions_deepbrainnet.csv.

CSV schema is fixed (the fairness analysis depends on it):
    subject_id, chronological_age, predicted_age, sex, site, ethnicity, model_name

Each scan takes ~5 min on CPU, so results are checkpointed to a JSONL file after
every scan. Re-running resumes from where it stopped instead of redoing work.
"""

import os

# See predict_one.py: DeepBrainNet is a legacy Keras 2 model and will not load
# under Keras 3. Must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import time
import traceback
from pathlib import Path

import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from antspynet.utilities import brain_age
from scipy.stats import pearsonr

from load_demographics import build_cohort

RESULTS = Path(r"D:\Projects\iAuditAge\results")
CSV_PATH = RESULTS / "predictions_deepbrainnet.csv"
FIG_PATH = RESULTS / "fig_pred_vs_actual.png"
CKPT_PATH = RESULTS / "_partial_predictions.jsonl"
COLUMNS = [
    "subject_id",
    "chronological_age",
    "predicted_age",
    "sex",
    "site",
    "ethnicity",
    "model_name",
]
MAX_SCANS = 20  # hard cap for this pilot


def load_checkpoint():
    """Return (rows, failed_ids) already recorded by a previous run."""
    rows, failed = [], {}
    if not CKPT_PATH.exists():
        return rows, failed
    for line in CKPT_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_failed"):
            failed[rec["subject_id"]] = rec["_error"]
        else:
            rows.append(rec)
    return rows, failed


def append_checkpoint(rec):
    with open(CKPT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=MAX_SCANS)
    ap.add_argument("--restart", action="store_true", help="discard checkpoint")
    args = ap.parse_args()
    limit = min(args.limit, MAX_SCANS)

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.restart and CKPT_PATH.exists():
        CKPT_PATH.unlink()

    cohort, _ = build_cohort()
    cohort = cohort.head(limit)

    rows, failures = load_checkpoint()
    done = {r["subject_id"] for r in rows} | set(failures)
    if done:
        print(f"resuming: {len(rows)} done, {len(failures)} failed, "
              f"{len(cohort) - len(done)} remaining")
    print(f"running brain_age on {len(cohort)} scans (cap {MAX_SCANS})\n")

    for i, r in enumerate(cohort.itertuples(index=False), 1):
        if r.subject_id in done:
            print(f"[{i}/{len(cohort)}] {r.subject_id}: already done, skipping")
            continue
        t0 = time.time()
        try:
            t1 = ants.image_read(r.filepath)
            predicted = float(brain_age(t1, do_preprocessing=True, verbose=False)["predicted_age"])
            rec = {
                "subject_id": r.subject_id,
                "chronological_age": float(r.chronological_age),
                "predicted_age": predicted,
                "sex": r.sex,
                "site": r.site,
                "ethnicity": r.ethnicity,
                "model_name": "DeepBrainNet",
            }
            rows.append(rec)
            append_checkpoint(rec)
            print(
                f"[{i}/{len(cohort)}] {r.subject_id}: pred={predicted:6.2f}  "
                f"actual={r.chronological_age:6.2f}  ({time.time() - t0:.0f}s)"
            )
        except Exception as e:
            failures[r.subject_id] = repr(e)
            append_checkpoint({"subject_id": r.subject_id, "_failed": True, "_error": repr(e)})
            print(f"[{i}/{len(cohort)}] {r.subject_id}: FAILED {e!r}")
            traceback.print_exc()

    if not rows:
        raise SystemExit("no scans succeeded - nothing to write")

    df = pd.DataFrame(rows)[COLUMNS].sort_values("subject_id").reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nwrote {CSV_PATH}  ({len(df)} rows)")

    # ---------------- VERIFY 3 ----------------
    print("\n" + "=" * 60)
    print("VERIFY 3 - PILOT GATE")
    print("=" * 60)
    print(f"n_success = {len(rows)}")
    print(f"n_failed  = {len(failures)}")
    for sid, err in failures.items():
        print(f"    {sid}: {err}")

    assert not df["predicted_age"].isna().any(), "NaNs present in predicted_age"
    print("assert no NaNs in predicted_age: PASS")

    r_val, p_val = pearsonr(df["predicted_age"], df["chronological_age"])
    errors = (df["predicted_age"] - df["chronological_age"]).abs()
    mae = errors.mean()
    bias = (df["predicted_age"] - df["chronological_age"]).mean()

    print(f"\nPearson r = {r_val:.3f}  (p = {p_val:.2e})")
    print(f"MAE       = {mae:.2f} years")
    print(f"bias      = {bias:+.2f} years (mean predicted - actual)")

    # scatter + identity line
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["chronological_age"], df["predicted_age"], s=45,
               c="#2b6cb0", edgecolor="white", zorder=3)
    lo = min(df["chronological_age"].min(), df["predicted_age"].min()) - 5
    hi = max(df["chronological_age"].max(), df["predicted_age"].max()) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="#e53e3e", lw=1.5, zorder=2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("chronological age (years)")
    ax.set_ylabel("predicted brain age (years)")
    ax.set_title(f"DeepBrainNet on IXI pilot (n={len(df)})\nr={r_val:.3f}, MAE={mae:.2f}y")
    ax.grid(alpha=0.3, zorder=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"saved {FIG_PATH}")

    pass_r = r_val > 0.6
    pass_mae = mae < 8
    print("\nPASS CRITERIA")
    print(f"  r > 0.6        : {'PASS' if pass_r else 'FAIL'}  (r = {r_val:.3f})")
    print(f"  MAE < 8 years  : {'PASS' if pass_mae else 'FAIL'}  (MAE = {mae:.2f})")

    if pass_r and pass_mae:
        print("\nPILOT GATE PASSED - safe to scale to full IXI.")
    else:
        print("\nPILOT GATE FAILED")
        print("3 most likely causes, and the check for each:")
        print("  1. Bad preprocessing / registration.")
        print("     CHECK: rerun predict_one.py with verbose=True and confirm the")
        print("            template registration step completes; view the")
        print("            preprocessed volume for a cropped or flipped brain.")
        print("  2. Filename -> demographics mismatch (wrong IXI_ID join).")
        print("     CHECK: scripts/verify_join.py compares every joined age against")
        print("            the NITRC XNAT session 'age' field; they must agree.")
        print("  3. Wrong SEX/age column picked out of IXI.xls.")
        print("     CHECK: rerun load_demographics.py and confirm the AGE column is")
        print("            years (~20-86), not DOB or a date serial.")


if __name__ == "__main__":
    main()
