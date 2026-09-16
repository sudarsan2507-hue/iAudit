"""Stage 2 - single-scan smoke test. The correctness gate before any batch run.

Runs antspynet's DeepBrainNet brain_age on ONE real T1 and checks the output is
a plausible age. First run downloads model weights to ~/.keras/ANTsXNet/.
"""

import os

# antspynet does `import tensorflow.keras as keras`, but DeepBrainNet ships as a
# legacy Keras 2 HDF5 model that Keras 3 cannot rebuild ("dense_3 expects 1
# input(s), but it received 2"). Route tensorflow.keras to tf_keras instead.
# Must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import math
import sys

import ants
from antspynet.utilities import brain_age

from load_demographics import build_cohort


def predict(filepath, verbose=True):
    t1 = ants.image_read(filepath)
    assert t1.dimension == 3, f"expected a 3D image, got dimension={t1.dimension}"
    print(f"shape   : {t1.shape}")
    print(f"spacing : {t1.spacing}")

    result = brain_age(t1, do_preprocessing=True, verbose=verbose)

    # antspynet returns a dict with per-slice predictions plus the aggregate.
    predicted = float(result["predicted_age"])
    return predicted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", help="subject_id, e.g. IXI251 (default: first)")
    args = ap.parse_args()

    cohort, _ = build_cohort()
    if args.subject:
        row = cohort[cohort["subject_id"] == args.subject]
        if row.empty:
            sys.exit(f"subject {args.subject} not found among downloaded T1s")
        row = row.iloc[0]
    else:
        row = cohort.iloc[0]

    print(f"subject : {row['subject_id']}  ({row['site']})")
    print(f"file    : {row['filepath']}")

    predicted = predict(row["filepath"])
    actual = float(row["chronological_age"])

    print("\n--- VERIFY 2 ---")
    assert math.isfinite(predicted), f"predicted_age is not finite: {predicted}"
    assert 0 < predicted < 120, f"predicted_age out of range: {predicted}"
    print(f"predicted_age      = {predicted:.2f}")
    print(f"chronological_age  = {actual:.2f}")
    print(f"|error|            = {abs(predicted - actual):.2f} years")
    print("assertions passed: finite, 0 < predicted < 120")


if __name__ == "__main__":
    main()
