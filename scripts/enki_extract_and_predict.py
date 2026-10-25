"""Stage R2 (part 2) - run extract_features.extract_one and predict_one.predict
(both UNCHANGED) on every downloaded eNKI T1. Checkpointed.
"""
import json
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import extract_one  # UNCHANGED
from predict_one import predict  # UNCHANGED (DeepBrainNet)

RESULTS = Path(r"D:\Projects\iAuditAge\results")
FETCH_CKPT = RESULTS / "_enki_fetch_checkpoint.jsonl"
CKPT_PATH = RESULTS / "_enki_features_predictions.jsonl"
OUT_CSV = RESULTS / "enki_features_predictions.csv"


def load_fetch_list():
    recs = []
    for line in FETCH_CKPT.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not rec.get("_failed"):
            recs.append(rec)
    return recs


def load_checkpoint():
    rows, failed = [], {}
    if not CKPT_PATH.exists():
        return rows, failed
    for line in CKPT_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_failed"):
            failed[rec["participant_id"]] = rec["_error"]
        else:
            rows.append(rec)
    return rows, failed


def append_checkpoint(rec):
    with open(CKPT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    scans = load_fetch_list()
    print(f"target: {len(scans)} eNKI T1 scans with valid downloads")

    rows, failures = load_checkpoint()
    done = {r["participant_id"] for r in rows} | set(failures)
    if done:
        print(f"resuming: {len(rows)} done, {len(failures)} failed, {len(scans) - len(done)} remaining")

    for i, s in enumerate(scans, 1):
        pid = s["participant_id"]
        if pid in done:
            continue
        t0 = time.time()
        try:
            feats = extract_one(s["filepath"])
            predicted_age = predict(s["filepath"], verbose=False)
            rec = {
                "participant_id": pid, "vendor": s["vendor"], "model": s.get("model"),
                "chronological_age": s["age"], "sex": s["sex"],
                "predicted_age": predicted_age, **feats,
            }
            rows.append(rec)
            append_checkpoint(rec)
            print(f"[{i}/{len(scans)}] {pid} ({s['vendor']}): age={s['age']:.0f} "
                  f"pred={predicted_age:.1f} total_brain={feats['total_brain_volume_ml']:.0f}mL "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            failures[pid] = repr(e)
            append_checkpoint({"participant_id": pid, "_failed": True, "_error": repr(e)})
            print(f"[{i}/{len(scans)}] {pid}: FAILED {e!r}")
            traceback.print_exc()

    if not rows:
        raise SystemExit("no features extracted - nothing to write")

    df = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")
    print(f"n failed = {len(failures)}")
    for pid, err in failures.items():
        print(f"    {pid}: {err}")


if __name__ == "__main__":
    main()
