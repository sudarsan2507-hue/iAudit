"""Stage R1 (part 2) - run the IDENTICAL deep_atropos feature-extraction
function used for IXI (extract_features.extract_one, byte-for-byte unchanged)
on every THP T1w scan. Checkpointed like the IXI run.
"""
import json
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import extract_one  # UNCHANGED function

T1_DIR = Path(r"E:\iAuditData\thp")
RESULTS = Path(r"D:\Projects\iAuditAge\results")
CKPT_PATH = RESULTS / "_thp_features.jsonl"
OUT_CSV = RESULTS / "thp_features.csv"
VENDOR_META_PATH = RESULTS / "_thp_vendor_metadata.json"

FNAME_RE = re.compile(r"^(sub-THP\d+)_(ses-[A-Za-z0-9]+)_run-(\d+)_T1w\.nii\.gz$")


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
            failed[rec["scan_id"]] = rec["_error"]
        else:
            rows.append(rec)
    return rows, failed


def append_checkpoint(rec):
    with open(CKPT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    vendor_meta = {r["nii_path"].split("/")[-1]: r for r in json.load(open(VENDOR_META_PATH))}

    files = sorted(T1_DIR.glob("*.nii.gz"))
    scans = []
    for p in files:
        m = FNAME_RE.match(p.name)
        if not m:
            print(f"WARNING: unparseable filename, skipping: {p.name}")
            continue
        subject, session, run = m.groups()
        meta = vendor_meta.get(p.name)
        vendor = meta["manufacturer"] if meta else None
        scans.append({"scan_id": p.name, "subject": subject, "session": session,
                       "run": run, "vendor": vendor, "filepath": str(p)})

    print(f"target: {len(scans)} THP T1w scans")

    rows, failures = load_checkpoint()
    done = {r["scan_id"] for r in rows} | set(failures)
    if done:
        print(f"resuming: {len(rows)} done, {len(failures)} failed, {len(scans) - len(done)} remaining")

    for i, s in enumerate(scans, 1):
        if s["scan_id"] in done:
            continue
        t0 = time.time()
        try:
            feats = extract_one(s["filepath"])
            rec = {**{k: s[k] for k in ("scan_id", "subject", "session", "run", "vendor")}, **feats}
            rows.append(rec)
            append_checkpoint(rec)
            print(f"[{i}/{len(scans)}] {s['scan_id']} ({s['vendor']}): "
                  f"total_brain={feats['total_brain_volume_ml']:.0f}mL  ({time.time()-t0:.0f}s)")
        except Exception as e:
            failures[s["scan_id"]] = repr(e)
            append_checkpoint({"scan_id": s["scan_id"], "_failed": True, "_error": repr(e)})
            print(f"[{i}/{len(scans)}] {s['scan_id']}: FAILED {e!r}")
            traceback.print_exc()

    if not rows:
        raise SystemExit("no features extracted - nothing to write")

    df = pd.DataFrame(rows).sort_values(["subject", "session", "run"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")
    print(f"n failed = {len(failures)}")
    for sid, err in failures.items():
        print(f"    {sid}: {err}")


if __name__ == "__main__":
    main()
