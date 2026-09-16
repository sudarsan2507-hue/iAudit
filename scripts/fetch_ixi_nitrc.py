"""Fetch raw IXI T1 scans from the NITRC XNAT mirror.

The original Imperial host (biomedic.doc.ic.ac.uk) returns 403 for everyone as of
2026-09. NITRC's XNAT instance mirrors the same RAW NIfTI files under their
original IXI filenames and allows anonymous REST access.

Downloads a fixed random sample of N subjects so the pilot is reproducible.
"""

import argparse
import json
import random
import socket
import sys
import time
import urllib.request
from pathlib import Path

# Per-call timeout= on urlopen doesn't always get honored for a stalled
# connection (DNS/TCP-level hangs have been observed on this network hitting
# NITRC, where the process sits at 0% CPU indefinitely). A hard default
# socket timeout catches those too.
socket.setdefaulttimeout(120)

BASE = "https://www.nitrc.org/ir/data"
UA = {"User-Agent": "Mozilla/5.0"}
OUT_DIR = Path(r"E:\iAuditData\t1")


def get_json(url, key="ResultSet", attempts=5):
    """GET json, retrying on throttling.

    NITRC intermittently answers rapid anonymous requests with a padded error
    page instead of the expected document, so validate the shape and retry.
    """
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            if key in data:
                return data
            last = f"missing {key!r} (throttled?)"
        except Exception as e:
            last = repr(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url} -> {last}")


def download(url, dest):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of subjects to fetch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # One call for every session in the project. The per-subject
    # /subjects/{id}/experiments endpoint is unreliable on this XNAT instance -
    # for some subjects it returns the full subject document instead of a
    # ResultSet - so go through the project-scoped listing instead.
    exps = get_json(f"{BASE}/projects/ixi/experiments?format=json")["ResultSet"]["Result"]
    print(f"IXI sessions on NITRC XNAT: {len(exps)}")

    random.Random(args.seed).shuffle(exps)

    got = 0
    for subj in exps:
        if got >= args.n:
            break
        exp = subj["ID"]
        try:
            files = get_json(f"{BASE}/experiments/{exp}/scans/T1/files?format=json")["ResultSet"]["Result"]
        except Exception as e:  # no T1 scan, or transient API error
            print(f"  skip {subj['label']}: {e}")
            continue

        nii = [f for f in files if f["Name"].endswith(".nii.gz") and f.get("collection") == "NIfTI"]
        if not nii:
            print(f"  skip {subj['label']}: no NIfTI T1")
            continue

        f = nii[0]
        dest = OUT_DIR / f["Name"]
        if dest.exists() and dest.stat().st_size == int(f["Size"]):
            print(f"  have {f['Name']}")
            got += 1
            continue
        try:
            size = download(f"https://www.nitrc.org/ir{f['URI']}", dest)
        except Exception as e:
            print(f"  FAIL {f['Name']}: {e}")
            continue
        print(f"  got  {f['Name']}  ({size/1e6:.1f} MB)")
        got += 1
        time.sleep(0.5)  # be polite to NITRC

    print(f"\nDownloaded/present: {got} T1 scans in {OUT_DIR}")
    if got < args.n:
        print(f"WARNING: wanted {args.n}, got {got}", file=sys.stderr)


if __name__ == "__main__":
    main()
