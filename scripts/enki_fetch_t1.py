"""Stage R2 (part 1) - eNKI (Nathan Kline Institute Rockland Sample), public
FCP-INDI S3 bucket, no DUA / no login required. For each selected participant
(20-86y, capped n=40 - see limitations note in stage_r3_pooled.py for why),
find their earliest T1w BIDS entry, download the .nii.gz + .json sidecar
(vendor confirmed from metadata, never guessed). Checkpointed like THP/IXI.

Single-connection throughput to this bucket was observed at ~30KB/s in this
environment, so downloads run with a small thread pool for parallelism -
listing/manifest-building stays sequential (cheap), only the big .nii.gz
transfers are parallelized.
"""
import json
import os
import subprocess
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "http://fcp-indi.s3.amazonaws.com"
PREFIX = "data/Projects/RocklandSample/RawDataBIDS/"
OUT_DIR = Path(r"E:\iAuditData\enki")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = Path(r"D:\Projects\iAuditAge\results")
CKPT_PATH = RESULTS / "_enki_fetch_checkpoint.jsonl"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
N_WORKERS = 8
ckpt_lock = threading.Lock()


def curl_get(url, out_path=None, max_time=60):
    args = ["curl", "-s", "--max-time", str(max_time), url]
    if out_path:
        args += ["-o", str(out_path)]
        subprocess.run(args, check=True)
        return None
    return subprocess.run(args, check=True, capture_output=True).stdout


def list_keys(prefix):
    keys = {}
    token = None
    while True:
        url = f"{BASE}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(curl_get(url))
        for c in root.findall("s3:Contents", NS):
            keys[c.find("s3:Key", NS).text] = int(c.find("s3:Size", NS).text)
        tok_el = root.find("s3:NextContinuationToken", NS)
        if tok_el is None:
            break
        token = tok_el.text
    return keys


def load_checkpoint():
    done = {}
    if not CKPT_PATH.exists():
        return done
    for line in CKPT_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        done[rec["participant_id"]] = rec
    return done


def append_checkpoint(rec):
    with ckpt_lock:
        with open(CKPT_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


def get_dataset_default_t1w_json():
    # BIDS inheritance principle: a subject with no subject-level T1w.json
    # sidecar inherits the dataset-root T1w.json. ds000206-style per-scan
    # override is the exception here, not the rule - most eNKI subjects have
    # none, so their real vendor is this dataset default, not "unknown".
    return json.loads(curl_get(f"{BASE}/{PREFIX}T1w.json").decode("utf-8"))


def build_manifest(sel, done, default_meta):
    manifest = []
    for i, row in enumerate(sel.itertuples(), 1):
        pid = row.participant_id
        if pid in done:
            continue
        sub = f"sub-{pid}"
        try:
            keys = list_keys(f"{PREFIX}{sub}/")
            t1_niis = sorted(k for k in keys if k.endswith("_T1w.nii.gz"))
            if not t1_niis:
                append_checkpoint({"participant_id": pid, "_failed": True,
                                    "_error": "no T1w.nii.gz found"})
                print(f"[list {i}/{len(sel)}] {pid}: NO T1w FOUND, skipping")
                continue
            nii_key = t1_niis[0]
            json_key = nii_key.replace(".nii.gz", ".json")
            meta = dict(default_meta)
            inherited = True
            if json_key in keys:
                subj_meta = json.loads(curl_get(f"{BASE}/{json_key}").decode("utf-8"))
                if subj_meta.get("Manufacturer"):
                    meta = subj_meta
                    inherited = False
            manifest.append({
                "participant_id": pid, "nii_key": nii_key,
                "expected_size": keys[nii_key],
                "vendor": meta.get("Manufacturer"), "model": meta.get("ManufacturerModelName") or meta.get("ManufacturersModelName"),
                "vendor_inherited": inherited,
                "age": row.age, "sex": row.sex,
            })
            print(f"[list {i}/{len(sel)}] {pid}: manifest ok "
                  f"({keys[nii_key]/1e6:.1f}MB, vendor={meta.get('Manufacturer')}"
                  f"{' [dataset default]' if inherited else ' [subject-specific]'})")
        except Exception as e:
            append_checkpoint({"participant_id": pid, "_failed": True, "_error": repr(e)})
            print(f"[list {i}/{len(sel)}] {pid}: FAILED (listing) {e!r}")
    return manifest


def download_one(item, idx, total):
    pid = item["participant_id"]
    sub = f"sub-{pid}"
    dest = OUT_DIR / f"{sub}_T1w.nii.gz"
    expected_size = item["expected_size"]
    try:
        if not (dest.exists() and dest.stat().st_size == expected_size):
            for attempt in range(3):
                try:
                    curl_get(f"{BASE}/{item['nii_key']}", out_path=dest, max_time=600)
                    if dest.stat().st_size == expected_size:
                        break
                except subprocess.CalledProcessError:
                    pass
                if attempt == 2:
                    raise ValueError(
                        f"download incomplete after 3 attempts: got "
                        f"{dest.stat().st_size if dest.exists() else 0} bytes, "
                        f"expected {expected_size}")
                time.sleep(2)
        rec = {"participant_id": pid, "nii_key": item["nii_key"], "vendor": item["vendor"],
               "model": item["model"], "vendor_inherited": item["vendor_inherited"],
               "filepath": str(dest), "age": item["age"], "sex": item["sex"]}
        append_checkpoint(rec)
        print(f"[dl {idx}/{total}] {pid}: got {dest.name} "
              f"({dest.stat().st_size/1e6:.1f}MB, vendor={item['vendor']})")
    except Exception as e:
        append_checkpoint({"participant_id": pid, "_failed": True, "_error": repr(e)})
        print(f"[dl {idx}/{total}] {pid}: FAILED {e!r}")


def main():
    sel = pd.read_csv(RESULTS / "_enki_selected.csv")
    done = load_checkpoint()
    print(f"target: {len(sel)} eNKI participants (already done: {len(done)})")

    default_meta = get_dataset_default_t1w_json()
    print(f"dataset-root T1w.json default: {default_meta}")
    manifest = build_manifest(sel, done, default_meta)
    print(f"\nmanifest built: {len(manifest)} to download, starting {N_WORKERS}-way parallel fetch\n")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(download_one, item, i, len(manifest))
                for i, item in enumerate(manifest, 1)]
        for f in as_completed(futs):
            f.result()

    done = load_checkpoint()
    ok = [r for r in done.values() if not r.get("_failed")]
    failed = [r for r in done.values() if r.get("_failed")]
    print(f"\ndone: {len(ok)} ok, {len(failed)} failed, of {len(sel)} targeted")


if __name__ == "__main__":
    main()
