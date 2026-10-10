"""Fetch every T1w JSON sidecar from ds000206 (THP) to confirm scanner vendor
per scan from metadata. No guessing - if Manufacturer is missing, that scan is
excluded and reported, not assumed.
"""
import json
import time
import urllib.request

TREE_PATH = r"C:\Users\sudar\AppData\Local\Temp\thp_tree.json"
RAW_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds000206/master/"
OUT_PATH = r"D:\Projects\iAuditAge\results\_thp_vendor_metadata.json"

tree = json.load(open(TREE_PATH))
t1w_json_paths = sorted(t["path"] for t in tree["tree"] if t["path"].endswith("T1w.json"))
print(f"found {len(t1w_json_paths)} T1w.json sidecars")

records = []
for i, path in enumerate(t1w_json_paths, 1):
    url = RAW_BASE + path
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                meta = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:
            if attempt == 2:
                meta = {"_fetch_error": repr(e)}
            time.sleep(1)
    manufacturer = meta.get("Manufacturer", None)
    model = meta.get("ManufacturersModelName", None)
    parts = path.split("/")
    subject = parts[0]
    session = parts[1]
    nii_path = path.replace(".json", ".nii.gz")
    records.append({
        "subject": subject, "session": session, "json_path": path, "nii_path": nii_path,
        "manufacturer": manufacturer, "model": model,
    })
    print(f"[{i}/{len(t1w_json_paths)}] {subject}/{session}: Manufacturer={manufacturer!r} Model={model!r}")

with open(OUT_PATH, "w") as f:
    json.dump(records, f, indent=2)
print(f"\nwrote {OUT_PATH}")

n_missing = sum(1 for r in records if r["manufacturer"] is None)
print(f"\nscans with NO Manufacturer field: {n_missing}/{len(records)}")

vendors = {}
for r in records:
    v = r["manufacturer"] or "UNKNOWN"
    vendors[v] = vendors.get(v, 0) + 1
print("\nvendor counts (per scan, all subjects/sessions):")
for v, c in sorted(vendors.items(), key=lambda x: -x[1]):
    print(f"  {v}: {c}")
