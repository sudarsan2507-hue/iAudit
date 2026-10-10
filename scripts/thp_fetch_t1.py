"""Download the 65 real-subject (excludes sub-THPBALL, a physical phantom)
T1w scans from ds000206 (THP, OpenNeuro) to E:\\iAuditData\\thp\\.
"""
import json
import time
import urllib.request
from pathlib import Path

TREE_PATH = r"C:\Users\sudar\AppData\Local\Temp\thp_tree.json"
# raw.githubusercontent.com only serves the git-annex pointer file for these
# (content is annexed to S3), so use OpenNeuro's own file-download API, which
# 302-redirects to the real S3 object.
API_BASE = "https://openneuro.org/crn/datasets/ds000206/snapshots/1.0.0/files/"
OUT_DIR = Path(r"E:\iAuditData\thp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tree = json.load(open(TREE_PATH))
t1w_nii_paths = sorted(
    t["path"] for t in tree["tree"]
    if t["path"].endswith("_T1w.nii.gz") and "sub-THPBALL" not in t["path"]
)
print(f"downloading {len(t1w_nii_paths)} real-subject T1w.nii.gz files")

got = 0
for i, path in enumerate(t1w_nii_paths, 1):
    fname = path.split("/")[-1]
    dest = OUT_DIR / fname
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [{i}/{len(t1w_nii_paths)}] have {fname}  ({dest.stat().st_size/1e6:.1f} MB)")
        got += 1
        continue
    api_path = path.replace("/", ":")  # sub-x/ses-y/anat/file -> sub-x:ses-y:anat:file
    url = API_BASE + api_path
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
                f.write(r.read())
            size = dest.stat().st_size
            if size < 1_000_000:
                raise ValueError(f"suspiciously small ({size} bytes) - not real content")
            print(f"  [{i}/{len(t1w_nii_paths)}] got  {fname}  ({size/1e6:.1f} MB)")
            got += 1
            break
        except Exception as e:
            print(f"  [{i}/{len(t1w_nii_paths)}] attempt {attempt+1} FAILED {fname}: {e}")
            time.sleep(2)

print(f"\ndownloaded/present: {got}/{len(t1w_nii_paths)} in {OUT_DIR}")
