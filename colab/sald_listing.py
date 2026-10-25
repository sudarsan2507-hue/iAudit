"""List every SALD T1 on FCP-INDI S3 with size and ETag, and check ETag == MD5 on one file."""
import hashlib
import re
import sys

import pandas as pd
import requests

BASE = "https://fcp-indi.s3.amazonaws.com"
PREFIX = "data/Projects/INDI/SALD/RawData_BIDS/"
OUT = r"E:\iAuditData\sald_probe\sald_t1_listing.csv"

rows, token = [], None
while True:
    params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
    if token:
        params["continuation-token"] = token
    xml = requests.get(BASE + "/", params=params, timeout=120).text
    for key, etag, size in re.findall(r"<Key>([^<]*)</Key>.*?<ETag>&quot;([^&]*)&quot;</ETag>.*?<Size>(\d+)</Size>",
                                      xml, flags=re.S):
        if key.endswith("_T1w.nii.gz"):
            rows.append({"key": key, "etag": etag, "size": int(size)})
    m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
    if not m:
        break
    token = m.group(1)

df = pd.DataFrame(rows)
df["sub_id"] = df["key"].str.extract(r"sub-(\d+)_T1w")[0].astype(int)
multipart = df["etag"].str.contains("-").sum()
print(f"T1 files: {len(df)}  multipart ETags (not plain MD5): {multipart}  "
      f"size {df['size'].min() / 1e6:.1f}-{df['size'].max() / 1e6:.1f} MB")
df.to_csv(OUT, index=False)

probe = sys.argv[1] if len(sys.argv) > 1 else None
if probe:
    h = hashlib.md5(open(probe, "rb").read()).hexdigest()
    et = df.loc[df["sub_id"] == 31274, "etag"].item()
    print(f"probe sub-031274: md5={h}  etag={et}  match={h == et}")
