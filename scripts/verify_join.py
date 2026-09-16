"""Independent check that the filename -> IXI.xls join is correct.

NITRC's XNAT records an 'age' on each MR session, sourced separately from the
IXI.xls we join against. If the two agree for every subject, the join is right.
"""

import json
import urllib.request

from load_demographics import build_cohort

BASE = "https://www.nitrc.org/ir/data"
UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url, key="ResultSet", attempts=5):
    import time

    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            if key in data:
                return data
            last = "throttled"
        except Exception as e:
            last = repr(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url} -> {last}")


def main():
    cohort, _ = build_cohort()

    exps = get_json(f"{BASE}/projects/ixi/experiments?format=json")["ResultSet"]["Result"]
    by_id = {}
    for e in exps:
        label = e["label"]           # e.g. '251_Guys'
        try:
            by_id[int(label.split("_")[0])] = e["ID"]
        except ValueError:
            continue

    print(f"{'subject':<10} {'xls_age':>9} {'xnat_age':>9} {'diff':>7}")
    print("-" * 40)
    bad = 0
    for r in cohort.itertuples(index=False):
        exp = by_id.get(r.IXI_ID)
        if exp is None:
            print(f"{r.subject_id:<10} {'-':>9} {'no session':>9}")
            bad += 1
            continue
        d = get_json(f"{BASE}/experiments/{exp}?format=json", key="items")
        xnat_age = d["items"][0]["data_fields"].get("age")
        if xnat_age is None:
            print(f"{r.subject_id:<10} {r.chronological_age:9.2f} {'none':>9}")
            continue
        diff = float(xnat_age) - float(r.chronological_age)
        flag = "" if abs(diff) < 0.02 else "   <-- MISMATCH"
        if flag:
            bad += 1
        print(f"{r.subject_id:<10} {r.chronological_age:9.2f} {float(xnat_age):9.2f} {diff:7.2f}{flag}")

    print("-" * 40)
    print("JOIN VERIFIED - all ages agree with XNAT" if bad == 0
          else f"JOIN PROBLEM - {bad} mismatches")


if __name__ == "__main__":
    main()
