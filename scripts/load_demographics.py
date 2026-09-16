"""Stage 1 - build the cohort table by joining T1 filenames to IXI demographics.

Filenames look like 'IXI002-Guys-0828-T1.nii.gz':
    IXI002  -> subject_id
    002     -> numeric IXI_ID used to join against IXI.xls
    Guys    -> acquisition site (Guys / HH / IOP)
"""

import re
from pathlib import Path

import pandas as pd

T1_DIR = Path(r"E:\iAuditData\t1")
XLS_PATH = Path(r"E:\iAuditData\IXI.xls")

# Coding published with the IXI dataset codebook.
SEX_MAP = {1: "M", 2: "F"}
ETHNIC_MAP = {
    1: "White",
    2: "Black or BlackBrit",
    3: "Asian or AsianBrit",
    4: "Chinese",
    5: "Other",
    6: "Other",
}

FNAME_RE = re.compile(r"^(IXI(\d+))-([A-Za-z]+)-(\d+)-T1\.nii\.gz$")


def parse_filename(name):
    """'IXI002-Guys-0828-T1.nii.gz' -> ('IXI002', 2, 'Guys')."""
    m = FNAME_RE.match(name)
    if not m:
        return None
    return {"subject_id": m.group(1), "IXI_ID": int(m.group(2)), "site": m.group(3)}


def dedupe_xls(df, age_col, sex_col, eth_col, verbose=False):
    """IXI.xls ships 52 duplicate IXI_ID rows (26 subjects).

    Almost all are a pair that agrees on everything except ETHNIC_ID, where one
    row carries the real code and its twin carries 0 (= missing). Keep the more
    informative row. A handful genuinely disagree on AGE or SEX - those are
    reported rather than silently resolved.
    """
    dupes = df[df.duplicated("IXI_ID", keep=False)]
    if verbose and len(dupes):
        conflicting = []
        for ixi_id, g in dupes.groupby("IXI_ID"):
            if g[age_col].nunique(dropna=True) > 1 or g[sex_col].nunique() > 1:
                conflicting.append(int(ixi_id))
        print(f"\nduplicate IXI_ID rows in IXI.xls: {len(dupes)} "
              f"({dupes['IXI_ID'].nunique()} subjects)")
        print(f"  resolved by preferring non-zero {eth_col}")
        print(f"  IXI_IDs with genuinely conflicting AGE/SEX: {conflicting or 'none'}")

    # rank: informative ethnicity first, then non-null age
    df = df.copy()
    df["_rank"] = (df[eth_col].fillna(0) == 0).astype(int) * 2 + df[age_col].isna().astype(int)
    df = df.sort_values(["IXI_ID", "_rank"]).drop_duplicates("IXI_ID", keep="first")
    return df.drop(columns="_rank")


def load_xls(verbose=False):
    df = pd.read_excel(XLS_PATH)
    if verbose:
        print("=" * 70)
        print(f"IXI.xls -> {XLS_PATH}   shape={df.shape}")
        print("-" * 70)
        print(f"{'column':<22} {'dtype':<16} n_non_null")
        for c in df.columns:
            print(f"{c:<22} {str(df[c].dtype):<16} {df[c].notna().sum()}")
        print("=" * 70)
    return df


def build_cohort(verbose=False):
    """Return one row per T1 file present, joined to demographics."""
    xls = load_xls(verbose=verbose)

    # Locate columns without assuming exact spelling (the SEX column name embeds
    # its own codebook: 'SEX_ID (1=m, 2=f)').
    sex_col = next((c for c in xls.columns if c.upper().startswith("SEX")), None)
    eth_col = next((c for c in xls.columns if "ETHNIC" in c.upper()), None)
    age_col = next((c for c in xls.columns if c.upper() == "AGE"), None)
    if verbose:
        print(f"sex column      : {sex_col!r}")
        print(f"ethnicity column: {eth_col!r}")
        print(f"age column      : {age_col!r}")

    xls = dedupe_xls(xls, age_col, sex_col, eth_col, verbose=verbose)

    files = sorted(p for p in T1_DIR.glob("*.nii.gz"))
    rows = []
    for p in files:
        parsed = parse_filename(p.name)
        if parsed is None:
            print(f"WARNING: unparseable filename, skipping: {p.name}")
            continue
        parsed["filepath"] = str(p)
        rows.append(parsed)

    if not rows:
        raise SystemExit(f"No parseable T1 files in {T1_DIR}")

    scans = pd.DataFrame(rows)

    keep = [c for c in [age_col, sex_col, eth_col] if c]
    merged = scans.merge(xls[["IXI_ID"] + keep], on="IXI_ID", how="left")

    merged["chronological_age"] = merged[age_col].astype(float)
    merged["sex"] = merged[sex_col].map(SEX_MAP)
    merged["ethnicity"] = (
        merged[eth_col].map(ETHNIC_MAP).fillna("Unknown") if eth_col else "Unknown"
    )
    merged["ethnic_id_raw"] = merged[eth_col] if eth_col else pd.NA

    # IXI.xls has 29/619 subjects with no DOB, so AGE is unrecoverable even
    # though a scan exists (e.g. IXI341: STUDY_DATE present, DOB/AGE blank).
    # Drop these before inference - there is no ground truth to score them
    # against, and a NaN chronological_age would corrupt Pearson r / MAE.
    no_age = merged[merged["chronological_age"].isna()]
    if len(no_age) and verbose:
        print(f"\ndropping {len(no_age)} subject(s) with no AGE in IXI.xls "
              f"(no DOB on file): {no_age['subject_id'].tolist()}")
    merged = merged[merged["chronological_age"].notna()].reset_index(drop=True)

    return merged, {"sex_col": sex_col, "eth_col": eth_col, "age_col": age_col}


def main():
    cohort, meta = build_cohort(verbose=True)

    print("\n--- VERIFY 1: first 5 subjects ---")
    cols = ["subject_id", "site", "chronological_age", "sex", "ethnicity"]
    print(cohort[cols].head(5).to_string(index=False))

    print(f"\nT1 files found: {len(cohort)}")

    print("\n--- counts by sex ---")
    print(cohort["sex"].value_counts(dropna=False).to_string())

    print("\n--- counts by site ---")
    print(cohort["site"].value_counts(dropna=False).to_string())

    print("\n--- ETHNICITY value counts (is this axis usable?) ---")
    print(cohort["ethnicity"].value_counts(dropna=False).to_string())
    print("\nraw ETHNIC_ID counts:")
    print(cohort["ethnic_id_raw"].value_counts(dropna=False).to_string())

    print("\n--- age sanity ---")
    age = cohort["chronological_age"]
    print(f"n_missing_age = {age.isna().sum()}")
    print(f"min={age.min():.1f}  max={age.max():.1f}  mean={age.mean():.1f}")
    ok = age.notna().all() and age.min() > 18 and age.max() < 95
    print(f"ages look like real adult ages: {ok}")

    print(f"\nSEX coding inferred from column name {meta['sex_col']!r}: "
          f"1=M, 2=F  ->  {SEX_MAP}")


if __name__ == "__main__":
    main()
