"""
diagnose_reshape.py
-------------------
Flag 1 (median jump at 2019) and Flag 2 (implausible 100% values) diagnostic.

Compares three candidate reshape rules on cached Comtrade chapter data:
    rule_a : dedup-keep-first across clCodes (current pipeline)
    rule_b : sum across all clCodes
    rule_c : filter to one dominant clCode
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

CACHE_DIR = Path("intermediary/cache/comtrade")

WIDE_RESOURCE_CHAPTERS = {
    "25", "26", "27",
    "71",
    "72", "73", "74", "75", "76", "78", "79", "80", "81",
}

TARGETS = [
    ("NOR", list(range(2015, 2025))),
    ("ZAF", list(range(2015, 2025))),
    ("AZE", list(range(2015, 2025))),
    ("NGA", list(range(2018, 2025))),
    ("COG", list(range(2010, 2020))),
]


def header(n, label):
    line = "=" * 78
    print(f"\n{line}\nBLOCK {n}: {label}\n{line}")


def find_column(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    print(f"ERROR: could not find {kind} column. Tried: {candidates}")
    print(f"Available: {list(df.columns)}")
    sys.exit(1)


def normalize_cmd(s):
    s = str(s).strip()
    if s.upper() in {"TOTAL", "T", "ALL"}:
        return "TOTAL"
    if s.isdigit() and len(s) <= 2:
        return s.zfill(2)
    return s


# BLOCK 1
header(1, "Cache file inventory")
if not CACHE_DIR.exists():
    print(f"ERROR: cache dir not found at {CACHE_DIR.resolve()}")
    sys.exit(1)
cache_files = sorted(CACHE_DIR.glob("*.csv"))
print(f"Found {len(cache_files)} files in {CACHE_DIR.resolve()}")
for f in cache_files[:12]:
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
if len(cache_files) > 12:
    print(f"  ... and {len(cache_files)-12} more")

# BLOCK 2
header(2, "Column inventory and unique-value sanity check")
frames = []
for f in cache_files:
    try:
        df = pd.read_csv(f, low_memory=False)
        df["_source_file"] = f.name
        frames.append(df)
    except Exception as e:
        print(f"  skipped {f.name}: {e}")
raw = pd.concat(frames, ignore_index=True)
print(f"Combined rows: {len(raw):,}")
print(f"Columns ({len(raw.columns)}): {list(raw.columns)}")

iso_col    = find_column(raw, ["reporterISO", "reporterCode", "reporterDesc"], "reporter ISO")
period_col = find_column(raw, ["period", "refYear", "year"], "period")
cmd_col    = find_column(raw, ["cmdCode", "commodityCode", "cmdcode"], "commodity code")
cl_col     = find_column(raw, ["classificationCode", "clCode", "classification"], "classification")
val_col    = find_column(raw, ["primaryValue", "TradeValue", "tradeValue"], "value")
flow_col   = find_column(raw, ["flowCode", "rgCode", "tradeFlowCode"], "flow")

print(f"\nResolved columns:")
print(f"  reporter ISO  : {iso_col}")
print(f"  period        : {period_col}")
print(f"  cmdCode       : {cmd_col}")
print(f"  classification: {cl_col}")
print(f"  value         : {val_col}")
print(f"  flow          : {flow_col}")

raw[flow_col] = raw[flow_col].astype(str).str.upper().str.strip()
before = len(raw)
raw = raw[raw[flow_col].isin(["X", "2"])].copy()
print(f"\nFiltered to exports: {len(raw):,} (dropped {before - len(raw):,})")

raw[period_col] = pd.to_numeric(raw[period_col], errors="coerce").astype("Int64")
raw[val_col]    = pd.to_numeric(raw[val_col], errors="coerce")
raw["cmd_norm"] = raw[cmd_col].map(normalize_cmd)

print(f"\nUnique classifications: {sorted(raw[cl_col].dropna().astype(str).unique().tolist())}")
chap_codes = sorted(c for c in raw["cmd_norm"].dropna().unique() if c != "TOTAL" and len(str(c)) == 2)
print(f"Unique chapter codes ({len(chap_codes)}): {chap_codes}")
print(f"Period range: {raw[period_col].min()} to {raw[period_col].max()}")
print(f"Rows by cmd_norm type: chapter={ (raw['cmd_norm'].str.len()==2).sum() :,}, TOTAL={(raw['cmd_norm']=='TOTAL').sum():,}")

# BLOCK 3
header(3, "Per-target row counts: how many clCodes per (reporter, year, chapter)?")
target_rows = []
for iso, years in TARGETS:
    sub = raw[(raw[iso_col] == iso) & (raw[period_col].isin(years))]
    target_rows.append(sub)
target_raw = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
print(f"Target country-year rows in raw: {len(target_raw):,}")

clcount = (target_raw[target_raw["cmd_norm"].str.len() == 2]
           .groupby([iso_col, period_col, "cmd_norm"])[cl_col]
           .nunique()
           .reset_index(name="n_clCodes"))
multi = clcount[clcount["n_clCodes"] > 1]
print(f"\nCountry-year-chapter cells with >1 clCode: {len(multi)}")
print("(These are the cells where dedup-keep-first changes the answer)")
if len(multi):
    print(multi.head(50).to_string(index=False))

print("\nTOTAL rows per (target country, year), by clCode:")
tot_rows = target_raw[target_raw["cmd_norm"] == "TOTAL"]
tot_pivot = (tot_rows.groupby([iso_col, period_col, cl_col])[val_col]
             .sum().reset_index())
print(tot_pivot.head(60).to_string(index=False))

# BLOCK 4
header(4, "Three reshape rules applied")
chap = raw[raw["cmd_norm"].str.len() == 2].copy()
tot  = raw[raw["cmd_norm"] == "TOTAL"].copy()

# rule_a: dedup keep first on chapter rows
a = (chap.sort_values([iso_col, period_col, "cmd_norm", cl_col])
        .drop_duplicates(subset=[iso_col, period_col, "cmd_norm"], keep="first")
        .groupby([iso_col, period_col, "cmd_norm"])[val_col].sum()
        .rename("rule_a").reset_index())

# rule_b: sum across clCodes
b = (chap.groupby([iso_col, period_col, "cmd_norm"])[val_col].sum()
        .rename("rule_b").reset_index())

# rule_c: filter to dominant clCode
cl_counts = chap[cl_col].astype(str).value_counts()
print(f"clCode row counts (chapter rows):\n{cl_counts.to_string()}")
dominant_cl = cl_counts.index[0]
print(f"\nUsing dominant clCode for rule_c: {dominant_cl}")

c = (chap[chap[cl_col].astype(str) == dominant_cl]
        .groupby([iso_col, period_col, "cmd_norm"])[val_col].sum()
        .rename("rule_c").reset_index())

chap_wide = a.merge(b, on=[iso_col, period_col, "cmd_norm"], how="outer") \
             .merge(c, on=[iso_col, period_col, "cmd_norm"], how="outer")

# TOTAL denominators
tot_a = (tot.sort_values([iso_col, period_col, cl_col])
            .drop_duplicates(subset=[iso_col, period_col], keep="first")
            .groupby([iso_col, period_col])[val_col].sum()
            .rename("total_a").reset_index())
tot_b = (tot.groupby([iso_col, period_col])[val_col].sum()
            .rename("total_b").reset_index())
tot_c = (tot[tot[cl_col].astype(str) == dominant_cl]
            .groupby([iso_col, period_col])[val_col].sum()
            .rename("total_c").reset_index())

totals = tot_a.merge(tot_b, on=[iso_col, period_col], how="outer") \
              .merge(tot_c, on=[iso_col, period_col], how="outer")

wide_chap = chap_wide[chap_wide["cmd_norm"].isin(WIDE_RESOURCE_CHAPTERS)]
wide_sum = (wide_chap.groupby([iso_col, period_col])[["rule_a", "rule_b", "rule_c"]]
            .sum().reset_index()
            .rename(columns={"rule_a":"wide_a","rule_b":"wide_b","rule_c":"wide_c"}))

merged = wide_sum.merge(totals, on=[iso_col, period_col], how="outer")
merged["share_a"] = merged["wide_a"] / merged["total_a"]
merged["share_b"] = merged["wide_b"] / merged["total_b"]
merged["share_c"] = merged["wide_c"] / merged["total_c"]

target_pairs = pd.DataFrame(
    [(iso, y) for iso, years in TARGETS for y in years],
    columns=[iso_col, period_col],
)
target_view = target_pairs.merge(merged, on=[iso_col, period_col], how="left")
print("\nWide-resource share for target country-years:")
print(target_view[[iso_col, period_col, "share_a", "share_b", "share_c"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "  nan"))

# Also show absolute USD numbers for NOR specifically, to see what dedup vs sum implies
print("\nNOR raw USD totals (in billions) under each rule:")
nor = merged[merged[iso_col] == "NOR"].copy()
for col in ["wide_a","wide_b","wide_c","total_a","total_b","total_c"]:
    nor[col] = nor[col] / 1e9
print(nor[[iso_col, period_col, "wide_a","total_a","wide_b","total_b","wide_c","total_c"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "  nan"))

# BLOCK 5
header(5, "Median wide-resource share over time (Flag 1 test)")
med = merged.groupby(period_col)[["share_a", "share_b", "share_c"]].median().reset_index()
med = med.dropna(subset=["share_a", "share_b", "share_c"], how="all")
print(med.to_string(index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "  nan"))

# BLOCK 6
header(6, "Country-years pinned at share >= 0.999 (Flag 2 test)")
for col in ["share_a", "share_b", "share_c"]:
    n = (merged[col] >= 0.999).sum()
    valid = merged[col].notna().sum()
    pct = (n / valid * 100) if valid else 0
    print(f"  {col}: {n} of {valid} country-years at 100% ({pct:.1f}%)")

print("\nCountry-years above 100% (share > 1.01, meaning numerator > denominator):")
for col in ["share_a", "share_b", "share_c"]:
    n = (merged[col] > 1.01).sum()
    print(f"  {col}: {n} country-years exceed 100%")

# BLOCK 7
header(7, "Saving diagnostic CSVs")
merged.to_csv("diagnose_reshape_output.csv", index=False)
print(f"  Wrote diagnose_reshape_output.csv ({len(merged):,} rows)")

cols_out = [iso_col, period_col, "cmd_norm", cl_col, val_col, flow_col, "_source_file"]
target_raw[cols_out].to_csv("diagnose_reshape_targets.csv", index=False)
print(f"  Wrote diagnose_reshape_targets.csv ({len(target_raw):,} rows)")

print("\n" + "=" * 78)
print(f"Done. Dominant clCode used for rule_c: {dominant_cl}")
print("=" * 78)
