"""
probe_total_raw.py — make a single unfiltered Comtrade TOTAL call and inspect
the raw response. No dedup, no filtering, no save. Just see what comes back.

Run from extension_2024/:
    export COMTRADE_KEY='your-key-here'
    /usr/local/bin/python3.10 probe_total_raw.py
"""

import os
import sys
import pandas as pd
import comtradeapicall

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

KEY = os.environ.get("COMTRADE_KEY")
if not KEY:
    print("ERROR: COMTRADE_KEY env var not set.")
    print("Run: export COMTRADE_KEY='your-key' before running this script.")
    sys.exit(1)


def probe(label, reporter_code, period_str):
    """Single getFinalData call, returns the raw DataFrame untouched."""
    print(f"\n{'=' * 78}\nPROBE: {label}\n{'=' * 78}")
    df = comtradeapicall.getFinalData(
        KEY,
        typeCode="C",
        freqCode="A",
        clCode="HS",
        period=period_str,
        reporterCode=reporter_code,
        cmdCode="TOTAL",
        flowCode="X",
        partnerCode="0",
        partner2Code=None,
        customsCode=None,
        motCode=None,
        maxRecords=250000,
        format_output="JSON",
        includeDesc=True,
    )
    if df is None or len(df) == 0:
        print("  API returned no data.")
        return None
    print(f"Rows returned: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    return df


# Norway reporter code = 578
print("Trying NOR (reporter 578) 2019-2024...")
nor = probe("NOR 2019-2024", "578", "2019,2020,2021,2022,2023,2024")

if nor is not None:
    # Show columns most likely to disambiguate duplicates
    show_cols = [c for c in [
        "period", "reporterISO", "reporterDesc",
        "classificationCode", "classificationSearchCode",
        "partnerCode", "partner2Code",
        "cmdCode", "flowCode", "flowDesc",
        "customsCode", "motCode",
        "aggrLevel", "isOriginalClassification",
        "primaryValue", "qty", "qtyUnitCode",
        "isReported", "isAggregate",
    ] if c in nor.columns]
    print("\nKey columns view, sorted by period and primaryValue:")
    print(nor[show_cols].sort_values(["period", "primaryValue"], ascending=[True, False]).to_string(index=False))

    # Per (period) row count and primaryValue spread
    print("\nPer-period row count + value distribution:")
    g = nor.groupby("period")["primaryValue"].agg(["count", "min", "max", "sum", "mean"])
    g.columns = ["n_rows", "min_value", "max_value", "sum_value", "mean_value"]
    print(g.to_string())

# Also probe AZE 2019-2024 (reporter code 31)
print("\n\nTrying AZE (reporter 31) 2019-2024...")
aze = probe("AZE 2019-2024", "31", "2019,2020,2021,2022,2023,2024")

if aze is not None:
    show_cols = [c for c in [
        "period", "reporterISO", "classificationCode",
        "partnerCode", "cmdCode", "flowCode",
        "customsCode", "motCode",
        "aggrLevel", "isOriginalClassification",
        "primaryValue", "isReported", "isAggregate",
    ] if c in aze.columns]
    print("\nKey columns:")
    print(aze[show_cols].sort_values(["period", "primaryValue"], ascending=[True, False]).to_string(index=False))

# Summary takeaway
print("\n" + "=" * 78)
print("INTERPRETATION GUIDE")
print("=" * 78)
print("""
If we see multiple rows per period that differ only in classificationCode
(HS92 vs HS2007 vs HS2017 vs HS2022), the fix is:
    Group by (reporterISO, period) and take the row with the LATEST clCode
    (or the largest primaryValue, since partial subset reports will be smaller).

If duplicates differ in partnerCode (some partner=0 world, others partner=specific
country), filter to partnerCode='0' first.

If duplicates differ in customsCode or motCode, pick the aggregate row
(typically customsCode='C00', motCode='0').

If we see only one row per period but the value is still tiny, the problem
is upstream in the API itself (this is what we need to know).
""")
