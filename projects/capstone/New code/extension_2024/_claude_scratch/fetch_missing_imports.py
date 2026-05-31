"""
Standalone: pull ONLY the missing 2019-2023 imports cache files (yearly).

What this script does:
  - 5 TOTAL imports calls:    TOTAL_2019_2019_M.csv ... TOTAL_2023_2023_M.csv
  - 30 subcode imports calls: subch_<sc>_<y>_<y>_M.csv for sc in
                              {2701, 2702, 2709, 2710, 2711, 2716}
                              and y in {2019, 2020, 2021, 2022, 2023}

What this script does NOT do:
  - Touch exports cache
  - Touch chapter imports cache (already complete via the split script)
  - Re-pull any cache file that already exists
  - Modify the notebook
  - Concat / pivot / merge anything (run the notebook for that after)

Run from anywhere:
    /usr/local/bin/python3.10 <this file path>
"""
import os
import time
from pathlib import Path

import pandas as pd
import comtradeapicall

# ---- config ----
CACHE_DIR = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/intermediary/cache/comtrade')
KEY = '7f352d733ef14582b72f8986bfbc9fcd'

YEARS = [2019, 2020, 2021, 2022, 2023]

SUBCODES = ['2701', '2702', '2709', '2710', '2711', '2716']

INTER_CALL_DELAY = 2      # seconds between successful calls
MAX_RETRIES = 3
RETRY_BACKOFF = 10        # seconds, multiplied by attempt number

# ---- helpers ----

def _clean_response(df, group_cols, value_col='primaryValue'):
    """Match the notebook's _clean_response: C00/mot=0 filter then group-max."""
    if df is None or len(df) == 0:
        return df
    if 'customsCode' in df.columns:
        df = df[df['customsCode'].astype(str) == 'C00']
    if 'motCode' in df.columns:
        df = df[pd.to_numeric(df['motCode'], errors='coerce') == 0]
    if len(df) == 0:
        return df
    agg = df.groupby(group_cols, as_index=False, dropna=False)[value_col].max()
    if 'reporterDesc' in df.columns:
        desc = df[group_cols + ['reporterDesc']].drop_duplicates(group_cols)
        agg = agg.merge(desc, on=group_cols, how='left')
    if 'cmdDesc' in df.columns:
        desc = df[group_cols + ['cmdDesc']].drop_duplicates(group_cols)
        agg = agg.merge(desc, on=group_cols, how='left')
    return agg


def fetch_total_imports_year(year):
    """Pull TOTAL imports for one year. Caches to TOTAL_{y}_{y}_M.csv."""
    cache_path = CACHE_DIR / f'TOTAL_{year}_{year}_M.csv'
    if cache_path.exists():
        print(f'  CACHED  TOTAL {year}')
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            df = comtradeapicall.getFinalData(
                KEY,
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=str(year),
                reporterCode=None,
                cmdCode='TOTAL',
                flowCode='M',
                partnerCode='0',
                partner2Code=None,
                customsCode='C00',
                motCode='0',
                maxRecords=250000,
                format_output='JSON',
                includeDesc=True,
            )
            dt = time.time() - t0

            if df is None or len(df) == 0:
                print(f'  EMPTY   TOTAL {year} (attempt {attempt})')
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                # Persist an empty header so future runs treat this as covered
                header = 'period,reporterISO,reporterDesc,total_imports_usd\n'
                cache_path.write_text(header)
                return True

            df = _clean_response(df, ['reporterISO', 'period'])
            keep = ['period', 'reporterISO', 'reporterDesc', 'primaryValue']
            df = df[[c for c in keep if c in df.columns]].copy()
            df = df.rename(columns={'primaryValue': 'total_imports_usd'})
            df['period'] = pd.to_numeric(df['period'], errors='coerce').astype('Int64')
            df.to_csv(cache_path, index=False)
            print(f'  OK      TOTAL {year}  ({len(df):,} rows, {dt:.1f}s)')
            return True

        except Exception as e:
            print(f'  ERROR   TOTAL {year} attempt {attempt}: {type(e).__name__}: {str(e)[:120]}')
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    print(f'  FAILED  TOTAL {year}  (all retries exhausted)')
    return False


def fetch_subcode_imports_year(subcode, year):
    """Pull HS27 subcode imports for one year. Caches to subch_{sc}_{y}_{y}_M.csv."""
    cache_path = CACHE_DIR / f'subch_{subcode}_{year}_{year}_M.csv'
    if cache_path.exists():
        print(f'  CACHED  HS{subcode} {year}')
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            df = comtradeapicall.getFinalData(
                KEY,
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=str(year),
                reporterCode=None,
                cmdCode=subcode,
                flowCode='M',
                partnerCode=0,
                partner2Code=None,
                customsCode='C00',
                motCode='0',
                maxRecords=250000,
                format_output='JSON',
                aggregateBy=None,
                breakdownMode='classic',
                countOnly=None,
                includeDesc=True,
            )
            dt = time.time() - t0

            if df is None or len(df) == 0:
                print(f'  EMPTY   HS{subcode} {year} (attempt {attempt})')
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                # Persist an empty header so future runs treat this as covered
                header = 'period,reporterISO,reporterDesc,cmdCode,cmdDesc,primaryValue\n'
                cache_path.write_text(header)
                return True

            df = _clean_response(df, ['reporterISO', 'period', 'cmdCode'])
            df.to_csv(cache_path, index=False)
            print(f'  OK      HS{subcode} {year}  ({len(df):,} rows, {dt:.1f}s)')
            return True

        except Exception as e:
            print(f'  ERROR   HS{subcode} {year} attempt {attempt}: {type(e).__name__}: {str(e)[:120]}')
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    print(f'  FAILED  HS{subcode} {year}  (all retries exhausted)')
    return False


# ---- run ----
print(f'Cache directory: {CACHE_DIR}')
print()

failures = []

# Phase 1: TOTAL imports
print('=== Phase 1: TOTAL imports (5 calls) ===')
for y in YEARS:
    ok = fetch_total_imports_year(y)
    if not ok:
        failures.append(('TOTAL', y))
    time.sleep(INTER_CALL_DELAY)

# Phase 2: subcode imports
print()
print('=== Phase 2: HS27 subcode imports (30 calls) ===')
for sc in SUBCODES:
    print(f'-- HS{sc} --')
    for y in YEARS:
        ok = fetch_subcode_imports_year(sc, y)
        if not ok:
            failures.append((f'HS{sc}', y))
        time.sleep(INTER_CALL_DELAY)

# ---- summary ----
print()
print('==================================================')
if failures:
    print(f'DONE with {len(failures)} failures:')
    for what, year in failures:
        print(f'  - {what} {year}')
    print()
    print('Re-run the script to retry only the failed ones (successes are cached).')
else:
    print('DONE: all 35 calls succeeded or were already cached.')
    print()
    print('Next step: re-run notebook from cell 22 onwards.')
    print('Chapter imports (cell 22) and subcode imports (cell 24) will hit')
    print('cache for every year; TOTAL imports (cell 23) will hit cache too.')
    print('Then cell 25 builds the net columns and cell 27 saves the CSV.')
print('==================================================')
