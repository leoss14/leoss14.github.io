#!/usr/bin/env python3.10
"""
Build extension_2024/e1b_trade_data.ipynb.

Pulls Comtrade exports for resource-related HS chapters (1995-2024) and
builds the wide-approach trade metrics. Designed to be resumable through
on-disk caching so a partial-run failure doesn't lose any work.
"""

import json
import os

EXT = "/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024"
out_path = os.path.join(EXT, "e1b_trade_data.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(text):
    lines = text.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]],
    }


cells = []

cells.append(md("""\
# extension_2024 / e1b: UN Comtrade trade data pull (wide approach)

Pulls country-year-chapter export values from UN Comtrade for 1995-2024 and
constructs four nested resource-export shares. Replaces the WB rents
indicators (% of GDP) used in the live pipeline.

## Approach

For each of 14 resource-related HS chapters plus TOTAL exports, one API call
returns all reporters × 12 years × world-aggregated partner. Three batches
cover 1995-2024. Results are cached per (chapter, batch) so partial failures
are resumable.

## HS chapter scope (the "wide" definition)

- **Extractives** (chapters 25, 26, 27, 44, 71): salt/sulfur/stone, ores,
  mineral fuels, raw wood, precious stones and metals
- **Base metals** (72, 74, 75, 76, 78, 79, 80, 81): iron/steel, copper,
  nickel, aluminium, lead, zinc, tin, other base metals (cobalt, tungsten,
  molybdenum, manganese, titanium)
- **Selected chemicals** (28, 29): inorganic and organic chemicals
  including phosphate, uranium hydroxide, lithium carbonate

## Output metrics

- `tight_share`        = (26 + 27 + 71) / TOTAL    [raw resource only]
- `extractives_share`  = (25 + 26 + 27 + 44 + 71) / TOTAL
- `base_metals_share`  = (72 + 74 + ... + 81) / TOTAL
- `wide_resource_share` = extractives + base_metals + (28 + 29) / TOTAL
- Per-chapter shares for case-study slicing
- Resource export concentration (Herfindahl over chapter shares)

## Prerequisites

- `pip install comtradeapicall`
- Set Comtrade API key via env var: `export COMTRADE_KEY='your-key'`

## Runtime estimate

- 45 API calls × ~50s each = ~38 minutes for a cold run
- Subsequent runs read from cache in seconds
"""))

cells.append(md("## 0. Setup"))

cells.append(code("""\
import os
import sys
import time
import re
from pathlib import Path

import pandas as pd
import comtradeapicall

# The notebook's working directory is extension_2024/. _config lives here.
import _config as cfg

# API key from environment (do not commit subscription keys to the repo)
KEY = os.environ.get('COMTRADE_KEY')
if not KEY:
    raise SystemExit(
        "Set COMTRADE_KEY env var before running this notebook:\\n"
        "  export COMTRADE_KEY='your-key-here'"
    )

# Cache directory: each (chapter, batch) result writes here as a CSV.
# Re-running this notebook skips any (chapter, batch) whose cache file exists.
# Delete files individually to force-refresh that slice.
COMTRADE_CACHE = Path(cfg.CACHE) / 'comtrade'
COMTRADE_CACHE.mkdir(parents=True, exist_ok=True)

# Set FORCE_REFRESH = True to ignore all cache files and re-pull
FORCE_REFRESH = False

# Polite delay between successful API calls (seconds)
INTER_CALL_DELAY = 2

print(f"Cache directory: {COMTRADE_CACHE}")
print(f"Year range: {cfg.YEAR_MIN}-{cfg.YEAR_MAX}")
print(f"FORCE_REFRESH: {FORCE_REFRESH}")"""))

cells.append(md("## 1. HS chapter scope"))

cells.append(code("""\
# Chapters to pull, grouped by economic role.
# The wide-approach final metric combines extractives + base_metals + chemicals.

EXTRACTIVES = {
    '25': 'Salt, sulfur, earths, stone, plastering, lime, cement',
    '26': 'Ores, slag and ash',
    '27': 'Mineral fuels, mineral oils, petroleum, gas',
    '44': 'Wood and articles of wood, wood charcoal',
    '71': 'Pearls, precious stones, precious metals, coins',
}

BASE_METALS = {
    '72': 'Iron and steel',
    '74': 'Copper and articles thereof',
    '75': 'Nickel and articles thereof',
    '76': 'Aluminium and articles thereof',
    '78': 'Lead and articles thereof',
    '79': 'Zinc and articles thereof',
    '80': 'Tin and articles thereof',
    '81': 'Other base metals (cobalt, tungsten, molybdenum, manganese, titanium)',
}

CHEMICALS = {
    '28': 'Inorganic chemicals (phosphate, uranium hydroxide, lithium carbonate)',
    '29': 'Organic chemicals',
}

ALL_CHAPTERS = {**EXTRACTIVES, **BASE_METALS, **CHEMICALS}

# Year batches (Comtrade max 12 years per call)
YEAR_BATCHES = []
start = cfg.YEAR_MIN
while start <= cfg.YEAR_MAX:
    end = min(start + 11, cfg.YEAR_MAX)
    YEAR_BATCHES.append((start, end))
    start = end + 1

print(f"Chapters: {len(ALL_CHAPTERS)} ({len(EXTRACTIVES)} extractives, "
      f"{len(BASE_METALS)} base metals, {len(CHEMICALS)} chemicals)")
print(f"Year batches: {YEAR_BATCHES}")
print(f"Total API calls needed (including TOTAL series): "
      f"{(len(ALL_CHAPTERS) + 1) * len(YEAR_BATCHES)}")"""))

cells.append(md("## 2. Pull function with caching and retries"))

cells.append(code("""\
def pull_one(chapter_code, year_start, year_end, max_retries=3):
    \"\"\"Pull a single (chapter, year-batch) slice from Comtrade or cache.

    Returns a DataFrame with columns: period, reporterISO, reporterDesc,
    cmdCode, cmdDesc, primaryValue.

    The Saudi-style duplicate-row quirk (multiple HS classifications submitted
    for the same reporter-year) is handled here: rows are deduped on
    (reporterISO, period, cmdCode) keeping the first occurrence.
    \"\"\"
    cache_path = COMTRADE_CACHE / f"ch{chapter_code}_{year_start}_{year_end}.csv"

    if cache_path.exists() and not FORCE_REFRESH:
        df = pd.read_csv(cache_path, dtype={'reporterISO': str, 'cmdCode': str})
        return df

    years = list(range(year_start, year_end + 1))
    period_str = ','.join(str(y) for y in years)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            df = comtradeapicall.getFinalData(
                KEY,
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=period_str,
                reporterCode=None,
                cmdCode=str(chapter_code),
                flowCode='X',
                partnerCode='0',
                partner2Code=None,
                customsCode=None,
                motCode=None,
                maxRecords=250000,
                format_output='JSON',
                includeDesc=True,
            )
            dt = time.time() - t0

            if df is None or len(df) == 0:
                print(f"  ch{chapter_code} {year_start}-{year_end}: empty (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                else:
                    cache_path.write_text("period,reporterISO,reporterDesc,cmdCode,cmdDesc,primaryValue\\n")
                    return pd.read_csv(cache_path, dtype={'reporterISO': str, 'cmdCode': str})

            keep = ['period', 'reporterISO', 'reporterDesc', 'cmdCode', 'cmdDesc', 'primaryValue']
            df = df[keep].copy()

            # Dedupe the multi-classification quirk
            before = len(df)
            df = df.drop_duplicates(subset=['reporterISO', 'period', 'cmdCode'], keep='first')
            dropped = before - len(df)

            df.to_csv(cache_path, index=False)
            print(f"  ch{chapter_code} {year_start}-{year_end}: {len(df):,} rows in {dt:.1f}s "
                  f"({dropped} dups removed)")
            return df

        except Exception as e:
            print(f"  ch{chapter_code} {year_start}-{year_end}: ERROR (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(10 * attempt)
            else:
                raise

    raise RuntimeError(f"Failed to fetch ch{chapter_code} {year_start}-{year_end}")"""))

cells.append(md("## 3. Pull all chapters"))

cells.append(code("""\
all_frames = []
total_calls = (len(ALL_CHAPTERS) + 1) * len(YEAR_BATCHES)
call_i = 0

for chapter in list(ALL_CHAPTERS.keys()):
    print(f"\\nChapter {chapter}: {ALL_CHAPTERS[chapter]}")
    for (y0, y1) in YEAR_BATCHES:
        call_i += 1
        cache_path = COMTRADE_CACHE / f"ch{chapter}_{y0}_{y1}.csv"
        cache_hit = cache_path.exists() and not FORCE_REFRESH
        if not cache_hit:
            print(f"  [{call_i}/{total_calls}] fetching {y0}-{y1}...")
        df = pull_one(chapter, y0, y1)
        all_frames.append(df)
        if not cache_hit:
            time.sleep(INTER_CALL_DELAY)

chapters_df = pd.concat(all_frames, ignore_index=True)
print(f"\\nAll chapter pulls done. Combined: {len(chapters_df):,} rows.")
print(f"Reporters: {chapters_df['reporterISO'].nunique()}")
print(f"Year range: {chapters_df['period'].min()} to {chapters_df['period'].max()}")"""))

cells.append(md("## 4. Pull TOTAL exports (denominator)"))

cells.append(code("""\
def pull_total(year_start, year_end, max_retries=3):
    \"\"\"Pull TOTAL exports for a year batch (all reporters).\"\"\"
    cache_path = COMTRADE_CACHE / f"TOTAL_{year_start}_{year_end}.csv"

    if cache_path.exists() and not FORCE_REFRESH:
        return pd.read_csv(cache_path, dtype={'reporterISO': str})

    years = list(range(year_start, year_end + 1))
    period_str = ','.join(str(y) for y in years)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            df = comtradeapicall.getFinalData(
                KEY,
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=period_str,
                reporterCode=None,
                cmdCode='TOTAL',
                flowCode='X',
                partnerCode='0',
                partner2Code=None,
                customsCode=None,
                motCode=None,
                maxRecords=250000,
                format_output='JSON',
                includeDesc=True,
            )
            dt = time.time() - t0
            if df is None or len(df) == 0:
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                cache_path.write_text("period,reporterISO,reporterDesc,primaryValue\\n")
                return pd.read_csv(cache_path)

            df = df[['period', 'reporterISO', 'reporterDesc', 'primaryValue']].copy()
            df = df.drop_duplicates(subset=['reporterISO', 'period'], keep='first')
            df = df.rename(columns={'primaryValue': 'total_exports_usd'})
            df.to_csv(cache_path, index=False)
            print(f"  TOTAL {year_start}-{year_end}: {len(df):,} rows in {dt:.1f}s")
            return df
        except Exception as e:
            print(f"  TOTAL {year_start}-{year_end}: ERROR (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(10 * attempt)
            else:
                raise

total_frames = []
print("Pulling TOTAL exports...")
for (y0, y1) in YEAR_BATCHES:
    call_i += 1
    cache_path = COMTRADE_CACHE / f"TOTAL_{y0}_{y1}.csv"
    cache_hit = cache_path.exists() and not FORCE_REFRESH
    if not cache_hit:
        print(f"  [{call_i}/{total_calls}] fetching TOTAL {y0}-{y1}...")
    df = pull_total(y0, y1)
    total_frames.append(df)
    if not cache_hit:
        time.sleep(INTER_CALL_DELAY)

totals_df = pd.concat(total_frames, ignore_index=True)
print(f"\\nTotal exports done. {len(totals_df):,} country-year rows.")"""))

cells.append(md("## 5. Reshape to wide and build metrics"))

cells.append(code("""\
# Wide-form: one row per (reporter, year), one column per chapter
wide = chapters_df.pivot_table(
    index=['reporterISO', 'period'],
    columns='cmdCode',
    values='primaryValue',
    aggfunc='sum',  # collapse any remaining dupes
).reset_index()

# Ensure all chapter columns exist even if some had no data
for ch in ALL_CHAPTERS:
    if ch not in wide.columns:
        wide[ch] = 0.0
    else:
        wide[ch] = wide[ch].fillna(0.0)

# Join total exports
wide = wide.merge(totals_df[['reporterISO', 'period', 'total_exports_usd']],
                  on=['reporterISO', 'period'], how='left')

# Build group sums
wide['extractives_usd'] = sum(wide[ch] for ch in EXTRACTIVES)
wide['base_metals_usd'] = sum(wide[ch] for ch in BASE_METALS)
wide['chemicals_usd']   = sum(wide[ch] for ch in CHEMICALS)
wide['wide_resource_usd'] = wide['extractives_usd'] + wide['base_metals_usd'] + wide['chemicals_usd']
wide['tight_resource_usd'] = wide['26'] + wide['27'] + wide['71']

# Shares
def safe_share(num, denom):
    return (num / denom).where(denom > 0, other=pd.NA)

wide['tight_share']       = safe_share(wide['tight_resource_usd'], wide['total_exports_usd'])
wide['extractives_share'] = safe_share(wide['extractives_usd'], wide['total_exports_usd'])
wide['base_metals_share'] = safe_share(wide['base_metals_usd'], wide['total_exports_usd'])
wide['wide_resource_share'] = safe_share(wide['wide_resource_usd'], wide['total_exports_usd'])
wide['hydrocarbon_share'] = safe_share(wide['27'], wide['total_exports_usd'])
wide['precious_share']    = safe_share(wide['71'], wide['total_exports_usd'])
wide['ores_share']        = safe_share(wide['26'], wide['total_exports_usd'])

# Per-chapter shares (named for clarity in the merged panel)
for ch in ALL_CHAPTERS:
    wide[f'hs{ch}_share'] = safe_share(wide[ch], wide['total_exports_usd'])

# Resource basket Herfindahl: concentration within the wide-resource bundle.
# 1.0 means all resource exports are one chapter; 1/14 = 0.071 means evenly spread.
chapter_share_cols = list(ALL_CHAPTERS.keys())
wide_resource_total = wide['wide_resource_usd'].where(wide['wide_resource_usd'] > 0, other=pd.NA)
herf = sum(((wide[ch] / wide_resource_total) ** 2) for ch in chapter_share_cols)
wide['resource_herfindahl'] = herf

# Sanity caps: trade shares cannot exceed 1.0 (rounding / data quirks can push slightly over)
for c in [c for c in wide.columns if c.endswith('_share')]:
    wide[c] = wide[c].clip(upper=1.0)

print(f"Wide shape: {wide.shape}")
print(f"Reporters: {wide['reporterISO'].nunique()}, Years: {wide['period'].nunique()}")
print()
print("Sample (Chile, last 5 years):")
print(wide[wide['reporterISO']=='CHL'].sort_values('period').tail(5)[
    ['reporterISO','period','total_exports_usd','tight_share','extractives_share',
     'base_metals_share','wide_resource_share','hs27_share','hs74_share']
].to_string(index=False))"""))

cells.append(md("## 6. Save merged trade metrics"))

cells.append(code("""\
out_path = Path(cfg.INTERMEDIARY) / 'trade_metrics.csv'
wide = wide.rename(columns={'reporterISO': 'Country Code', 'period': 'Year'})
wide.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(f"Schema: {len(wide.columns)} columns, {len(wide):,} rows")"""))

cells.append(md("## 7. Coverage diagnostic"))

cells.append(code("""\
import numpy as np

print("Coverage by year (countries with non-null total_exports_usd):")
cov = wide.groupby('Year').agg(
    n_countries=('Country Code', 'nunique'),
    n_with_total=('total_exports_usd', lambda s: s.notna().sum()),
    median_wide_share=('wide_resource_share', 'median'),
).reset_index()
print(cov.to_string(index=False))
print()

# Spot-check case-study countries
print("Case-study countries, wide_resource_share by year (last 10):")
for iso in ['CHL', 'COG', 'AZE', 'SAU', 'NOR', 'NGA', 'ZAF']:
    sub = wide[(wide['Country Code'] == iso)].sort_values('Year').tail(10)
    if len(sub):
        years_str = ' '.join(str(int(r['Year'])) for _, r in sub.iterrows())
        shares_str = ' '.join(f"{(r['wide_resource_share'] or 0)*100:5.1f}" for _, r in sub.iterrows())
        print(f"  {iso}: {sub['Year'].min():.0f}-{sub['Year'].max():.0f}  "
              f"wide% last 10 = [{shares_str}]")
    else:
        print(f"  {iso}: no rows")"""))

cells.append(md("---\n\n## Summary\n\nOnce this notebook completes, downstream notebooks can merge `intermediary/trade_metrics.csv` with the main panel on (Country Code, Year). The wide-approach metrics replace the rents-based resource-dependence indicators:\n\n- `wide_resource_share` replaces `Total natural resources rents (% of GDP)` as the headline indicator\n- `hydrocarbon_share` (hs27) replaces `Oil rents + Natural gas rents (% of GDP)`\n- `ores_share` (hs26) replaces `Mineral rents (% of GDP)` for raw ores\n- `base_metals_share` is new; lets you distinguish raw exporters from processed-metal exporters (relevant for the Chile case study)\n- `resource_herfindahl` is new; concentration of the resource basket within the resource bundle\n\nThe per-chapter shares (hs25_share through hs81_share) are kept for case-study work."))

# Wrap into a notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "py310",
            "language": "python",
            "name": "py310",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {out_path}")
print(f"Total cells: {len(cells)}")
