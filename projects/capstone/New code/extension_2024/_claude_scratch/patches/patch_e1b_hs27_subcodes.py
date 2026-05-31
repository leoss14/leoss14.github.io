"""Add HS27 sub-code pulls to e1b_trade_data.ipynb.

Inserts three new cells after the existing chapter-pull cells:
  - HS27 sub-code definitions
  - Sub-code pull loop (cached separately from chapter cache)
  - Sub-code merge into wide table + new share columns

Existing cells are untouched. The notebook still works without the new pulls
(they're additive).
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1b_trade_data.ipynb'

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELLS
# ─────────────────────────────────────────────────────────────────────────────

MD_SUBCODE_HEADER = '''## 10. HS27 sub-code disaggregation

HS chapter 27 (Mineral fuels) bundles oil, gas, coal, and refined petroleum
into a single line. For the resource-curse analysis these need to be
distinguished because they have different price dynamics, geological
characteristics, and policy implications.

Pulls 6 sub-codes (4-digit HS) for the full year range:

| Code | Content |
|---|---|
| HS2701 | Coal (anthracite, bituminous, etc.) |
| HS2702 | Lignite, peat |
| HS2709 | Crude petroleum oils |
| HS2710 | Refined petroleum products (gasoline, diesel, fuel oil, lubricants) |
| HS2711 | Petroleum gases (natural gas, LPG) |
| HS2716 | Electricity (negligible; included for completeness) |

Cached separately from the chapter-level cache to avoid collision.

Produces 4 new share variables: `coal_share` (HS2701+2702), `crude_oil_share`
(HS2709), `refined_oil_share` (HS2710), `gas_share` (HS2711). `hydrocarbon_share`
(chapter HS27 total) is preserved unchanged for backward compatibility and as
a robustness check.
'''

CODE_SUBCODE_DEFS = '''# Sub-codes of HS27 to pull at 4-digit resolution.
HS27_SUBCODES = {
    '2701': 'Coal (anthracite, bituminous, etc.)',
    '2702': 'Lignite, peat',
    '2709': 'Crude petroleum oils',
    '2710': 'Refined petroleum products',
    '2711': 'Petroleum gases (natural gas, LPG)',
    '2716': 'Electricity',
}

HS27_SUBCODE_CACHE = COMTRADE_CACHE  # same directory, different filename pattern

# Same year batches as chapter pulls
print(f"Sub-codes: {list(HS27_SUBCODES.keys())}")
print(f"Year batches: {YEAR_BATCHES}")
print(f"Total sub-code API calls: {len(HS27_SUBCODES) * len(YEAR_BATCHES)}")
'''

CODE_SUBCODE_PULL = '''def pull_one_subcode(subcode, year_start, year_end, max_retries=3):
    """Pull a single (HS sub-code, year-batch) slice from Comtrade or cache.

    Cache key: subch_<subcode>_<y0>_<y1>.csv (distinct from chapter cache).
    """
    cache_path = HS27_SUBCODE_CACHE / f"subch_{subcode}_{year_start}_{year_end}.csv"

    if cache_path.exists() and not FORCE_REFRESH:
        return pd.read_csv(cache_path, dtype={'reporterISO': str,
                                              'cmdCode': str,
                                              'period': int})

    years = list(range(year_start, year_end + 1))
    period_str = ','.join(str(y) for y in years)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            print(f"    fetching HS{subcode} {year_start}-{year_end} (attempt {attempt})...")
            df = ct.getFinalData(
                subscription_key,
                typeCode='C',           # commodities
                freqCode='A',           # annual
                clCode='HS',
                period=period_str,
                reporterCode=None,      # all reporters
                cmdCode=subcode,        # 4-digit sub-code
                flowCode='X',           # exports
                partnerCode=0,          # World
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
            elapsed = time.time() - t0
            print(f"      got {len(df):,} rows in {elapsed:.1f}s")

            df = _clean_response(df, ['reporterISO', 'period', 'cmdCode'])
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            print(f"      attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(INTER_CALL_DELAY * 3)
    raise RuntimeError(f"All retries failed for HS{subcode} {year_start}-{year_end}")


# Pull all (sub-code, year-batch) combinations
subcode_frames = []
total_subcode_calls = len(HS27_SUBCODES) * len(YEAR_BATCHES)
sub_call_i = 0

for subcode in list(HS27_SUBCODES.keys()):
    print(f"\\nHS{subcode}: {HS27_SUBCODES[subcode]}")
    for (y0, y1) in YEAR_BATCHES:
        sub_call_i += 1
        cache_path = HS27_SUBCODE_CACHE / f"subch_{subcode}_{y0}_{y1}.csv"
        cache_hit = cache_path.exists() and not FORCE_REFRESH
        if not cache_hit:
            print(f"  [{sub_call_i}/{total_subcode_calls}] fetching {y0}-{y1}...")
        df = pull_one_subcode(subcode, y0, y1)
        subcode_frames.append(df)
        if not cache_hit:
            time.sleep(INTER_CALL_DELAY)

subcodes_df = pd.concat(subcode_frames, ignore_index=True)
subcodes_df['period'] = pd.to_numeric(subcodes_df['period'], errors='coerce').astype(int)

print(f"\\nAll sub-code pulls done. Combined: {len(subcodes_df):,} rows.")
print(f"Reporters: {subcodes_df['reporterISO'].nunique()}")
print(f"Year range: {subcodes_df['period'].min()} to {subcodes_df['period'].max()}")
'''

CODE_SUBCODE_MERGE = '''# Pivot sub-codes to wide, one column per sub-code
subcodes_wide = subcodes_df.pivot_table(
    index=['reporterISO', 'period'],
    columns='cmdCode',
    values='primaryValue',
    aggfunc='sum',
).reset_index()

for sc in HS27_SUBCODES:
    if sc not in subcodes_wide.columns:
        subcodes_wide[sc] = 0.0
    else:
        subcodes_wide[sc] = subcodes_wide[sc].fillna(0.0)

# Merge into the main wide table (already has total_exports_usd and chapter shares)
wide = wide.merge(
    subcodes_wide,
    on=['reporterISO', 'period'],
    how='left',
    suffixes=('', '_sub'),
)

# Compute new disaggregated shares
wide['coal_share']         = safe_share(wide['2701'] + wide['2702'],
                                         wide['total_exports_usd'])
wide['crude_oil_share']    = safe_share(wide['2709'], wide['total_exports_usd'])
wide['refined_oil_share']  = safe_share(wide['2710'], wide['total_exports_usd'])
wide['gas_share']          = safe_share(wide['2711'], wide['total_exports_usd'])
# Electricity (2716) intentionally omitted as a share — negligible

# Sanity check: HS27 sub-codes sum should approximate the chapter total
chap27_sum_check = (
    wide['2701'] + wide['2702'] + wide['2709']
    + wide['2710'] + wide['2711'] + wide['2716']
)
chap27_direct = wide['27']
discrepancy = (chap27_sum_check - chap27_direct).abs() / chap27_direct.replace(0, pd.NA)

print(f"HS27 sub-code consistency check:")
print(f"  Median relative discrepancy:   {discrepancy.median()*100:.2f}%")
print(f"  95th percentile discrepancy:   {discrepancy.quantile(0.95)*100:.2f}%")
print(f"  Max discrepancy:               {discrepancy.max()*100:.2f}%")

# Cap shares at 1.0 defensively
for c in ['coal_share', 'crude_oil_share', 'refined_oil_share', 'gas_share']:
    wide[c] = wide[c].clip(upper=1.0)

print(f"\\nNew columns added to wide: coal_share, crude_oil_share, "
      f"refined_oil_share, gas_share")
print(f"hydrocarbon_share preserved unchanged.")
'''


def patch():
    with open(NB) as f:
        nb = json.load(f)

    # Idempotency
    for c in nb['cells']:
        if c['cell_type'] != 'markdown':
            continue
        if '10. HS27 sub-code disaggregation' in ''.join(c.get('source', [])):
            print('Section 10 already present. No-op.')
            return

    # Find the cell that defines and exports trade_metrics (the cell that runs
    # the merge -- we'll insert BEFORE it so the new columns are available
    # when the export runs). Cell 18 is the "save trade metrics" cell.
    # We insert after the wide-table-construction cell (cell 16).

    insert_after = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'wide_resource_share' in s and 'wide.merge' in s and 'pivot_table' in s:
            insert_after = i
            break

    if insert_after is None:
        print('ERROR: could not find the wide-table cell to insert after.')
        sys.exit(1)

    print(f'Inserting new cells after cell {insert_after}')

    new_cells = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': MD_SUBCODE_HEADER.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': CODE_SUBCODE_DEFS.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': CODE_SUBCODE_PULL.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': CODE_SUBCODE_MERGE.splitlines(keepends=True)},
    ]

    nb['cells'] = nb['cells'][:insert_after + 1] + new_cells + nb['cells'][insert_after + 1:]

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Added {len(new_cells)} cells. Total cells now: {len(nb["cells"])}')


if __name__ == '__main__':
    patch()
