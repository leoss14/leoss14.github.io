"""
Modify e1b_trade_data.ipynb to also pull imports and compute net trade metrics.

Strategy: minimal-diff, additive only.
  - Cell 8  (pull_one, pull_total):       parameterize with flow_code='X' default
  - Cell 19 (pull_one_subcode):           parameterize with flow_code='X' default
  - Insert 4 NEW cells after cell 20:     mirror exports pulls for imports
  - Replace cell 22 (save):               write extended trade_metrics.csv

Cache filenames stay as-is when flow_code='X' (no suffix), get '_M' suffix when
'M', so the existing 49+ cache files do not need regeneration.

Run this script ONCE to update the notebook. Then run the notebook.
"""
import json
from pathlib import Path
from copy import deepcopy

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1b_trade_data.ipynb')
BACKUP = NB_PATH.with_suffix('.ipynb.bak_before_imports')

with open(NB_PATH) as f:
    nb = json.load(f)

# Backup
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, indent=1))
    print(f'Backup written: {BACKUP}')
else:
    print(f'Backup exists already: {BACKUP}')

def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': src.splitlines(keepends=True)}

def md_cell(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(keepends=True)}

def set_cell_source(idx, src):
    nb['cells'][idx]['source'] = src.splitlines(keepends=True)
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None


# ============================================================
# Cell 8: parameterize pull_one and pull_total with flow_code
# ============================================================
CELL_8 = '''\
def _clean_response(df, group_cols, value_col='primaryValue'):
    """Apply C00/mot=0 filter and reduce duplicates by group-wise max."""
    n_raw = len(df)

    if 'customsCode' in df.columns:
        df = df[df['customsCode'].astype(str) == 'C00']
    if 'motCode' in df.columns:
        df = df[pd.to_numeric(df['motCode'], errors='coerce') == 0]
    n_after_filter = len(df)

    if n_after_filter > 0:
        keep_cols = group_cols + [c for c in
                                  ['reporterDesc', 'cmdDesc'] if c in df.columns]
        agg = (df.groupby(group_cols, as_index=False, dropna=False)
                 [value_col].max())
        if 'reporterDesc' in df.columns:
            desc = df[group_cols + ['reporterDesc']].drop_duplicates(group_cols)
            agg = agg.merge(desc, on=group_cols, how='left')
        if 'cmdDesc' in df.columns:
            desc = df[group_cols + ['cmdDesc']].drop_duplicates(group_cols)
            agg = agg.merge(desc, on=group_cols, how='left')
    else:
        agg = df
    n_final = len(agg)

    print(f"    raw={n_raw:,}  after C00/mot=0 filter={n_after_filter:,}  "
          f"after group-max={n_final:,}")
    return agg


def _flow_suffix(flow_code):
    """Cache-filename suffix per flow. Empty for exports so existing cache
    files (created before the imports refactor) remain valid."""
    if flow_code == 'X':
        return ''
    if flow_code == 'M':
        return '_M'
    raise ValueError(f"Unsupported flow_code={flow_code!r}")


def pull_one(chapter_code, year_start, year_end, max_retries=3, flow_code='X'):
    """Pull a single (chapter, year-batch) slice from Comtrade or cache.

    flow_code: 'X' = exports (default), 'M' = imports.
    """
    suffix = _flow_suffix(flow_code)
    cache_path = COMTRADE_CACHE / f"ch{chapter_code}_{year_start}_{year_end}{suffix}.csv"

    if cache_path.exists() and not FORCE_REFRESH:
        return pd.read_csv(cache_path, dtype={'reporterISO': str,
                                              'cmdCode': str,
                                              'period': int})

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
                flowCode=flow_code,
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
                print(f"  ch{chapter_code} {year_start}-{year_end} flow={flow_code}: "
                      f"empty (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                header = 'period,reporterISO,reporterDesc,cmdCode,cmdDesc,primaryValue\\n'
                cache_path.write_text(header)
                return pd.read_csv(cache_path, dtype={'reporterISO': str,
                                                      'cmdCode': str,
                                                      'period': int})

            print(f"  ch{chapter_code} {year_start}-{year_end} flow={flow_code}: "
                  f"fetched in {dt:.1f}s")
            df = _clean_response(
                df,
                group_cols=['reporterISO', 'period', 'cmdCode'],
                value_col='primaryValue',
            )

            keep = ['period', 'reporterISO', 'reporterDesc',
                    'cmdCode', 'cmdDesc', 'primaryValue']
            df = df[[c for c in keep if c in df.columns]].copy()
            df['period'] = pd.to_numeric(df['period'],
                                          errors='coerce').astype('Int64')
            df.to_csv(cache_path, index=False)
            return df

        except Exception as e:
            print(f"  ch{chapter_code} {year_start}-{year_end} flow={flow_code}: "
                  f"ERROR (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(10 * attempt)
            else:
                raise

    raise RuntimeError(f"Failed to fetch ch{chapter_code} {year_start}-{year_end} flow={flow_code}")


def pull_total(year_start, year_end, max_retries=3, flow_code='X'):
    """Pull TOTAL trade for a year batch (all reporters).

    flow_code: 'X' = total exports (default), 'M' = total imports.
    Value column is renamed to total_exports_usd or total_imports_usd accordingly.
    """
    suffix = _flow_suffix(flow_code)
    cache_path = COMTRADE_CACHE / f"TOTAL_{year_start}_{year_end}{suffix}.csv"
    value_label = 'total_exports_usd' if flow_code == 'X' else 'total_imports_usd'

    if cache_path.exists() and not FORCE_REFRESH:
        return pd.read_csv(cache_path, dtype={'reporterISO': str,
                                              'period': int})

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
                flowCode=flow_code,
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
                print(f"  TOTAL {year_start}-{year_end} flow={flow_code}: "
                      f"empty (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                header = f'period,reporterISO,reporterDesc,{value_label}\\n'
                cache_path.write_text(header)
                return pd.read_csv(cache_path,
                                    dtype={'reporterISO': str, 'period': int})

            print(f"  TOTAL {year_start}-{year_end} flow={flow_code}: "
                  f"fetched in {dt:.1f}s")
            df = _clean_response(
                df,
                group_cols=['reporterISO', 'period'],
                value_col='primaryValue',
            )

            keep = ['period', 'reporterISO', 'reporterDesc', 'primaryValue']
            df = df[[c for c in keep if c in df.columns]].copy()
            df = df.rename(columns={'primaryValue': value_label})
            df['period'] = pd.to_numeric(df['period'],
                                          errors='coerce').astype('Int64')
            df.to_csv(cache_path, index=False)
            return df

        except Exception as e:
            print(f"  TOTAL {year_start}-{year_end} flow={flow_code}: "
                  f"ERROR (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(10 * attempt)
            else:
                raise

    raise RuntimeError(f"Failed to fetch TOTAL {year_start}-{year_end} flow={flow_code}")
'''
set_cell_source(8, CELL_8)
print('Modified cell 8 (pull_one, pull_total: added flow_code parameter)')


# ============================================================
# Cell 19: parameterize pull_one_subcode with flow_code
# ============================================================
CELL_19 = '''\
def pull_one_subcode(subcode, year_start, year_end, max_retries=3, flow_code='X'):
    """Pull a single (HS sub-code, year-batch) slice from Comtrade or cache.

    flow_code: 'X' = exports (default), 'M' = imports.
    Cache key: subch_<subcode>_<y0>_<y1>[_M].csv
    """
    suffix = _flow_suffix(flow_code)
    cache_path = HS27_SUBCODE_CACHE / f"subch_{subcode}_{year_start}_{year_end}{suffix}.csv"

    if cache_path.exists() and not FORCE_REFRESH:
        return pd.read_csv(cache_path, dtype={'reporterISO': str,
                                              'cmdCode': str,
                                              'period': int})

    years = list(range(year_start, year_end + 1))
    period_str = ','.join(str(y) for y in years)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            print(f"    fetching HS{subcode} {year_start}-{year_end} flow={flow_code} (attempt {attempt})...")
            df = comtradeapicall.getFinalData(
                KEY,
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=period_str,
                reporterCode=None,
                cmdCode=subcode,
                flowCode=flow_code,
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
            elapsed = time.time() - t0
            print(f"      got {len(df):,} rows in {elapsed:.1f}s")

            df = _clean_response(df, ['reporterISO', 'period', 'cmdCode'])
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            print(f"      attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(INTER_CALL_DELAY * 3)
    raise RuntimeError(f"All retries failed for HS{subcode} {year_start}-{year_end} flow={flow_code}")


# Pull all (sub-code, year-batch) combinations for EXPORTS (existing behavior)
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
        df = pull_one_subcode(subcode, y0, y1, flow_code='X')
        subcode_frames.append(df)
        if not cache_hit:
            time.sleep(INTER_CALL_DELAY)

subcodes_df = pd.concat(subcode_frames, ignore_index=True)
subcodes_df['period'] = pd.to_numeric(subcodes_df['period'], errors='coerce').astype(int)

print(f"\\nAll sub-code pulls done. Combined: {len(subcodes_df):,} rows.")
print(f"Reporters: {subcodes_df['reporterISO'].nunique()}")
print(f"Year range: {subcodes_df['period'].min()} to {subcodes_df['period'].max()}")
'''
set_cell_source(19, CELL_19)
print('Modified cell 19 (pull_one_subcode: added flow_code parameter)')


# ============================================================
# NEW cells: after cell 20 insert imports pulls and net-metric computation
# ============================================================
NEW_MD_HEADER = '''## 11. Pull imports for net trade metrics

Re-runs the chapter, TOTAL, and sub-code pulls with `flow_code='M'` (imports).
Used to compute net trade metrics that absorb re-exports through customs
hubs (Belgium-Antwerp, Netherlands-Rotterdam, Singapore). Caches are
distinct from the exports caches (`_M` suffix in the filename), so this
section runs alongside the existing exports pipeline without interference.
'''

NEW_CELL_CHAPTER_IMPORTS = '''\
# Pull all chapter slices for IMPORTS
imports_frames = []
total_calls_M = (len(ALL_CHAPTERS) + 1) * len(YEAR_BATCHES)
m_call_i = 0

for chapter in list(ALL_CHAPTERS.keys()):
    print(f"\\n[IMPORTS] Chapter {chapter}: {ALL_CHAPTERS[chapter]}")
    for (y0, y1) in YEAR_BATCHES:
        m_call_i += 1
        cache_path = COMTRADE_CACHE / f"ch{chapter}_{y0}_{y1}_M.csv"
        cache_hit = cache_path.exists() and not FORCE_REFRESH
        if not cache_hit:
            print(f"  [{m_call_i}/{total_calls_M}] fetching {y0}-{y1}...")
        df = pull_one(chapter, y0, y1, flow_code='M')
        imports_frames.append(df)
        if not cache_hit:
            time.sleep(INTER_CALL_DELAY)

chapters_imports_df = pd.concat(imports_frames, ignore_index=True)
chapters_imports_df['period'] = pd.to_numeric(chapters_imports_df['period'],
                                                errors='coerce').astype(int)

print(f"\\nAll chapter import pulls done. Combined: {len(chapters_imports_df):,} rows.")
print(f"Reporters: {chapters_imports_df['reporterISO'].nunique()}")
print(f"Year range: {chapters_imports_df['period'].min()} to {chapters_imports_df['period'].max()}")
'''

NEW_CELL_TOTAL_IMPORTS = '''\
# Pull TOTAL imports
total_imports_frames = []
print("[IMPORTS] Pulling TOTAL imports...")
for (y0, y1) in YEAR_BATCHES:
    m_call_i += 1
    cache_path = COMTRADE_CACHE / f"TOTAL_{y0}_{y1}_M.csv"
    cache_hit = cache_path.exists() and not FORCE_REFRESH
    if not cache_hit:
        print(f"  [{m_call_i}/{total_calls_M}] fetching TOTAL {y0}-{y1}...")
    df = pull_total(y0, y1, flow_code='M')
    total_imports_frames.append(df)
    if not cache_hit:
        time.sleep(INTER_CALL_DELAY)

totals_imports_df = pd.concat(total_imports_frames, ignore_index=True)
totals_imports_df['period'] = pd.to_numeric(totals_imports_df['period'],
                                              errors='coerce').astype(int)

print(f"\\nTotal imports done. {len(totals_imports_df):,} country-year rows.")
'''

NEW_CELL_SUBCODE_IMPORTS = '''\
# Pull HS27 sub-codes for IMPORTS
subcode_imports_frames = []
sub_m_call_i = 0

for subcode in list(HS27_SUBCODES.keys()):
    print(f"\\n[IMPORTS] HS{subcode}: {HS27_SUBCODES[subcode]}")
    for (y0, y1) in YEAR_BATCHES:
        sub_m_call_i += 1
        cache_path = HS27_SUBCODE_CACHE / f"subch_{subcode}_{y0}_{y1}_M.csv"
        cache_hit = cache_path.exists() and not FORCE_REFRESH
        if not cache_hit:
            print(f"  [{sub_m_call_i}/{total_subcode_calls}] fetching {y0}-{y1}...")
        df = pull_one_subcode(subcode, y0, y1, flow_code='M')
        subcode_imports_frames.append(df)
        if not cache_hit:
            time.sleep(INTER_CALL_DELAY)

subcodes_imports_df = pd.concat(subcode_imports_frames, ignore_index=True)
subcodes_imports_df['period'] = pd.to_numeric(subcodes_imports_df['period'],
                                                errors='coerce').astype(int)

print(f"\\nAll sub-code import pulls done. Combined: {len(subcodes_imports_df):,} rows.")
'''

NEW_CELL_BUILD_NET = '''\
# Build the imports-side wide table that mirrors `wide`, then compute net columns.

# Pivot chapters: imports wide
imports_wide = chapters_imports_df.pivot_table(
    index=['reporterISO', 'period'],
    columns='cmdCode',
    values='primaryValue',
    aggfunc='sum',
).reset_index()
for ch in ALL_CHAPTERS:
    if ch not in imports_wide.columns:
        imports_wide[ch] = 0.0
    else:
        imports_wide[ch] = imports_wide[ch].fillna(0.0)
# Rename chapter columns with _imp suffix so we can merge cleanly
chapter_imp_rename = {ch: f"{ch}_imp" for ch in ALL_CHAPTERS}
imports_wide = imports_wide.rename(columns=chapter_imp_rename)

# Join total imports
imports_wide = imports_wide.merge(
    totals_imports_df[['reporterISO', 'period', 'total_imports_usd']],
    on=['reporterISO', 'period'], how='left',
)

# Pivot HS27 sub-codes for imports
subcodes_imports_wide = subcodes_imports_df.pivot_table(
    index=['reporterISO', 'period'],
    columns='cmdCode',
    values='primaryValue',
    aggfunc='sum',
).reset_index()
for sc in HS27_SUBCODES:
    if sc not in subcodes_imports_wide.columns:
        subcodes_imports_wide[sc] = 0.0
    else:
        subcodes_imports_wide[sc] = subcodes_imports_wide[sc].fillna(0.0)
subcode_imp_rename = {sc: f"{sc}_imp" for sc in HS27_SUBCODES}
subcodes_imports_wide = subcodes_imports_wide.rename(columns=subcode_imp_rename)

imports_wide = imports_wide.merge(
    subcodes_imports_wide,
    on=['reporterISO', 'period'], how='left',
)

# Merge imports_wide INTO existing wide
wide = wide.merge(imports_wide, on=['reporterISO', 'period'], how='left')

# Fill any missing imports with 0 (most likely: country did not report imports
# for a chapter in that year)
for ch in ALL_CHAPTERS:
    if f"{ch}_imp" in wide.columns:
        wide[f"{ch}_imp"] = wide[f"{ch}_imp"].fillna(0.0)
for sc in HS27_SUBCODES:
    if f"{sc}_imp" in wide.columns:
        wide[f"{sc}_imp"] = wide[f"{sc}_imp"].fillna(0.0)
if 'total_imports_usd' in wide.columns:
    wide['total_imports_usd'] = wide['total_imports_usd'].fillna(0.0)

# Net chapter values: exports minus imports (can be negative for net importers).
for ch in ALL_CHAPTERS:
    wide[f"{ch}_net"] = wide[ch] - wide[f"{ch}_imp"]
for sc in HS27_SUBCODES:
    wide[f"{sc}_net"] = wide[sc] - wide[f"{sc}_imp"]

# Net group aggregates
wide['extractives_usd_net'] = sum(wide[f"{ch}_net"] for ch in EXTRACTIVES)
wide['base_metals_usd_net'] = sum(wide[f"{ch}_net"] for ch in BASE_METALS)
wide['chemicals_usd_net']   = sum(wide[f"{ch}_net"] for ch in CHEMICALS)
wide['wide_resource_usd_net'] = (wide['extractives_usd_net']
                                  + wide['base_metals_usd_net']
                                  + wide['chemicals_usd_net'])
wide['tight_resource_usd_net'] = wide['26_net'] + wide['27_net'] + wide['71_net']

# Net shares: net resource exports as fraction of gross total exports.
# Using gross total_exports_usd as the denominator (rather than net) keeps the
# share interpretable on the same basis as the existing exports-side shares.
def safe_share(num, denom):
    return (num / denom).where(denom > 0, other=pd.NA)

wide['tight_share_net']         = safe_share(wide['tight_resource_usd_net'],
                                              wide['total_exports_usd'])
wide['extractives_share_net']   = safe_share(wide['extractives_usd_net'],
                                              wide['total_exports_usd'])
wide['base_metals_share_net']   = safe_share(wide['base_metals_usd_net'],
                                              wide['total_exports_usd'])
wide['wide_resource_share_net'] = safe_share(wide['wide_resource_usd_net'],
                                              wide['total_exports_usd'])
wide['hydrocarbon_share_net']   = safe_share(wide['27_net'],
                                              wide['total_exports_usd'])
wide['precious_share_net']      = safe_share(wide['71_net'],
                                              wide['total_exports_usd'])
wide['ores_share_net']          = safe_share(wide['26_net'],
                                              wide['total_exports_usd'])

# Per-chapter net shares
for ch in ALL_CHAPTERS:
    wide[f"hs{ch}_share_net"] = safe_share(wide[f"{ch}_net"],
                                             wide['total_exports_usd'])

# Sub-code net shares
wide['coal_share_net']        = safe_share(wide['2701_net'] + wide['2702_net'],
                                             wide['total_exports_usd'])
wide['crude_oil_share_net']   = safe_share(wide['2709_net'],
                                             wide['total_exports_usd'])
wide['refined_oil_share_net'] = safe_share(wide['2710_net'],
                                             wide['total_exports_usd'])
wide['gas_share_net']         = safe_share(wide['2711_net'],
                                             wide['total_exports_usd'])

# Net resource basket Herfindahl: net resource composition concentration.
# Set negatives to 0 when computing shares-within-resource basket.
wide_resource_total_net = wide['wide_resource_usd_net'].where(
    wide['wide_resource_usd_net'] > 0, other=pd.NA)
chapter_share_cols = list(ALL_CHAPTERS.keys())
herf_net = sum((((wide[f"{ch}_net"].clip(lower=0) / wide_resource_total_net) ** 2))
               for ch in chapter_share_cols)
wide['resource_herfindahl_net'] = herf_net

# Note: net shares can be negative when the country is a net importer in a
# category (e.g. Belgium is a net importer of diamonds in some years). The
# downstream analysis can clip at 0 if a non-negative share is required.
print("Net columns added. Examples (Belgium 2019-2023):")
sample_cols = ['reporterISO', 'period', 'wide_resource_share',
               'wide_resource_share_net', 'tight_share', 'tight_share_net']
print(wide[wide['reporterISO']=='BEL'].sort_values('period').tail(5)[sample_cols].to_string(index=False))
print()
print("Iraq 2019-2023 (gross vs net should be similar; minimal imports):")
print(wide[wide['reporterISO']=='IRQ'].sort_values('period').tail(5)[sample_cols].to_string(index=False))
'''

# Insert new cells after current cell 20 (subcode pivot+shares). Note current
# cell indices: 20 is the subcode pivot/shares cell; 21 is the "Save merged"
# markdown header; 22 is the save cell. We insert before cell 21.
INSERT_AT = 21
new_cells = [
    md_cell(NEW_MD_HEADER),
    code_cell(NEW_CELL_CHAPTER_IMPORTS),
    code_cell(NEW_CELL_TOTAL_IMPORTS),
    code_cell(NEW_CELL_SUBCODE_IMPORTS),
    code_cell(NEW_CELL_BUILD_NET),
]
nb['cells'] = nb['cells'][:INSERT_AT] + new_cells + nb['cells'][INSERT_AT:]
print(f'Inserted {len(new_cells)} new cells at position {INSERT_AT}')

# The save cell (originally cell 22, now at INSERT_AT + len(new_cells) + 1)
# does `wide.to_csv(...)`, which already includes all the new net columns
# automatically (since they were added to `wide`). No edit needed there.

# Save modified notebook
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'\\nSaved modified notebook: {NB_PATH}')
print(f'Total cells now: {len(nb["cells"])}')
print()
print("===========================================")
print("Run the notebook end-to-end to regenerate trade_metrics.csv")
print("with both gross and net columns. Cache will fill in 49 new")
print("'_M' suffix files (chapters + totals + subcodes); existing")
print("exports cache files are untouched.")
print("===========================================")
