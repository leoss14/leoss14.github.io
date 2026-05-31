"""
Patch e1b_trade_data.ipynb: split the 2019-2023 imports batch into
five year-by-year calls in all three import loops (TOTAL, chapter, subcode).

Cache files for 1995-2018 imports remain valid. The 2019-2023 cache file
(if it exists from a failed attempt) is NOT touched; the new cache filenames
are TOTAL_2019_2019_M.csv etc.

After this patch:
  Old:  YEAR_BATCHES = [(1995,2000), (2001,2006), (2007,2012), (2013,2018), (2019,2023)]
  Imports use:  YEAR_BATCHES_M = the first four batches PLUS yearly: (2019,2019), (2020,2020), ..., (2023,2023)
  Exports still use the original YEAR_BATCHES (their cache is fully populated).
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1b_trade_data.ipynb')
BACKUP = NB_PATH.with_suffix('.ipynb.bak_before_yearly')

with open(NB_PATH) as f:
    nb = json.load(f)

if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, indent=1))
    print(f'Backup written: {BACKUP}')
else:
    print(f'Backup already exists: {BACKUP}')


def set_cell_source(idx, src):
    nb['cells'][idx]['source'] = src.splitlines(keepends=True)
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None


# ============================================================
# Cell 22 - chapter imports loop: use YEAR_BATCHES_M
# ============================================================
CELL_22 = '''\
# Pull all chapter slices for IMPORTS.
# 2019-2023 split year-by-year because the multi-year batch hits a
# Comtrade payload/timeout ceiling for imports (response aborts).
YEAR_BATCHES_M = [b for b in YEAR_BATCHES if b != (2019, 2023)] + [
    (2019, 2019), (2020, 2020), (2021, 2021), (2022, 2022), (2023, 2023),
]
print(f"Imports year-batches: {YEAR_BATCHES_M}")

imports_frames = []
total_calls_M = (len(ALL_CHAPTERS) + 1) * len(YEAR_BATCHES_M)
m_call_i = 0

for chapter in list(ALL_CHAPTERS.keys()):
    print(f"\\n[IMPORTS] Chapter {chapter}: {ALL_CHAPTERS[chapter]}")
    for (y0, y1) in YEAR_BATCHES_M:
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
set_cell_source(22, CELL_22)
print('Patched cell 22 (chapter imports: yearly split for 2019-2023)')


# ============================================================
# Cell 23 - TOTAL imports loop: use YEAR_BATCHES_M
# ============================================================
CELL_23 = '''\
# Pull TOTAL imports (uses YEAR_BATCHES_M from cell above)
total_imports_frames = []
print("[IMPORTS] Pulling TOTAL imports...")
for (y0, y1) in YEAR_BATCHES_M:
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
set_cell_source(23, CELL_23)
print('Patched cell 23 (TOTAL imports: yearly split for 2019-2023)')


# ============================================================
# Cell 24 - HS27 sub-code imports loop: use YEAR_BATCHES_M
# ============================================================
CELL_24 = '''\
# Pull HS27 sub-codes for IMPORTS (uses YEAR_BATCHES_M from cell above)
subcode_imports_frames = []
sub_m_call_i = 0
total_subcode_calls_M = len(HS27_SUBCODES) * len(YEAR_BATCHES_M)

for subcode in list(HS27_SUBCODES.keys()):
    print(f"\\n[IMPORTS] HS{subcode}: {HS27_SUBCODES[subcode]}")
    for (y0, y1) in YEAR_BATCHES_M:
        sub_m_call_i += 1
        cache_path = HS27_SUBCODE_CACHE / f"subch_{subcode}_{y0}_{y1}_M.csv"
        cache_hit = cache_path.exists() and not FORCE_REFRESH
        if not cache_hit:
            print(f"  [{sub_m_call_i}/{total_subcode_calls_M}] fetching {y0}-{y1}...")
        df = pull_one_subcode(subcode, y0, y1, flow_code='M')
        subcode_imports_frames.append(df)
        if not cache_hit:
            time.sleep(INTER_CALL_DELAY)

subcodes_imports_df = pd.concat(subcode_imports_frames, ignore_index=True)
subcodes_imports_df['period'] = pd.to_numeric(subcodes_imports_df['period'],
                                                errors='coerce').astype(int)

print(f"\\nAll sub-code import pulls done. Combined: {len(subcodes_imports_df):,} rows.")
'''
set_cell_source(24, CELL_24)
print('Patched cell 24 (subcode imports: yearly split for 2019-2023)')


with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'\nSaved patched notebook: {NB_PATH}')
print()
print("===================================================")
print("Behavior:")
print("  - Exports loops unchanged (cache fully populated)")
print("  - Imports use YEAR_BATCHES_M: 1995-2000, 2001-2006,")
print("    2007-2012, 2013-2018, 2019, 2020, 2021, 2022, 2023")
print("  - Cache hits for 1995-2018 imports skip API")
print("  - 2019-2023 fetched as 5 separate yearly calls")
print("  - If a stale 2019-2023 _M cache file exists,")
print("    it's no longer referenced (safe to leave or delete)")
print("===================================================")
