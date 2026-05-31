"""
Split existing 2019-2023 whole-batch import cache files into yearly slices.

The notebook's patched yearly loop looks for cache files like
ch25_2019_2019_M.csv. The data is already on disk inside ch25_2019_2023_M.csv,
just under a different filename. This script reads each existing whole-batch
file, splits it by period, and writes per-year files.

Does NOT modify or delete the existing 2019_2023_M.csv files (left as-is).
"""
from pathlib import Path
import pandas as pd

CACHE = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/intermediary/cache/comtrade')

YEARS = [2019, 2020, 2021, 2022, 2023]

# Find all *_2019_2023_M.csv files (chapter, TOTAL, subcode)
patterns = list(CACHE.glob('*_2019_2023_M.csv'))
print(f'Found {len(patterns)} whole-batch _M files:')
for p in patterns:
    print(f'  {p.name}')

print()
created = 0
skipped = 0
for p in patterns:
    # Determine the filename stem prefix (e.g. "ch25", "TOTAL", "subch_2701")
    name = p.name
    # Replace the year-range portion with each year
    # e.g. ch25_2019_2023_M.csv -> ch25_2019_2019_M.csv, ch25_2020_2020_M.csv, ...
    df = pd.read_csv(p)
    if 'period' not in df.columns:
        print(f'  SKIP (no period col): {p.name}')
        continue

    for y in YEARS:
        new_name = name.replace('_2019_2023_M.csv', f'_{y}_{y}_M.csv')
        new_path = CACHE / new_name
        if new_path.exists():
            skipped += 1
            continue
        sub = df[df['period'] == y]
        sub.to_csv(new_path, index=False)
        created += 1

print()
print(f'Created: {created} yearly cache files')
print(f'Skipped (already existed): {skipped}')
print()
print('The existing _2019_2023_M.csv files are untouched.')
print('The notebook will now find yearly cache hits for chapter and subcode')
print('imports, and only hit the API for the TOTAL imports yearly windows.')
