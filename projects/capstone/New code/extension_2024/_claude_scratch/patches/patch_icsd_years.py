"""Make the ICSD year-column list resilient to source vintage:
intersect requested years with what's actually present in the file.

This fixes the KeyError when an older ICSD snapshot is used (only goes to 2019),
while still picking up newer years if a fresher CSV is dropped in.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'

OLD = 'year_cols = [str(y) for y in range(cfg.YEAR_MIN, cfg.YEAR_MAX + 1)]\n'
NEW = (
    '    # Intersect requested years with what the source actually contains.\n'
    '    # The teammate\'s 2025-12-23 ICSD snapshot only goes to 2019; a fresher\n'
    '    # IMF vintage would extend further. Either works without code changes.\n'
    '    _requested_years = [str(y) for y in range(cfg.YEAR_MIN, cfg.YEAR_MAX + 1)]\n'
    '    year_cols = [y for y in _requested_years if y in imf_icsd_df.columns]\n'
    '    _missing = [y for y in _requested_years if y not in imf_icsd_df.columns]\n'
    '    if _missing:\n'
    '        print(f"  ICSD source missing years: {_missing[0]}-{_missing[-1]} "\n'
    '              f"({len(_missing)} years). These will be NaN until a fresher vintage is pulled.")\n'
)

with open(NB) as f:
    nb = json.load(f)

target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'gfcf_codes' in s and 'imf_icsd_df' in s:
        target = i
        break

if target is None:
    print('ERROR: ICSD cell not found.')
    sys.exit(1)

src = ''.join(nb['cells'][target]['source'])

if '_requested_years' in src:
    print(f'cell {target}: already patched.')
    sys.exit(0)

# Look for the line with appropriate leading indentation (4 spaces, inside the else block)
old_indented = '    year_cols = [str(y) for y in range(cfg.YEAR_MIN, cfg.YEAR_MAX + 1)]\n'

if old_indented not in src:
    print('ERROR: expected line not found in cell. First match attempts:')
    for line in src.splitlines():
        if 'year_cols' in line and 'range' in line:
            print(f'  found: {line!r}')
    sys.exit(1)

src = src.replace(old_indented, NEW)
nb['cells'][target]['source'] = src.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: patched.')
