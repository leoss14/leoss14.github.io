"""Patch cell 4 of e1_data_pull.ipynb: IMF PPP base year 2017 -> 2021."""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'
CELL_IDX = 4
OLD = '"Purchasing power parity; 2017 international dollar"'
NEW = '"Purchasing power parity; 2021 international dollar"'

with open(NB) as f:
    nb = json.load(f)

src = ''.join(nb['cells'][CELL_IDX].get('source', []))

if OLD not in src:
    if NEW in src:
        print('Cell 4 already patched. No-op.')
        sys.exit(0)
    print(f'ERROR: expected unit string not found in cell {CELL_IDX}.')
    sys.exit(1)

new_src = src.replace(OLD, NEW)
nb['cells'][CELL_IDX]['source'] = new_src.splitlines(keepends=True)
nb['cells'][CELL_IDX]['outputs'] = []
nb['cells'][CELL_IDX]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched cell {CELL_IDX}: PPP base year 2017 -> 2021.')
