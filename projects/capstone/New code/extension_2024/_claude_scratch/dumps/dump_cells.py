"""Dump source of specific cells from e1_data_pull.ipynb."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'
with open(NB) as f:
    nb = json.load(f)

WANTED = [4, 14, 16, 18, 24, 28, 30]
for i in WANTED:
    c = nb['cells'][i]
    src = ''.join(c.get('source', []))
    print('=' * 70)
    print(f'CELL [{i}]   ({c["cell_type"]}, {len(src.splitlines())} lines)')
    print('=' * 70)
    print(src)
    print()
