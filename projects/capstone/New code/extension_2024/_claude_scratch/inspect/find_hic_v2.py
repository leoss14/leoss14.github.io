"""Locate the 1995 high-income exclusion logic in v2."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/1_cleaning_master_data_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
print(f'cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'high' in src.lower() and ('income' in src.lower() or 'HIC' in src):
        print(f'--- cell {i} ({c["cell_type"]}) ---')
        print(src[:2000])
        print()
