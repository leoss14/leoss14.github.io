"""Dump key cells from v2 ML for spec."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/5_ML_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
for i in [2, 4, 6, 8]:
    src = ''.join(nb['cells'][i].get('source', []))
    print('=' * 70)
    print(f'CELL [{i}]')
    print('=' * 70)
    print(src[:3000])
    print()
