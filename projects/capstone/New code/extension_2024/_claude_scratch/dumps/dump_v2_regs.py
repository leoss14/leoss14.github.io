"""Dump v2 regression specs."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/6_Regressions_Unified.ipynb'
with open(NB) as f:
    nb = json.load(f)
for i in [13, 15, 17, 19, 26]:
    print('=' * 70)
    print(f'CELL [{i}]')
    print('=' * 70)
    print(''.join(nb['cells'][i].get('source', [])))
    print()
