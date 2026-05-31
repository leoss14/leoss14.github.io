"""Dump specific imputation cells."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/3_Imputing_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
WANTED = [23, 25, 27, 28, 30, 35, 36]
for i in WANTED:
    c = nb['cells'][i]
    src = ''.join(c.get('source', []))
    print('=' * 70)
    print(f'CELL [{i}] ({c["cell_type"]}, {len(src.splitlines())} lines)')
    print('=' * 70)
    print(src)
    print()
