"""Dump the v2 clustering function so I know the input features and k."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/4_Clustering_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
for i in [4, 6, 8]:
    src = ''.join(nb['cells'][i].get('source', []))
    print('=' * 70)
    print(f'CELL [{i}]')
    print('=' * 70)
    print(src)
    print()
