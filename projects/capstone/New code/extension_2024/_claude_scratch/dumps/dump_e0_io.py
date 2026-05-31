"""Show what e0 writes and where."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e0_NR_extraction.ipynb'
with open(NB) as f:
    nb = json.load(f)
for i in [3, 31, 33]:
    src = ''.join(nb['cells'][i].get('source', []))
    print('=' * 70)
    print(f'CELL [{i}]')
    print('=' * 70)
    print(src)
    print()
