"""Dump cells 9 (source priority lists) and 25 (combine_and_clean) and 31 (run)
from e0_NR_extraction.ipynb so I know exactly where to wire the Pink Sheet."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e0_NR_extraction.ipynb'
with open(NB) as f:
    nb = json.load(f)
for i in [9, 25]:
    src = ''.join(nb['cells'][i].get('source', []))
    print('=' * 70)
    print(f'CELL [{i}]')
    print('=' * 70)
    print(src)
    print()
