"""Dump load_gas_prices (cell 22) and load_consolidated_prices (cell 23) to verify
unit handling."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e0_NR_extraction.ipynb'
with open(NB) as f:
    nb = json.load(f)
# Find by content
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'load_gas_prices' in src and 'def load_gas_prices' in src:
        print('=' * 70)
        print(f'CELL [{i}] -- load_gas_prices')
        print('=' * 70)
        print(src)
        print()
    if 'load_consolidated_prices' in src and 'def load_consolidated_prices' in src:
        print('=' * 70)
        print(f'CELL [{i}] -- load_consolidated_prices')
        print('=' * 70)
        print(src)
        print()
