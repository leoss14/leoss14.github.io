"""Locate the 1995 high-income filter logic in v2 codebase."""
import json, os
candidates = [
    '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/3_Imputing_FINAL.ipynb',
    '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/1_cleaning_master_data_FINAL.ipynb',
    '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/0_NR_extraction_FINAL.ipynb',
]
patterns = ['high', 'HIC', 'income', 'IncomeGroup', 'income_group', 'INX']

for f in candidates:
    if not os.path.exists(f):
        continue
    print('=' * 70)
    print(os.path.basename(f))
    print('=' * 70)
    with open(f) as nb_f:
        nb = json.load(nb_f)
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        for p in patterns:
            if p.lower() in src.lower():
                # print just the lines mentioning it
                for ln in src.splitlines():
                    if p.lower() in ln.lower():
                        print(f'  [cell {i:2d}] {ln.strip()[:120]}')
                break
    print()
