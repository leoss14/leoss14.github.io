"""Locate WB high-income / sample-selection logic in 3_Imputing_FINAL.ipynb."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/3_Imputing_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lower = src.lower()
    if 'high' in lower and 'income' in lower:
        print(f'--- cell {i} ---')
        print(src)
        print()
    elif 'HIC' in src or 'high_income' in src or 'income_group' in src:
        print(f'--- cell {i} (HIC) ---')
        print(src)
        print()
