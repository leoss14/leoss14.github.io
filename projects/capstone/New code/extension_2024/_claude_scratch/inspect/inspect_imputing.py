"""Dump structure + key cells from 3_Imputing_FINAL.ipynb."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/3_Imputing_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)

print(f'cells: {len(nb["cells"])}')
print()
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        head = src.split('\n')[0][:90]
        print(f'[{i:2d}] MD  {head}')
    else:
        lines = src.split('\n')
        first = next((ln for ln in lines if ln.strip() and not ln.strip().startswith('#')), '')[:80]
        tags = []
        for kw, lbl in [('interpolate','INTERP'), ('fillna','FILLNA'), ('ffill','FFILL'),
                        ('bfill','BFILL'), ('KNNImputer','KNN'), ('IterativeImputer','ITER'),
                        ('SimpleImputer','SIMPLE'), ('groupby','GROUPBY'),
                        ('isna','ISNA'), ('dropna','DROPNA'),
                        ('to_csv','WRITE'), ('read_csv','READ')]:
            if kw in src:
                tags.append(lbl)
        print(f'[{i:2d}] PY {tags}  ({len(lines)}ln) {first}')
