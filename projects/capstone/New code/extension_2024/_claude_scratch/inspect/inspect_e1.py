"""Extract structure of e1_data_pull.ipynb: section headers and source-pull cells."""
import json, re

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'

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
        comments = [ln for ln in lines if ln.strip().startswith('#')][:2]
        tag = ''
        for kw, lbl in [('wbdata','WB-WDI'), ('wb_data','WB-WDI'), ('V-Dem','VDEM'),
                        ('vdem','VDEM'), ('pwt','PWT'), ('WGI','WGI'), ('imf','IMF'),
                        ('weo','IMF-WEO'), ('Atlas','ATLAS'), ('eci','ECI'),
                        ('pd.read_csv','READ-CSV'), ('pd.read_excel','READ-XL'),
                        ('to_csv','WRITE-CSV')]:
            if kw.lower() in src.lower():
                tag += f' [{lbl}]'
        print(f'[{i:2d}] PY{tag}  ({len(lines)} lines)  {first}')
