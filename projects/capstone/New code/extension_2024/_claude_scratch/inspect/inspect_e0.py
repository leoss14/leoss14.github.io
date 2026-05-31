"""Map out e0_NR_extraction.ipynb structure: inputs, outputs, year coverage."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e0_NR_extraction.ipynb'
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
        for kw, lbl in [('read_csv','READ-CSV'), ('read_excel','READ-XL'),
                        ('to_csv','WRITE-CSV'), ('to_parquet','WRITE-PQT'),
                        ('nr_production','NR-CACHE'),
                        ('capstone-client-submission','LOCAL'),
                        ('githubusercontent','REMOTE'),
                        ('rawdata','RAWDATA'),
                        ('intermediary','INTERMED'),
                        ('Statistical Review','EI'), ('Minerals','MIN'),
                        ('Oil Gas','OGC')]:
            if kw in src:
                tags.append(lbl)
        print(f'[{i:2d}] PY {tags}  ({len(lines)}ln) {first}')
