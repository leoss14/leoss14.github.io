"""Inspect v2 regression notebook: specifications, FE structure, DV."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/6_Regressions_Unified.ipynb'
with open(NB) as f:
    nb = json.load(f)
print(f'cells: {len(nb["cells"])}')
print()
# Headers and key code lines
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        head = src.split('\n')[0][:90]
        print(f'[{i:2d}] MD {head}')
    else:
        lines = src.split('\n')
        first = next((ln for ln in lines if ln.strip() and not ln.strip().startswith('#')), '')[:80]
        tags = []
        for kw, lbl in [('PanelOLS','PNL'), ('OLS', 'OLS'),
                        ('entity_effects','FE-i'), ('time_effects','FE-t'),
                        ('cluster_entity','CL-i'), ('cluster_time','CL-t'),
                        ('formula', 'FORM'), ('iv_2sls', 'IV'),
                        ('Driscoll', 'DRSC'), ('robust', 'ROB'),
                        ('to_csv','OUT'), ('lag', 'LAG'),
                        ('summary','SUM')]:
            if kw.lower() in src.lower():
                tags.append(lbl)
        print(f'[{i:2d}] PY {tags} {first}')
