"""
Fix the residual file-path bug in cells 31, 32, 33 of e3_clusters.ipynb.

Cells 31-33 do:  pd.read_csv(INTER / 'Master_v2_clusters.csv')
which reads the gross clusters file even when MODE='net'. Should be wrapped
in _suffix() like the writes in cells 22 and 29 already are.

Also fix the hard-coded HS_NAMES dict in cell 32 which references hsXX_share
keys; need to ensure they continue to use the original gross names since
the dict is just for display labels (we don't want 'hsXX_share_net' label
text leaking into the map hover tooltips).
"""
import json
from pathlib import Path

NB = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb')

with open(NB) as f:
    nb = json.load(f)

patched = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    new = src

    # Wrap any read_csv('Master_v2_clusters.csv') in _suffix()
    for fname in ['Master_v2_clusters.csv', 'Master_v2_clusters_RR.csv',
                  'Master_v2_clusters_RR_silhouette.csv']:
        if f"'{fname}'" in new and f"_suffix('{fname}')" not in new:
            new = new.replace(f"'{fname}'", f"_suffix('{fname}')")
        if f'"{fname}"' in new and f'_suffix("{fname}")' not in new:
            new = new.replace(f'"{fname}"', f'_suffix("{fname}")')

    if new != src:
        cell['source'] = new.splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = None
        patched.append(i)

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Patched cells: {patched}')
print(f'Re-run cells {patched} (and any downstream) to refresh outputs with the correct MODE-aware filename.')
