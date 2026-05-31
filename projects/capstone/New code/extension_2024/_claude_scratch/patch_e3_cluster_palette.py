"""Patch cell 32 of e3_clusters.ipynb:
  - swap CLUSTER_COLOURS to match the page's cluster-description box palette
  - add CLUSTER_NAMES so the legend shows readable names
  - drop the embedded fig title (the HTML iframe already has an <h4>)
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb')

nb = json.load(open(NB_PATH))
cell = nb['cells'][32]
src = ''.join(cell['source'])

# --- replacement 1: CLUSTER_COLOURS palette ---
old_palette = """# Support up to k=5
CLUSTER_COLOURS = {
    0: '#4a6fa5',  # steel blue
    1: '#c23a3a',  # red
    2: '#2e7d4a',  # green
    3: '#d97706',  # amber
    4: '#7c3aed',  # purple
}
NON_RR_COLOUR = '#e5e7eb'"""

new_palette = """# Cluster palette matched to the cluster-description boxes on page_new.html.
# Cluster id -> name -> page box hex:
#   0 Mixed Extractives          #2A9D8F  (teal)
#   1 Hydrocarbon Petrostates    #457B9D  (blue)
#   2 Base-Metals Exporters      #E63946  (red)
#   3 Precious Metals and Stones #E9C46A  (yellow)
CLUSTER_COLOURS = {
    0: '#2A9D8F',
    1: '#457B9D',
    2: '#E63946',
    3: '#E9C46A',
    4: '#7c3aed',  # placeholder for k=5 explorations
}
CLUSTER_NAMES = {
    0: 'Mixed Extractives',
    1: 'Hydrocarbon Petrostates',
    2: 'Base-Metals Exporters',
    3: 'Precious Metals and Stones',
    4: 'Cluster 4',
}
NON_RR_COLOUR = '#e5e7eb'"""

if old_palette not in src:
    raise SystemExit('Could not find old palette block in cell 32')
src = src.replace(old_palette, new_palette)

# --- replacement 2: legend trace name ---
old_legend = "name=f'Cluster {cid}', showlegend=True,"
new_legend = "name=CLUSTER_NAMES.get(cid, f'Cluster {cid}'), showlegend=True,"
if old_legend not in src:
    raise SystemExit('Could not find legend trace name line in cell 32')
src = src.replace(old_legend, new_legend)

# --- replacement 3: drop the embedded title ---
old_layout = """fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=50, b=10),"""
new_layout = """fig.update_layout(
        title=None,
        margin=dict(l=10, r=10, t=10, b=10),"""
if old_layout not in src:
    raise SystemExit('Could not find layout block in cell 32')
src = src.replace(old_layout, new_layout)

cell['source'] = src.splitlines(keepends=True)
nb['cells'][32] = cell

NB_PATH.with_suffix('.ipynb.bak_before_cluster_palette').write_text(NB_PATH.read_text())
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Patched: {NB_PATH}')
print(f'Backup:  {NB_PATH}.bak_before_cluster_palette')
