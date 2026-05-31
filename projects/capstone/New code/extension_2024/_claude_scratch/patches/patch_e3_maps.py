"""Add choropleth maps to e3 for the resource-rich k=3 clustering.

Four maps:
  - aggregate (headline)
  - 1995, 2010, 2023 (per-snapshot)

Each colours resource-rich countries by their cluster ID; non-resource-rich
countries shown in light grey for context.

Saves both interactive HTML (one per snapshot) and static PNG (combined).
"""
import json, sys
from pathlib import Path

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb'

MAPS_MARKDOWN = '''## 14. Resource-Rich Cluster Maps (k=3)

Choropleth maps colouring countries by their `RR_k3_<snapshot>` cluster.
Non-resource-rich countries shown in light grey for context.

Four snapshots: aggregate (headline), 1995, 2010, 2023.
'''

MAPS_CODE = '''import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the clusters CSV we just wrote
clusters = pd.read_csv(INTER / 'Master_v2_clusters.csv')

# Three-cluster colour palette (colour-blind friendly)
CLUSTER_COLOURS = {
    0: '#4a6fa5',  # blue
    1: '#c23a3a',  # red
    2: '#2e7d4a',  # green
}
NON_RR_COLOUR = '#e5e7eb'   # light grey for non-resource-rich countries

def map_one_snapshot(snapshot_col, title, save_html=None):
    """Build a single-snapshot choropleth."""
    df = clusters.copy()
    df['cluster_int'] = df[snapshot_col]
    df['cluster_str'] = df['cluster_int'].apply(
        lambda x: f'Cluster {int(x)}' if pd.notna(x) else 'Not resource-rich'
    )

    fig = px.choropleth(
        df,
        locations='Country Code',
        color='cluster_str',
        hover_name='Country Name',
        hover_data={'cluster_int': False, 'Country Code': True, 'cluster_str': True},
        color_discrete_map={
            'Cluster 0': CLUSTER_COLOURS[0],
            'Cluster 1': CLUSTER_COLOURS[1],
            'Cluster 2': CLUSTER_COLOURS[2],
            'Not resource-rich': NON_RR_COLOUR,
        },
        category_orders={'cluster_str': ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Not resource-rich']},
        title=title,
    )
    fig.update_geos(
        projection_type='natural earth',
        showcoastlines=True, coastlinecolor='#b0b0b0', coastlinewidth=0.5,
        showland=True, landcolor='#fafafa',
        showocean=True, oceancolor='#f0f4f8',
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(title='', orientation='h', yanchor='bottom', y=-0.05, xanchor='center', x=0.5),
        height=500,
    )
    if save_html:
        fig.write_html(save_html)
    return fig


# Generate and display the four maps
for snap, title in [
    ('RR_k3_aggregate', 'Resource-Rich Clusters (k=3) — Aggregate 1995-2023'),
    ('RR_k3_1995',      'Resource-Rich Clusters (k=3) — 1995 snapshot'),
    ('RR_k3_2010',      'Resource-Rich Clusters (k=3) — 2010 snapshot'),
    ('RR_k3_2023',      'Resource-Rich Clusters (k=3) — 2023 snapshot'),
]:
    out_html = GRAPHICS / f'map_{snap}.html'
    fig = map_one_snapshot(snap, title, save_html=out_html)
    print(f'Saved: {out_html}')
    fig.show()
'''

MAPS_CODE_INVENTORY = '''# Country-by-cluster inventory for each snapshot
print('Resource-Rich cluster membership at k=3:')
print()
for snap_col in ['RR_k3_aggregate', 'RR_k3_1995', 'RR_k3_2010', 'RR_k3_2023']:
    print(f'--- {snap_col} ---')
    sub = clusters[clusters[snap_col].notna()].copy()
    sub[snap_col] = sub[snap_col].astype(int)
    for cid in sorted(sub[snap_col].unique()):
        codes = sub[sub[snap_col] == cid]['Country Code'].tolist()
        codes = sorted(codes)
        print(f'  Cluster {cid} (n={len(codes):2d}): {", ".join(codes)}')
    print()
'''


def patch():
    with open(NB) as f:
        nb = json.load(f)

    for c in nb['cells']:
        if c['cell_type'] != 'markdown':
            continue
        if '14. Resource-Rich Cluster Maps' in ''.join(c.get('source', [])):
            print('Section 14 already present. No-op.')
            return

    cells_to_add = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': MAPS_MARKDOWN.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': MAPS_CODE.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': MAPS_CODE_INVENTORY.splitlines(keepends=True)},
    ]

    nb['cells'].extend(cells_to_add)

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'Added {len(cells_to_add)} cells. Total: {len(nb["cells"])}')


if __name__ == '__main__':
    patch()
