"""Replace the maps cell with a richer hover.

Hover now shows:
  - Cluster ID
  - For each of the 4 aggregate shares: value + percentile rank within
    the resource-rich subset
  - The country's top-2 HS chapters by share, with their names
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb'

NEW_MAPS_CODE = '''import plotly.express as px
import plotly.graph_objects as go

# Load the clusters CSV and the imputed data for share values
clusters = pd.read_csv(INTER / 'Master_v2_clusters.csv')
panel0 = next(iter_imputations())[1]

# HS chapter to descriptive name
HS_NAMES = {
    'hs25_share': 'Salt/sulfur/cement (25)',
    'hs26_share': 'Ores, slag, ash (26)',
    'hs27_share': 'Hydrocarbons (27)',
    'hs28_share': 'Inorganic chem. (28)',
    'hs29_share': 'Organic chem. (29)',
    'hs44_share': 'Wood (44)',
    'hs71_share': 'Precious metals (71)',
    'hs72_share': 'Iron & steel (72)',
    'hs74_share': 'Copper (74)',
    'hs75_share': 'Nickel (75)',
    'hs76_share': 'Aluminium (76)',
    'hs78_share': 'Lead (78)',
    'hs79_share': 'Zinc (79)',
    'hs80_share': 'Tin (80)',
    'hs81_share': 'Other metals (81)',
}
AGG_SHARES = ['hydrocarbon_share', 'ores_share', 'base_metals_share', 'precious_share']
HS_CHAPTER_SHARES = list(HS_NAMES.keys())

CLUSTER_COLOURS = {0: '#4a6fa5', 1: '#c23a3a', 2: '#2e7d4a'}
NON_RR_COLOUR = '#e5e7eb'


def build_hover_features(snapshot):
    """For each country, compute aggregate share values + percentile ranks +
    top-2 HS chapters. Returns a DataFrame keyed on Country Code."""
    if snapshot == 'aggregate':
        df = panel0.groupby('Country Code')[AGG_SHARES + HS_CHAPTER_SHARES].mean()
    else:
        year = int(snapshot)
        df = panel0[panel0['Year'] == year].set_index('Country Code')[AGG_SHARES + HS_CHAPTER_SHARES]
    df = df.reset_index()

    # Restrict percentile calculation to resource-rich countries only
    rr_codes = set(clusters[clusters['Resource_Rich'] == True]['Country Code'])
    rr_df = df[df['Country Code'].isin(rr_codes)].set_index('Country Code')

    # Percentile rank of each aggregate share within resource-rich subset
    for col in AGG_SHARES:
        pct = rr_df[col].rank(pct=True) * 100
        df = df.merge(
            pct.rename(f'{col}_pct').reset_index(),
            on='Country Code', how='left',
        )

    # Top-2 HS chapters by share per country
    def top2(row):
        vals = row[HS_CHAPTER_SHARES].sort_values(ascending=False).head(2)
        return ', '.join([f'{HS_NAMES[v]} ({row[v]*100:.1f}%)' for v in vals.index])
    df['top2_hs'] = df.apply(top2, axis=1)

    return df


def map_one_snapshot(snapshot_col, snapshot_key, title, save_html=None):
    """Build a snapshot choropleth with rich hover."""
    hover_features = build_hover_features(snapshot_key)
    df = clusters[['Country Code', 'Country Name', snapshot_col]].copy()
    df = df.merge(hover_features, on='Country Code', how='left')

    df['cluster_int'] = df[snapshot_col]
    df['cluster_str'] = df['cluster_int'].apply(
        lambda x: f'Cluster {int(x)}' if pd.notna(x) else 'Not resource-rich'
    )

    # Construct hover text
    def hover_row(r):
        if pd.isna(r['cluster_int']):
            return f"<b>{r['Country Name']}</b> ({r['Country Code']})<br>Not resource-rich"
        parts = [
            f"<b>{r['Country Name']}</b> ({r['Country Code']})",
            f"Cluster {int(r['cluster_int'])}",
            "",
            "<b>Aggregate shares (% of exports, percentile in RR sample):</b>",
            f"  Hydrocarbons:  {r['hydrocarbon_share']*100:5.1f}%  (pct {r['hydrocarbon_share_pct']:4.1f})",
            f"  Ores:          {r['ores_share']*100:5.1f}%  (pct {r['ores_share_pct']:4.1f})",
            f"  Base metals:   {r['base_metals_share']*100:5.1f}%  (pct {r['base_metals_share_pct']:4.1f})",
            f"  Precious:      {r['precious_share']*100:5.1f}%  (pct {r['precious_share_pct']:4.1f})",
            "",
            f"<b>Top chapters:</b>  {r['top2_hs']}",
        ]
        return "<br>".join(parts)

    df['hover'] = df.apply(hover_row, axis=1)

    fig = go.Figure(go.Choropleth(
        locations=df['Country Code'],
        z=df['cluster_int'].fillna(-1),
        text=df['hover'],
        hovertemplate='%{text}<extra></extra>',
        colorscale=[
            [0.00, NON_RR_COLOUR],
            [0.25, NON_RR_COLOUR],
            [0.25, CLUSTER_COLOURS[0]],
            [0.50, CLUSTER_COLOURS[0]],
            [0.50, CLUSTER_COLOURS[1]],
            [0.75, CLUSTER_COLOURS[1]],
            [0.75, CLUSTER_COLOURS[2]],
            [1.00, CLUSTER_COLOURS[2]],
        ],
        zmin=-1, zmax=2,
        showscale=False,
    ))

    # Legend via separate scatter traces (workaround for choropleth)
    for cid, colour in CLUSTER_COLOURS.items():
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode='markers',
            marker=dict(size=12, color=colour),
            name=f'Cluster {cid}', showlegend=True,
        ))
    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None],
        mode='markers',
        marker=dict(size=12, color=NON_RR_COLOUR),
        name='Not resource-rich', showlegend=True,
    ))

    fig.update_geos(
        projection_type='natural earth',
        showcoastlines=True, coastlinecolor='#b0b0b0', coastlinewidth=0.5,
        showland=True, landcolor='#fafafa',
        showocean=True, oceancolor='#f0f4f8',
    )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.05, xanchor='center', x=0.5),
        height=550,
    )
    if save_html:
        fig.write_html(save_html)
    return fig


# Generate and display the four maps
SNAPSHOTS_TO_MAP = [
    ('RR_k3_aggregate', 'aggregate', 'Resource-Rich Clusters (k=3) — Aggregate 1995-2023'),
    ('RR_k3_1995',      '1995',      'Resource-Rich Clusters (k=3) — 1995'),
    ('RR_k3_2010',      '2010',      'Resource-Rich Clusters (k=3) — 2010'),
    ('RR_k3_2023',      '2023',      'Resource-Rich Clusters (k=3) — 2023'),
]
for snap_col, snap_key, title in SNAPSHOTS_TO_MAP:
    out_html = GRAPHICS / f'map_{snap_col}.html'
    fig = map_one_snapshot(snap_col, snap_key, title, save_html=out_html)
    print(f'Saved: {out_html}')
    fig.show()
'''


def patch():
    with open(NB) as f:
        nb = json.load(f)

    # Find the existing maps cell (the first cell containing 'map_one_snapshot' and 'choropleth')
    target = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'map_one_snapshot' in s and 'choropleth' in s:
            target = i
            break

    if target is None:
        print('ERROR: existing maps cell not found.')
        sys.exit(1)

    nb['cells'][target]['source'] = NEW_MAPS_CODE.splitlines(keepends=True)
    nb['cells'][target]['outputs'] = []
    nb['cells'][target]['execution_count'] = None

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'cell {target}: replaced maps code with rich hover.')


if __name__ == '__main__':
    patch()
