"""Regenerate the four RR_k4 cluster maps using the updated palette/names
from cell 32 of e3_clusters.ipynb.

This is a standalone runner: reads from intermediary/Master_v2_clusters.csv
and one imputation panel, writes Graphics/NB3/map_RR_k4_*.html. Avoids
re-executing the whole e3 notebook.
"""
import os, sys
from pathlib import Path

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
os.chdir(EXT)
sys.path.insert(0, str(EXT))

import pandas as pd
import plotly.graph_objects as go

import _config as cfg  # noqa: F401
from _mice_pool import iter_imputations

INTER    = EXT / 'intermediary'
GRAPHICS = EXT / 'Graphics' / 'NB3'
GRAPHICS.mkdir(parents=True, exist_ok=True)

clusters = pd.read_csv(INTER / 'Master_v2_clusters.csv')
panel0   = next(iter_imputations())[1]

HS_NAMES = {
    'hs25_share': 'Salt/sulfur/cement (25)', 'hs26_share': 'Ores, slag, ash (26)',
    'hs27_share': 'Hydrocarbons (27)',       'hs28_share': 'Inorganic chem. (28)',
    'hs29_share': 'Organic chem. (29)',      'hs44_share': 'Wood (44)',
    'hs71_share': 'Precious metals (71)',    'hs72_share': 'Iron & steel (72)',
    'hs74_share': 'Copper (74)',             'hs75_share': 'Nickel (75)',
    'hs76_share': 'Aluminium (76)',          'hs78_share': 'Lead (78)',
    'hs79_share': 'Zinc (79)',               'hs80_share': 'Tin (80)',
    'hs81_share': 'Other metals (81)',
}
AGG_SHARES = ['coal_share', 'crude_oil_share', 'refined_oil_share', 'gas_share',
              'ores_share', 'base_metals_share', 'precious_share']
HS_CHAPTER_SHARES = list(HS_NAMES.keys())

CLUSTER_COLOURS = {
    0: '#2A9D8F', 1: '#457B9D', 2: '#E63946', 3: '#E9C46A', 4: '#7c3aed',
}
CLUSTER_NAMES = {
    0: 'Mixed Extractives',
    1: 'Hydrocarbon Petrostates',
    2: 'Base-Metals Exporters',
    3: 'Precious Metals and Stones',
    4: 'Cluster 4',
}
NON_RR_COLOUR = '#e5e7eb'


def build_colorscale(k):
    n_bands = k + 1
    scale = []
    for i in range(n_bands):
        lo = i / n_bands
        hi = (i + 1) / n_bands
        col = NON_RR_COLOUR if i == 0 else CLUSTER_COLOURS[i - 1]
        scale.append([lo, col])
        scale.append([hi, col])
    return scale


def build_hover_features(snapshot):
    if snapshot == 'aggregate':
        df = panel0.groupby('Country Code')[AGG_SHARES + HS_CHAPTER_SHARES].mean()
    else:
        year = int(snapshot)
        df = panel0[panel0['Year'] == year].set_index('Country Code')[AGG_SHARES + HS_CHAPTER_SHARES]
    df = df.reset_index()

    rr_codes = set(clusters[clusters['Resource_Rich'] == True]['Country Code'])
    rr_df = df[df['Country Code'].isin(rr_codes)].set_index('Country Code')
    for col in AGG_SHARES:
        pct = rr_df[col].rank(pct=True) * 100
        df = df.merge(pct.rename(f'{col}_pct').reset_index(),
                      on='Country Code', how='left')

    def top2(row):
        vals = row[HS_CHAPTER_SHARES].sort_values(ascending=False).head(2)
        return ', '.join([f'{HS_NAMES[v]} ({row[v]*100:.1f}%)' for v in vals.index])
    df['top2_hs'] = df.apply(top2, axis=1)
    return df


def map_one_snapshot(snapshot_col, snapshot_key, k, save_html=None):
    hover_features = build_hover_features(snapshot_key)
    df = clusters[['Country Code', 'Country Name', snapshot_col]].copy()
    df = df.merge(hover_features, on='Country Code', how='left')
    df['cluster_int'] = df[snapshot_col]

    SHARE_LABELS = {
        'coal_share': 'Coal',
        'crude_oil_share': 'Crude oil',
        'refined_oil_share': 'Refined oil',
        'gas_share': 'Gas',
        'ores_share': 'Ores',
        'base_metals_share': 'Base metals',
        'precious_share': 'Precious metals',
    }

    def hover_row(r):
        if pd.isna(r['cluster_int']):
            return f"<b>{r['Country Name']}</b> ({r['Country Code']})<br>Not resource-rich"
        name = CLUSTER_NAMES.get(int(r['cluster_int']), f"Cluster {int(r['cluster_int'])}")
        wide = sum(r[k] for k in AGG_SHARES if pd.notna(r[k]))
        # top 2 commodities by share
        ranked = sorted(
            [(SHARE_LABELS[k], r[k]) for k in AGG_SHARES if pd.notna(r[k]) and r[k] > 0],
            key=lambda kv: -kv[1]
        )[:2]
        top_str = ', '.join(f'{lbl.lower()} {v*100:.0f}%' for lbl, v in ranked)
        return f"<b>{r['Country Name']}</b> ({r['Country Code']})<br>{name}<br>Resource share {wide*100:.0f}% ({top_str})"

    df['hover'] = df.apply(hover_row, axis=1)

    fig = go.Figure(go.Choropleth(
        locations=df['Country Code'],
        z=df['cluster_int'].fillna(-1),
        text=df['hover'],
        hovertemplate='%{text}<extra></extra>',
        colorscale=build_colorscale(k),
        zmin=-1, zmax=k - 1,
        showscale=False,
    ))

    for cid in range(k):
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode='markers',
            marker=dict(size=12, color=CLUSTER_COLOURS[cid]),
            name=CLUSTER_NAMES.get(cid, f'Cluster {cid}'),
            showlegend=True,
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
        title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=-0.05,
                    xanchor='center', x=0.5,
                    font=dict(family='Public Sans, sans-serif', size=12)),
        height=550,
        font=dict(family='Public Sans, sans-serif', color='#1a2744'),
        paper_bgcolor='white',
    )
    if save_html:
        fig.write_html(save_html, include_plotlyjs='cdn')
    return fig


SNAPSHOTS = [('aggregate', 'Aggregate 1995-2023'),
             ('1995', '1995'), ('2010', '2010'), ('2023', '2023')]
for snap_key, snap_label in SNAPSHOTS:
    snap_col = f'RR_k4_{snap_key}'
    out_html = GRAPHICS / f'map_{snap_col}.html'
    map_one_snapshot(snap_col, snap_key, 4, save_html=out_html)
    print(f'Wrote: {out_html}')
