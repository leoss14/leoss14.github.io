"""
cluster_compare_b_c_map.py

Builds a single HTML map with three switchable views of the K=4 clustering:
    A. canonical: lat/lon weighted by volume (the current page clustering)
    B. OD-only:   PCA components of the origin-destination profile
    C. hybrid:    standardised (lat, lon) + PCA OD components

Hover on any zone shows the cluster assignment under ALL THREE options at
once, so the differences are visible without switching views. The buttons
above the map switch which clustering is colored in.

Output: outputs/sample/_cluster_compare_b_c.html
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from trip_queries import save_chart, PALETTE

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
DATA = ROOT / 'data'
CHART_DIR = ROOT / 'outputs' / 'sample'

CLUSTER_COLORS = [
    PALETTE['navy'],   # 0 = Manhattan
    PALETTE['rose'],   # 1 = Brooklyn
    PALETTE['gold'],   # 2 = Bronx + Upper Manhattan
    PALETTE['sage'],   # 3 = Queens
]
CLUSTER_NAMES = {0: 'Manhattan', 1: 'Brooklyn', 2: 'Bronx + Upper Manhattan', 3: 'Queens'}
K = 4

# Load comparison table written by cluster_compare_b_c.py
print('Loading inputs...')
result = pd.read_csv(TABLES / 'zone_clusters_compare_ABC.csv')
print(f'  comparison table: {len(result)} zones')

# Load shapefile and merge cluster columns
zones_gdf = gpd.read_file(DATA / 'taxi_zones' / 'taxi_zones.shp').to_crs(epsg=4326)
zones_gdf = zones_gdf.rename(columns={'LocationID': 'zone_id'})
zones_gdf = zones_gdf.merge(
    result[['zone_id','A_canonical','B_od_only','C_hybrid','pickups',
            'A_canonical_name','B_od_only_name','C_hybrid_name']],
    on='zone_id', how='left',
)
valid = zones_gdf.dropna(subset=['A_canonical']).copy()
valid['A_canonical'] = valid['A_canonical'].astype(int)
valid['B_od_only']   = valid['B_od_only'].astype(int)
valid['C_hybrid']    = valid['C_hybrid'].astype(int)
print(f'  matched polygons: {len(valid)}')

# Mark whether each zone agrees with canonical under B and C
valid['B_diff'] = valid['A_canonical'] != valid['B_od_only']
valid['C_diff'] = valid['A_canonical'] != valid['C_hybrid']

# Borough outlines for context
borough_outlines = valid.dissolve(by='borough', as_index=False)[['borough','geometry']]
zones_geojson = json.loads(valid.set_index('zone_id')[['geometry']].to_json())
boroughs_geojson = json.loads(borough_outlines.reset_index(drop=True)[['geometry']].to_json())

# Hover text: shows assignments under all three views simultaneously
def make_hover(row):
    diff_str = ''
    if row['A_canonical'] != row['C_hybrid']:
        diff_str = '<br><i>A and C disagree on this zone</i>'
    return (
        f"<b>{row['zone']}</b> ({row['borough']})<br>"
        f"Pickups: {row['pickups']:,.0f}<br>"
        f"<b>A</b> canonical:&nbsp;&nbsp;{row['A_canonical_name']}<br>"
        f"<b>B</b> OD-only:&nbsp;&nbsp;&nbsp;{row['B_od_only_name']}<br>"
        f"<b>C</b> hybrid:&nbsp;&nbsp;&nbsp;&nbsp;{row['C_hybrid_name']}"
        f"{diff_str}<extra></extra>"
    )
valid['hover'] = valid.apply(make_hover, axis=1)

# Build figure: three choropleth traces (one per view) + one borough-outline trace
print('Building figure...')
fig = go.Figure()

for view_col, view_name in [('A_canonical', 'A'), ('B_od_only', 'B'), ('C_hybrid', 'C')]:
    fig.add_trace(go.Choroplethmapbox(
        geojson=zones_geojson,
        locations=valid['zone_id'],
        z=valid[view_col],
        zmin=0, zmax=K-1,
        colorscale=[[i/(K-1), CLUSTER_COLORS[i]] for i in range(K)],
        showscale=False,
        marker_line_color='rgba(255,255,255,0.7)',
        marker_line_width=0.3,
        marker_opacity=0.62,
        text=valid['hover'],
        hovertemplate='%{text}',
        name=f'view_{view_name}',
        visible=(view_name == 'A'),
    ))

# Borough outlines (always visible)
n_b = len(borough_outlines)
fig.add_trace(go.Choroplethmapbox(
    geojson=boroughs_geojson,
    locations=list(range(n_b)),
    z=[0]*n_b,
    colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
    showscale=False,
    marker_line_color='#1a1a1a',
    marker_line_width=1.8,
    hoverinfo='skip',
    name='boroughs',
    visible=True,
))

# Legend traces: invisible scatter points so the cluster color key shows up
for i in range(K):
    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=12, color=CLUSTER_COLORS[i]),
        name=CLUSTER_NAMES[i],
        showlegend=True,
        hoverinfo='skip',
    ))

# Visibility patterns. Trace order: [A, B, C, boroughs, legend0, legend1, legend2, legend3]
visibility = {
    'A': [True,  False, False, True, True, True, True, True],
    'B': [False, True,  False, True, True, True, True, True],
    'C': [False, False, True,  True, True, True, True, True],
}

fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_zoom=9.4,
    mapbox_center={'lat': 40.715, 'lon': -73.95},
    margin=dict(l=0, r=0, t=40, b=0),
    height=760,
    legend=dict(
        orientation='h',
        yanchor='top', y=1.05,
        xanchor='right', x=0.99,
        bgcolor='rgba(255,255,255,0.92)',
        bordercolor='#dadde6', borderwidth=1,
        font=dict(size=11),
    ),
    updatemenus=[
        dict(
            type='buttons',
            direction='right',
            x=0.01, xanchor='left',
            y=1.05, yanchor='top',
            showactive=True,
            active=0,
            buttons=[
                dict(label='A: canonical (lat/lon, weighted)',
                     method='update', args=[{'visible': visibility['A']}]),
                dict(label='B: OD-only',
                     method='update', args=[{'visible': visibility['B']}]),
                dict(label='C: hybrid (lat/lon + OD)',
                     method='update', args=[{'visible': visibility['C']}]),
            ],
            font=dict(size=11),
            bgcolor='white',
            bordercolor='#9aa3b2',
            pad=dict(r=4, l=4, t=4, b=4),
        ),
    ],
)

OUT = CHART_DIR / '_cluster_compare_b_c.html'
save_chart(fig, OUT)
print(f'Wrote {OUT}')
print(f'Open in a browser: file://{OUT}')
