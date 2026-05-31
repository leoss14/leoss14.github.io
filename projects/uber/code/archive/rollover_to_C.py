"""
rollover_to_C.py

One-shot rollover from the lat/lon-only canonical (A) to the hybrid
clustering (C, lat/lon + OD profile PCA).

Steps:
  1. Backup current zone_clusters_canonical.csv -> _OLD_lat_lon_only.csv
  2. Write the new canonical from zone_clusters_compare_ABC.csv (C column)
  3. Rebuild outputs/sample/op_clusters.html using the new canonical
     (heatmap/bubbles/basemap toggles, same visual style as before)
  4. Build outputs/sample/op_clusters_od_only.html for the appendix
     (option B, OD-only, single choropleth view)

Run regen_k4_charts.py and regen_k4_appendix.py separately afterwards to
rebuild the per-cluster charts in Parts 2/3/appendix.
"""
from __future__ import annotations
import sys
import shutil
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

K = 4
CLUSTER_COLORS = [PALETTE['navy'], PALETTE['rose'], PALETTE['gold'], PALETTE['sage']]
CLUSTER_NAMES = {0: 'Manhattan', 1: 'Brooklyn', 2: 'Bronx + Upper Manhattan', 3: 'Queens'}

# ---------------------------------------------------------------------------
# Step 1+2: Backup and overwrite canonical
# ---------------------------------------------------------------------------
print('[1/4] Backing up old canonical and writing new one...')

old_canon = TABLES / 'zone_clusters_canonical.csv'
backup = TABLES / 'zone_clusters_canonical_OLD_lat_lon_only.csv'
if old_canon.exists() and not backup.exists():
    shutil.copy(old_canon, backup)
    print(f'  backed up to {backup.name}')

compare = pd.read_csv(TABLES / 'zone_clusters_compare_ABC.csv')

# New canonical mirrors the old schema: zone_id, zone, borough, latitude,
# longitude, total_pickups, cluster, cluster_name
new_canon = pd.DataFrame({
    'zone_id':       compare['zone_id'],
    'zone':          compare['zone_name'],
    'borough':       compare['borough'],
    'latitude':      compare['latitude'],
    'longitude':     compare['longitude'],
    'total_pickups': compare['pickups'],
    'cluster':       compare['C_hybrid'].astype(int),
    'cluster_name':  compare['C_hybrid_name'],
})
new_canon.to_csv(old_canon, index=False)
print(f'  wrote new canonical: {len(new_canon)} zones, K=4 hybrid')
print(f'  cluster sizes: {dict(new_canon["cluster_name"].value_counts())}')

# ---------------------------------------------------------------------------
# Step 3: Build new Part 1 map (op_clusters.html) using new canonical
# ---------------------------------------------------------------------------
print('[2/4] Loading shapefile and merging new canonical...')
zones_gdf = gpd.read_file(DATA / 'taxi_zones' / 'taxi_zones.shp').to_crs(epsg=4326)
zones_gdf = zones_gdf.rename(columns={'LocationID': 'zone_id'})
zones_gdf = zones_gdf.merge(new_canon[['zone_id','cluster','cluster_name','total_pickups']],
                            on='zone_id', how='left')
valid = zones_gdf.dropna(subset=['cluster']).copy()
valid['cluster'] = valid['cluster'].astype(int)

# Centroids for bubble positions
centroids = pd.read_csv(DATA / 'zone_centroids.csv').drop_duplicates(subset='zone_id', keep='first')
valid = valid.merge(centroids[['zone_id','latitude','longitude']], on='zone_id', how='left')

# Bubble sizes on a log scale, mapped to 3-22 px
pickups = valid['total_pickups'].clip(lower=1)
log_v = np.log10(pickups)
log_min, log_max = log_v.min(), log_v.max()
valid['bubble_size'] = 3 + 19 * (log_v - log_min) / (log_max - log_min)

borough_outlines = valid.dissolve(by='borough', as_index=False)[['borough','geometry']]
zones_geojson = json.loads(valid.set_index('zone_id')[['geometry']].to_json())
boroughs_geojson = json.loads(borough_outlines.reset_index(drop=True)[['geometry']].to_json())

print('[3/4] Building op_clusters.html (K=4 hybrid, heatmap/bubbles toggles)...')
fig = go.Figure()

# Trace 0: choropleth (zones coloured by cluster)
fig.add_trace(go.Choroplethmapbox(
    geojson=zones_geojson,
    locations=valid['zone_id'],
    z=valid['cluster'],
    zmin=0, zmax=K-1,
    colorscale=[[i/(K-1), CLUSTER_COLORS[i]] for i in range(K)],
    showscale=False,
    marker_line_color='rgba(255,255,255,0.7)',
    marker_line_width=0.3,
    marker_opacity=0.55,
    customdata=valid[['zone','borough','total_pickups','cluster_name']],
    hovertemplate=(
        '<b>%{customdata[0]}</b> (%{customdata[1]})<br>'
        'Pickups: %{customdata[2]:,.0f}<br>'
        'Cluster: %{customdata[3]}<extra></extra>'
    ),
    name='choropleth',
    visible=True,
))

# Trace 1: borough outlines
n_b = len(borough_outlines)
fig.add_trace(go.Choroplethmapbox(
    geojson=boroughs_geojson,
    locations=list(range(n_b)),
    z=[0]*n_b,
    colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
    showscale=False,
    marker_line_color='#1a1a1a',
    marker_line_width=2,
    hoverinfo='skip',
    name='boroughs',
    visible=True,
))

# Trace 2: volume bubbles (sized by log pickups)
bubble_colors = [CLUSTER_COLORS[c] for c in valid['cluster']]
fig.add_trace(go.Scattermapbox(
    lat=valid['latitude'], lon=valid['longitude'],
    mode='markers',
    marker=dict(size=valid['bubble_size'], color=bubble_colors, opacity=0.75, sizemode='diameter'),
    customdata=valid[['zone','borough','total_pickups','cluster_name']],
    hovertemplate=(
        '<b>%{customdata[0]}</b> (%{customdata[1]})<br>'
        'Pickups: %{customdata[2]:,.0f}<br>'
        'Cluster: %{customdata[3]}<extra></extra>'
    ),
    name='bubbles',
    visible=False,
))

view_modes = {
    'heatmap': [True, True, False],
    'bubbles': [False, True, True],
    'both':    [True, True, True],
}

fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_zoom=9.4,
    mapbox_center={'lat': 40.715, 'lon': -73.95},
    margin=dict(l=0, r=0, t=0, b=0),
    height=720,
    updatemenus=[
        dict(
            type='buttons', direction='right',
            x=0.01, xanchor='left', y=1.02, yanchor='bottom',
            showactive=True,
            buttons=[
                dict(label='Basemap on', method='relayout',
                     args=[{'mapbox.style': 'carto-positron'}]),
                dict(label='Basemap off', method='relayout',
                     args=[{'mapbox.style': 'white-bg'}]),
            ],
            font=dict(size=11), bgcolor='white', bordercolor='#9aa3b2',
            pad=dict(r=4, l=4, t=4, b=4),
        ),
        dict(
            type='buttons', direction='right',
            x=0.99, xanchor='right', y=1.02, yanchor='bottom',
            showactive=True, active=0,
            buttons=[
                dict(label='Heatmap', method='update',
                     args=[{'visible': view_modes['heatmap']}]),
                dict(label='Bubbles', method='update',
                     args=[{'visible': view_modes['bubbles']}]),
                dict(label='Both', method='update',
                     args=[{'visible': view_modes['both']}]),
            ],
            font=dict(size=11), bgcolor='white', bordercolor='#9aa3b2',
            pad=dict(r=4, l=4, t=4, b=4),
        ),
    ],
)

OUT_PART1 = CHART_DIR / 'op_clusters.html'
save_chart(fig, OUT_PART1)
print(f'  wrote {OUT_PART1.name}')

# ---------------------------------------------------------------------------
# Step 4: Build appendix B map (OD-only)
# ---------------------------------------------------------------------------
print('[4/4] Building op_clusters_od_only.html for appendix...')

# Merge B labels onto the gdf
zones_gdf_b = gpd.read_file(DATA / 'taxi_zones' / 'taxi_zones.shp').to_crs(epsg=4326)
zones_gdf_b = zones_gdf_b.rename(columns={'LocationID': 'zone_id'})
zones_gdf_b = zones_gdf_b.merge(
    compare[['zone_id','B_od_only','B_od_only_name','pickups','zone_name']],
    on='zone_id', how='left',
)
valid_b = zones_gdf_b.dropna(subset=['B_od_only']).copy()
valid_b['B_od_only'] = valid_b['B_od_only'].astype(int)

zones_geojson_b = json.loads(valid_b.set_index('zone_id')[['geometry']].to_json())

fig_b = go.Figure()
fig_b.add_trace(go.Choroplethmapbox(
    geojson=zones_geojson_b,
    locations=valid_b['zone_id'],
    z=valid_b['B_od_only'],
    zmin=0, zmax=K-1,
    colorscale=[[i/(K-1), CLUSTER_COLORS[i]] for i in range(K)],
    showscale=False,
    marker_line_color='rgba(255,255,255,0.7)',
    marker_line_width=0.3,
    marker_opacity=0.62,
    customdata=valid_b[['zone_name','borough','pickups','B_od_only_name']],
    hovertemplate=(
        '<b>%{customdata[0]}</b> (%{customdata[1]})<br>'
        'Pickups: %{customdata[2]:,.0f}<br>'
        'OD-only cluster: %{customdata[3]}<extra></extra>'
    ),
    name='B',
    visible=True,
))
# Borough outlines
fig_b.add_trace(go.Choroplethmapbox(
    geojson=boroughs_geojson,
    locations=list(range(n_b)),
    z=[0]*n_b,
    colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
    showscale=False,
    marker_line_color='#1a1a1a',
    marker_line_width=2,
    hoverinfo='skip',
    name='boroughs',
    visible=True,
))
fig_b.update_layout(
    mapbox_style='carto-positron',
    mapbox_zoom=9.4,
    mapbox_center={'lat': 40.715, 'lon': -73.95},
    margin=dict(l=0, r=0, t=0, b=0),
    height=580,
)
OUT_APPENDIX_B = CHART_DIR / 'op_clusters_od_only.html'
save_chart(fig_b, OUT_APPENDIX_B)
print(f'  wrote {OUT_APPENDIX_B.name}')

print()
print('Done. Next: run regen_k4_charts.py and regen_k4_appendix.py to rebuild downstream charts.')
