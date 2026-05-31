"""
rebuild_op_clusters_simple.py

Rebuild outputs/sample/op_clusters.html with just the choropleth and
borough outlines, no bubbles, no basemap toggle.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import save_chart, PALETTE

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
DATA = ROOT / 'data'
CHART_DIR = ROOT / 'outputs' / 'sample'

K = 4
CLUSTER_COLORS = [PALETTE['navy'], PALETTE['rose'], PALETTE['gold'], PALETTE['sage']]

print('Loading canonical and shapefile...')
canon = pd.read_csv(TABLES / 'zone_clusters_canonical.csv')

zones_gdf = gpd.read_file(DATA / 'taxi_zones' / 'taxi_zones.shp').to_crs(epsg=4326)
zones_gdf = zones_gdf.rename(columns={'LocationID': 'zone_id'})
zones_gdf = zones_gdf.merge(
    canon[['zone_id','cluster','cluster_name','total_pickups']],
    on='zone_id', how='left',
)
valid = zones_gdf.dropna(subset=['cluster']).copy()
valid['cluster'] = valid['cluster'].astype(int)

borough_outlines = valid.dissolve(by='borough', as_index=False)[['borough','geometry']]
zones_geojson = json.loads(valid.set_index('zone_id')[['geometry']].to_json())
boroughs_geojson = json.loads(borough_outlines.reset_index(drop=True)[['geometry']].to_json())

print('Building figure (choropleth + borough outlines only)...')
fig = go.Figure()

# Choropleth: zones coloured by cluster
fig.add_trace(go.Choroplethmapbox(
    geojson=zones_geojson,
    locations=valid['zone_id'],
    z=valid['cluster'],
    zmin=0, zmax=K-1,
    colorscale=[[i/(K-1), CLUSTER_COLORS[i]] for i in range(K)],
    showscale=False,
    marker_line_color='rgba(255,255,255,0.7)',
    marker_line_width=0.3,
    marker_opacity=0.62,
    customdata=valid[['zone','borough','total_pickups','cluster_name']],
    hovertemplate=(
        '<b>%{customdata[0]}</b> (%{customdata[1]})<br>'
        'Pickups: %{customdata[2]:,.0f}<br>'
        'Cluster: %{customdata[3]}<extra></extra>'
    ),
    name='choropleth',
))

# Borough outlines (black)
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
))

fig.update_layout(
    mapbox_style='carto-positron',
    mapbox_zoom=9.4,
    mapbox_center={'lat': 40.715, 'lon': -73.95},
    margin=dict(l=0, r=0, t=0, b=0),
    height=720,
)

OUT = CHART_DIR / 'op_clusters.html'
save_chart(fig, OUT)
print(f'wrote {OUT.name}')
