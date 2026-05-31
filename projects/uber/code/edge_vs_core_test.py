"""
edge_vs_core_test.py

Within each cluster, did the 2019-to-2026 share change concentrate in edge
zones (near another cluster) or core zones (deep in own cluster)?

For each zone:
  - share_2019, share_2026 = zone share of total Uber pickups in that year
  - delta_share = share_2026 - share_2019 (in percentage points)
  - dist_own  = geodesic distance to own cluster's volume-weighted centroid (km)
  - dist_other = min geodesic distance to any other cluster's centroid (km)
  - edge_ness = dist_own / (dist_own + dist_other)
        0.0 = at own centroid (pure core)
        0.5 = equidistant (boundary)
        >0.5 = closer to another cluster than to own

Then within each cluster: correlation between edge_ness and delta_share,
plus the top 5 gainers and top 5 losers by zone.

Output:
  outputs/tables/edge_vs_core_test.csv
  printed summary
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt
from scipy.stats import spearmanr

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
DATA = ROOT / 'data'

LBL = ['Manhattan', 'Brooklyn', 'Bronx + Upper Manhattan', 'Queens']

print('[1/5] Loading inputs...')
canon = pd.read_csv(TABLES / 'zone_clusters_canonical.csv')
centroids = pd.read_csv(DATA / 'zone_centroids.csv').drop_duplicates(subset='zone_id', keep='first')

print('[2/5] Loading Uber trips (PU, year, weight)...')
df = pd.read_parquet(TABLES / 'trip_sample_full.parquet',
    columns=['PULocationID','pickup_datetime','operator','sampling_weight'])
df = df[df['operator']=='Uber'].copy()
df['year'] = df['pickup_datetime'].dt.year
print(f'  {len(df):,} Uber trips')

print('[3/5] Computing per-zone yearly shares...')
z2c = dict(zip(canon['zone_id'], canon['cluster']))
df['cluster'] = df['PULocationID'].map(z2c)
df = df.dropna(subset=['cluster']).copy()
df['cluster'] = df['cluster'].astype(int)

zone_year = df.groupby(['PULocationID','year'])['sampling_weight'].sum().reset_index().rename(columns={'PULocationID':'zone_id'})
year_total = df.groupby('year')['sampling_weight'].sum().to_dict()
zone_year['share'] = zone_year.apply(lambda r: r['sampling_weight'] / year_total[r['year']], axis=1)

share_2019 = zone_year[zone_year['year']==2019].set_index('zone_id')['share']
share_2026 = zone_year[zone_year['year']==2026].set_index('zone_id')['share']

zones_all = sorted(set(share_2019.index) | set(share_2026.index))
zones_df = pd.DataFrame(index=zones_all)
zones_df['share_2019'] = share_2019.reindex(zones_df.index, fill_value=0)
zones_df['share_2026'] = share_2026.reindex(zones_df.index, fill_value=0)
zones_df['delta_pp'] = (zones_df['share_2026'] - zones_df['share_2019']) * 100
zones_df = zones_df.reset_index().rename(columns={'index':'zone_id'})
zones_df = zones_df.merge(canon[['zone_id','zone','borough','cluster','cluster_name','total_pickups']], on='zone_id', how='left')
zones_df = zones_df.merge(centroids[['zone_id','latitude','longitude']], on='zone_id', how='left')
zones_df = zones_df.dropna(subset=['cluster','latitude']).copy()
zones_df['cluster'] = zones_df['cluster'].astype(int)
print(f'  zones with full data: {len(zones_df)}')

print('[4/5] Computing cluster centroids (volume-weighted) and edge-ness...')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2*R*asin(sqrt(d))

# Volume-weighted cluster centroid = pickup-weighted mean of zone centroids
cluster_centroids = {}
for c in range(4):
    g = zones_df[zones_df['cluster']==c]
    w = g['total_pickups']
    lat = (g['latitude'] * w).sum() / w.sum()
    lon = (g['longitude'] * w).sum() / w.sum()
    cluster_centroids[c] = (lat, lon)
    print(f'  cluster {c} ({LBL[c]:24s}): centroid ({lat:.4f}, {lon:.4f})')

def dist_to_centroid(row, c):
    lat_c, lon_c = cluster_centroids[c]
    return haversine(row['latitude'], row['longitude'], lat_c, lon_c)

zones_df['dist_own_km'] = zones_df.apply(lambda r: dist_to_centroid(r, int(r['cluster'])), axis=1)
zones_df['dist_other_km'] = zones_df.apply(
    lambda r: min(dist_to_centroid(r, c) for c in range(4) if c != int(r['cluster'])),
    axis=1
)
zones_df['edge_ness'] = zones_df['dist_own_km'] / (zones_df['dist_own_km'] + zones_df['dist_other_km'])

zones_df.to_csv(TABLES / 'edge_vs_core_test.csv', index=False)
print(f'  wrote edge_vs_core_test.csv ({len(zones_df)} rows)')

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print('\n[5/5] Within-cluster summary:')
print()
for c in range(4):
    g = zones_df[zones_df['cluster']==c].copy()
    rho, p = spearmanr(g['edge_ness'], g['delta_pp'])
    # Split into edge half and core half by median edge_ness within cluster
    med = g['edge_ness'].median()
    edge = g[g['edge_ness'] >= med]
    core = g[g['edge_ness'] <  med]
    print(f'{LBL[c]}  (n={len(g)}, cluster share went {g["share_2019"].sum()*100:.1f}% -> {g["share_2026"].sum()*100:.1f}%)')
    print(f'  Spearman(edge_ness, delta_pp): rho = {rho:+.3f}  (p = {p:.3f})')
    print(f'  EDGE half (edge_ness >= {med:.3f}, n={len(edge)}):  sum delta_pp = {edge["delta_pp"].sum():+.2f}')
    print(f'  CORE half (edge_ness <  {med:.3f}, n={len(core)}):  sum delta_pp = {core["delta_pp"].sum():+.2f}')
    print(f'  Top gainers in cluster:')
    print(g.nlargest(5, 'delta_pp')[['zone','borough','dist_own_km','edge_ness','delta_pp','total_pickups']].to_string(index=False))
    print(f'  Top losers in cluster:')
    print(g.nsmallest(5, 'delta_pp')[['zone','borough','dist_own_km','edge_ness','delta_pp','total_pickups']].to_string(index=False))
    print()

# Overall (across clusters)
overall_rho, overall_p = spearmanr(zones_df['edge_ness'], zones_df['delta_pp'])
print(f'OVERALL Spearman(edge_ness, delta_pp): rho = {overall_rho:+.3f}  (p = {overall_p:.3f})')
