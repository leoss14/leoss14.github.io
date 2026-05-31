"""
cluster_compare_b_c.py

Compare three clustering schemes for the 259 NYC TLC pickup zones used in
the Uber/Lyft analysis page:

  A. canonical:  K-means on (lat, lon), weighted by zone pickup volume.
                 This is the clustering currently used on the page.
  B. OD-only:    K-means on PCA components of the row-normalised origin-
                 destination profile (where my pickups go + where my
                 dropoffs come from), weighted by volume.
  C. hybrid:     K-means on standardised (lat, lon) concatenated with the
                 PCA components, weighted by volume.

All three use K = 4 and the same seed.

Outputs:
  outputs/tables/zone_clusters_compare_ABC.csv
  printed summary to stdout
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
DATA = ROOT / 'data'

K = 4
SEED = 42
N_PCA = 8

print('[1/8] Loading inputs...')
canonical = pd.read_csv(TABLES / 'zone_clusters_canonical.csv').drop_duplicates(subset='zone_id', keep='first')
centroids = pd.read_csv(DATA / 'zone_centroids.csv').drop_duplicates(subset='zone_id', keep='first')
print(f'  canonical: {len(canonical)} zones, centroids: {len(centroids)} zones')

print('[2/8] Loading 8.34M-trip sample (PU, DO, weight)...')
df = pd.read_parquet(TABLES / 'trip_sample_full.parquet',
                     columns=['PULocationID','DOLocationID','sampling_weight'])
df = df.dropna(subset=['PULocationID','DOLocationID'])
df['PULocationID'] = df['PULocationID'].astype(int)
df['DOLocationID'] = df['DOLocationID'].astype(int)
print(f'  {len(df):,} trips')

print('[3/8] Building OD matrix...')
valid = set(centroids['zone_id']) - {1}  # exclude Newark per page convention
df = df[df['PULocationID'].isin(valid) & df['DOLocationID'].isin(valid)]
od_long = df.groupby(['PULocationID','DOLocationID'])['sampling_weight'].sum().reset_index()
od = od_long.pivot(index='PULocationID', columns='DOLocationID', values='sampling_weight').fillna(0)
zone_index = sorted(set(od.index) & set(od.columns))
od = od.reindex(index=zone_index, columns=zone_index, fill_value=0)
print(f'  OD matrix: {od.shape}, weighted trips: {od.values.sum():,.0f}')

print('[4/8] Building row-normalised PU and DO profiles, concatenating...')
row_pu = od.sum(axis=1)
row_do = od.T.sum(axis=1)
pu_profile = od.div(row_pu.replace(0, 1), axis=0).fillna(0)
do_profile = od.T.div(row_do.replace(0, 1), axis=0).fillna(0)
profile = pd.concat([pu_profile.add_prefix('pu_'), do_profile.add_prefix('do_')], axis=1)
print(f'  profile shape: {profile.shape}')

print(f'[5/8] PCA to {N_PCA} components...')
pca = PCA(n_components=N_PCA, random_state=SEED)
Z = pca.fit_transform(profile.values)
print(f'  explained variance per component: {pca.explained_variance_ratio_.round(3).tolist()}')
print(f'  cumulative:                       {pca.explained_variance_ratio_.cumsum().round(3).tolist()}')

centroids_idx = centroids.set_index('zone_id')
geom = centroids_idx.loc[profile.index, ['latitude','longitude']]
w = row_pu.loc[profile.index].values.astype(float)
w = np.maximum(w, 1.0)

print('[6/8] Fitting B and C...')
# B: OD-only
Xb = StandardScaler().fit_transform(Z)
km_b = KMeans(n_clusters=K, random_state=SEED, n_init=20)
labels_b = km_b.fit_predict(Xb, sample_weight=w)

# C: hybrid (lat, lon, Z), standardised so each feature has unit variance
Xc = StandardScaler().fit_transform(np.concatenate([geom.values, Z], axis=1))
km_c = KMeans(n_clusters=K, random_state=SEED, n_init=20)
labels_c = km_c.fit_predict(Xc, sample_weight=w)

zone_to_A = dict(zip(canonical['zone_id'], canonical['cluster']))
labels_a = np.array([zone_to_A.get(z, -1) for z in profile.index])

def align(labels_new, labels_ref, wts):
    cost = np.zeros((K, K))
    for i in range(K):
        mi = labels_ref == i
        for j in range(K):
            mj = labels_new == j
            cost[i, j] = -wts[mi & mj].sum()
    _, ci = linear_sum_assignment(cost)
    remap = {ci[i]: i for i in range(K)}
    return np.array([remap[l] for l in labels_new])

lb = align(labels_b, labels_a, w)
lc = align(labels_c, labels_a, w)

print('[7/8] Building comparison table...')
result = pd.DataFrame({
    'zone_id': profile.index,
    'A_canonical': labels_a,
    'B_od_only':   lb,
    'C_hybrid':    lc,
    'pickups':     w,
})
result = result.merge(centroids[['zone_id','zone_name','borough','latitude','longitude']], on='zone_id')

cluster_names = {0:'Manhattan', 1:'Brooklyn', 2:'Bronx+UM', 3:'Queens'}
for col in ['A_canonical','B_od_only','C_hybrid']:
    result[f'{col}_name'] = result[col].map(cluster_names)

result.to_csv(TABLES / 'zone_clusters_compare_ABC.csv', index=False)
print(f'  wrote zone_clusters_compare_ABC.csv ({len(result)} zones)')

print('[8/8] Summary statistics:')
print()
total_w = w.sum()
agree_b = (result['A_canonical'] == result['B_od_only']).sum()
agree_c = (result['A_canonical'] == result['C_hybrid']).sum()
agree_b_w = w[result['A_canonical'] == result['B_od_only']].sum() / total_w
agree_c_w = w[result['A_canonical'] == result['C_hybrid']].sum() / total_w
print(f'AGREEMENT WITH CANONICAL (A):')
print(f'  Option B (OD-only):       {agree_b:3d}/{len(result)} zones = {100*agree_b/len(result):4.1f}%   |   volume-weighted: {100*agree_b_w:4.1f}%')
print(f'  Option C (lat/lon + OD):  {agree_c:3d}/{len(result)} zones = {100*agree_c/len(result):4.1f}%   |   volume-weighted: {100*agree_c_w:4.1f}%')
print()

print('CLUSTER SIZES (zones, share of pickup volume):')
print(f'  {"":18s}  {"A canonical":18s}  {"B OD-only":18s}  {"C hybrid":18s}')
for k in range(K):
    a_n = (result['A_canonical']==k).sum(); a_v = w[result['A_canonical']==k].sum()/total_w
    b_n = (result['B_od_only']==k).sum();   b_v = w[result['B_od_only']==k].sum()/total_w
    c_n = (result['C_hybrid']==k).sum();    c_v = w[result['C_hybrid']==k].sum()/total_w
    print(f'  {cluster_names[k]:18s}  {a_n:3d} ({100*a_v:4.1f}%)        {b_n:3d} ({100*b_v:4.1f}%)        {c_n:3d} ({100*c_v:4.1f}%)')
print()

print('SPOTLIGHT ZONES:')
spot_ids = {132:'JFK', 138:'LaGuardia', 100:'Garment District', 161:'Midtown Center',
            230:'Times Sq/Theatre', 79:'East Village', 13:'Battery Park', 261:'World Trade Center',
            74:'East Harlem N', 75:'East Harlem S', 41:'Central Harlem', 42:'Central Harlem N',
            186:'Penn Station/Madison Sq', 90:'Flatiron', 246:'West Chelsea',
            263:'Yorkville W', 262:'Yorkville E', 244:'Washington Heights N',
            243:'Washington Heights S', 116:'Hamilton Heights', 152:'Manhattanville',
            120:'Highbridge Park', 153:'Marble Hill', 127:'Inwood',
            7:'Astoria', 226:'Sunnyside', 196:'Rego Park',
            255:'Williamsburg', 33:'Brooklyn Heights', 65:'Downtown Brooklyn'}
spot = result[result['zone_id'].isin(spot_ids.keys())].copy()
spot['name'] = spot['zone_id'].map(spot_ids)
spot = spot.sort_values('A_canonical')
print(spot[['zone_id','name','borough','A_canonical_name','B_od_only_name','C_hybrid_name']].to_string(index=False))
print()

print('ZONES WHERE B DIFFERS FROM A (top 20 by volume):')
diff_b = result[result['A_canonical'] != result['B_od_only']].sort_values('pickups', ascending=False).head(20)
print(diff_b[['zone_id','zone_name','borough','A_canonical_name','B_od_only_name','pickups']].assign(
    pickups=lambda d: d['pickups'].map('{:,.0f}'.format)).to_string(index=False))
print()

print('ZONES WHERE C DIFFERS FROM A (top 20 by volume):')
diff_c = result[result['A_canonical'] != result['C_hybrid']].sort_values('pickups', ascending=False).head(20)
print(diff_c[['zone_id','zone_name','borough','A_canonical_name','C_hybrid_name','pickups']].assign(
    pickups=lambda d: d['pickups'].map('{:,.0f}'.format)).to_string(index=False))
print()

print('Done.')
