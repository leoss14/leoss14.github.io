# %% [markdown]
# # Uber NYC Ride-Hailing: Geographic Evolution 2018-2025
#
# Comparative spatial analysis of Uber pickup and dropoff patterns across New York City
# taxi zones, using K-means clustering on geographic coordinates, Lorenz curve concentration
# measures, and Local Indicators of Spatial Association (LISA) to quantify how ride-hailing
# demand has redistributed over seven years.

# %% [markdown]
# ---
# ## 1. Setup and Configuration

# %%
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, levene, entropy
from math import radians, cos, sin, asin, sqrt
import geopandas as gpd
import libpysal
from esda.moran import Moran, Moran_Local
import json
import gc
import os

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = '/Users/leoss/Desktop/Portfolio/Website-/projects/uber/'
DATA_DIR = BASE_DIR + 'data/'
OUTPUT_DIR = BASE_DIR + 'outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATH_2018 = DATA_DIR + 'fhv_tripdata_2018-01.parquet'
PATH_2025 = DATA_DIR + 'fhvhv_tripdata_2025-01.parquet'
PATH_CENTROIDS = DATA_DIR + 'zone_centroids.csv'
PATH_SHAPEFILE = DATA_DIR + 'taxi_zones/taxi_zones.shp'

# ── Analysis parameters ───────────────────────────────────────────────────
SAMPLE_SIZE = 20_000_000
N_CLUSTERS = 6
N_DEMAND_CLUSTERS = 4
K_NEAREST = 5
N_BOOTSTRAP = 1000

UBER_2018_BASES = ['B02512', 'B02598', 'B02617', 'B02682', 'B02764', 'B02765', 'B02835', 'B02836']
LYFT_2018_BASES = ['B02510']
UBER_2025_LICENSE = 'HV0003'
LYFT_2025_LICENSE = 'HV0005'

AIRPORT_ZONE_IDS = {132, 138, 1}
AIRPORT_LABELS = {132: 'JFK', 138: 'LaGuardia', 1: 'Newark (EWR)'}

# ── Unified style system ─────────────────────────────────────────────────
STYLE = {
    'font_family': 'IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif',
    'tick_size': 11,
    'axis_title_size': 13,
    'legend_size': 11,
    'template': 'plotly_white',
    'plot_bg': 'rgba(0,0,0,0)',
    'paper_bg': 'white',
    'chart_height': 550,
    'margin_default': dict(l=60, r=40, t=20, b=50),
    'margin_map': dict(l=0, r=0, t=20, b=0),
    'grid_color': '#e5e7eb',
    'grid_width': 0.5,
    'hover_font_size': 13,
    'hover_font_color': '#1a2744',
    # Year comparison palette
    'year_2018': '#ff6b6b',
    'year_2025': '#4ecdc4',
    # Cluster palette (6 clusters)
    'cluster_colors': ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4'],
    # Borough palette
    'borough_colors': {
        'Manhattan': '#4363d8', 'Brooklyn': '#3cb44b', 'Queens': '#f58231',
        'Bronx': '#e6194b', 'Staten Island': '#911eb4', 'EWR': '#42d4f4',
    },
    # LISA palette
    'lisa_colors': {
        'HH': '#d7191c', 'LL': '#2c7bb6', 'HL': '#fdae61',
        'LH': '#abd9e9', 'ns': '#e8e8e8',
    },
    'lisa_labels': {
        'HH': 'High-High (hot spot)', 'LL': 'Low-Low (cold spot)',
        'HL': 'High-Low (outlier)', 'LH': 'Low-High (outlier)',
        'ns': 'Not significant',
    },
    # Map defaults
    'map_style': 'carto-positron-nolabels',
    'map_center': {'lat': 40.7128, 'lon': -73.9352},
    'map_zoom': 9.5,
}

# Demand profile colors
DEMAND_COLORS = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']
OPERATOR_COLORS = {'Uber': '#1a1a2e', 'Lyft': '#ff00bf', 'Other FHV': '#888888'}

print(f"Configuration:")
print(f"  Sample size: {SAMPLE_SIZE:,} trips per year")
print(f"  Clusters: {N_CLUSTERS}")
print(f"  Output: {OUTPUT_DIR}")


# %% [markdown]
# ---
# ## 2. Helper Functions

# %%
# ── Layout helpers ────────────────────────────────────────────────────────

def base_layout(height=None, width=None, **kwargs):
    """Standard layout applied to every chart."""
    layout = dict(
        title='',
        font=dict(family=STYLE['font_family']),
        template=STYLE['template'],
        plot_bgcolor=STYLE['plot_bg'],
        paper_bgcolor=STYLE['paper_bg'],
        height=height or STYLE['chart_height'],
        margin=STYLE['margin_default'],
        hoverlabel=dict(
            font_size=STYLE['hover_font_size'],
            font_color=STYLE['hover_font_color'],
            bgcolor='white',
            bordercolor='#ccc',
        ),
    )
    if width:
        layout['width'] = width
    layout.update(kwargs)
    return layout


def styled_axis(**kwargs):
    """Standard axis styling."""
    return dict(
        tickfont=dict(size=STYLE['tick_size']),
        title_font=dict(size=STYLE['axis_title_size']),
        gridcolor=STYLE['grid_color'],
        gridwidth=STYLE['grid_width'],
        **kwargs,
    )


def save_html(fig, filename):
    """Save figure as CDN-loaded HTML with mode bar suppressed."""
    fig.write_html(
        OUTPUT_DIR + filename,
        include_plotlyjs='cdn',
        config={'displayModeBar': False},
    )
    print(f"  Saved: {filename}")


def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color to rgba string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def add_bg_layer(fig):
    """Add grey background taxi zone layer to a map figure."""
    fig.add_trace(go.Choroplethmap(
        geojson=taxi_zones_geo_4326,
        locations=all_zone_ids,
        featureidkey='properties.LocationID',
        z=[1] * len(all_zone_ids),
        colorscale=[[0, '#e8e8e8'], [1, '#e8e8e8']],
        marker_opacity=0.4, marker_line_width=0.3, marker_line_color='#ccc',
        showscale=False, hoverinfo='skip',
    ))


def filter_geojson(zone_id_set):
    """Filter global taxi_zones_geo_4326 to a subset of zone IDs (as strings)."""
    return {
        'type': 'FeatureCollection',
        'features': [f for f in taxi_zones_geo_4326['features']
                     if f['properties']['LocationID'] in zone_id_set]
    }


# ── Geographic / statistical helpers ──────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def haversine_km_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance (numpy arrays, returns km)."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def name_cluster_by_location(center_lat, center_lon, zone_centroids):
    """Name cluster based on nearest major zone."""
    dists = np.sqrt(
        (zone_centroids['latitude'] - center_lat) ** 2 +
        (zone_centroids['longitude'] - center_lon) ** 2
    )
    nearest = zone_centroids.iloc[dists.idxmin()]
    return f"{nearest['borough']}: {nearest['zone_name']}"


def match_clusters_by_proximity(centers_a, centers_b):
    """Match clusters by geographic proximity (greedy nearest, cityblock)."""
    dists = cdist(centers_a, centers_b, metric='cityblock')
    matches = {}
    used = set()
    pairs = sorted(
        [(i, j, dists[i, j]) for i in range(len(centers_a)) for j in range(len(centers_b))],
        key=lambda x: x[2]
    )
    for i, j, d in pairs:
        if i not in matches and j not in used:
            matches[i] = j
            used.add(j)
    return matches


def merge_zone_info(df, zone_col, centroids, prefix):
    """Merge zone centroid info onto a dataframe."""
    merged = df.merge(
        centroids[['zone_id', 'zone_name', 'borough', 'latitude', 'longitude']],
        left_on=zone_col, right_on='zone_id', how='left'
    )
    return merged.rename(columns={
        'zone_id': f'{prefix}_zone_id',
        'zone_name': f'{prefix}_zone_name',
        'borough': f'{prefix}_borough',
        'latitude': f'{prefix}_lat',
        'longitude': f'{prefix}_lon',
    })


def short_label(name):
    """Extract borough from 'Borough: Zone Name' format."""
    return name.split(':')[0].strip()


def make_short_labels(cluster_names, n_clusters):
    """Create short labels, appending (2), (3) for duplicate boroughs."""
    labels = {}
    seen = {}
    for i in range(n_clusters):
        base = short_label(cluster_names[i])
        if base in seen:
            seen[base] += 1
            labels[i] = f"{base} ({seen[base]})"
        else:
            seen[base] = 1
            labels[i] = base
    return labels


def lorenz_data(zone_counts_series):
    """Return (cumulative share of zones, cumulative share of trips)."""
    vals = zone_counts_series.sort_values().values
    cum_zones = np.arange(1, len(vals) + 1) / len(vals)
    cum_trips = np.cumsum(vals) / vals.sum()
    return cum_zones, cum_trips


def gini_from_lorenz(cum_zones, cum_trips):
    """Gini coefficient via trapezoidal integration under the Lorenz curve."""
    return 1 - 2 * np.trapezoid(cum_trips, cum_zones)


def mismatch_ratios(df, pu_col='PU_zone_id', do_col='DO_zone_id'):
    """Compute PU/(PU+DO) ratio per zone. 0.5 = perfectly balanced."""
    pu = df.groupby(pu_col).size().rename('pu')
    do = df.dropna(subset=[do_col]).groupby(do_col).size().rename('do')
    combined = pd.concat([pu, do], axis=1).fillna(0)
    combined['ratio'] = combined['pu'] / (combined['pu'] + combined['do'])
    combined['total'] = combined['pu'] + combined['do']
    return combined


def hourly_profile(df, filter_col=None, filter_val=None):
    """Return normalized 24-bin hourly distribution."""
    if filter_col is not None:
        df = df[df[filter_col] == filter_val]
    counts = df.groupby('hour').size().reindex(range(24), fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.ones(24) / 24
    return (counts / total).values


def jsd(p, q):
    """Jensen-Shannon divergence (symmetric, bounded [0, 1] with log2)."""
    eps = 1e-12
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))


def pivot_od(df, n):
    """Build cluster-to-cluster OD share matrix (%)."""
    valid = df.dropna(subset=['PU_cluster', 'DO_cluster']).copy()
    valid['PU_cluster'] = valid['PU_cluster'].astype(int)
    valid['DO_cluster'] = valid['DO_cluster'].astype(int)
    od = valid.groupby(['PU_cluster', 'DO_cluster']).size().reset_index(name='trips')
    total = od['trips'].sum()
    matrix = np.zeros((n, n))
    for _, r in od.iterrows():
        if int(r['PU_cluster']) < n and int(r['DO_cluster']) < n:
            matrix[int(r['PU_cluster']), int(r['DO_cluster'])] = 100 * r['trips'] / total
    return matrix


def compute_zone_localization(df, nearest_zones_dict,
                               pu_zone_col='PU_zone_id', do_zone_col='DO_zone_id'):
    """For each PU zone, fraction of trips where DO is in same or K nearest zones."""
    valid = df.dropna(subset=[pu_zone_col, do_zone_col]).copy()
    valid[pu_zone_col] = valid[pu_zone_col].astype(int)
    valid[do_zone_col] = valid[do_zone_col].astype(int)
    od_counts = valid.groupby([pu_zone_col, do_zone_col]).size().reset_index(name='trips')
    od_counts['is_local'] = od_counts.apply(
        lambda r: int(r[do_zone_col]) in nearest_zones_dict.get(int(r[pu_zone_col]), set()),
        axis=1
    )
    total_by_zone = od_counts.groupby(pu_zone_col)['trips'].sum().rename('total')
    local_by_zone = od_counts[od_counts['is_local']].groupby(pu_zone_col)['trips'].sum().rename('local')
    zone_local = pd.concat([total_by_zone, local_by_zone], axis=1).fillna(0)
    zone_local['local_share'] = 100 * zone_local['local'] / zone_local['total']
    return zone_local


def bootstrap_gini(df, zone_col='PU_zone_id', n_boot=N_BOOTSTRAP):
    """Bootstrap Gini by resampling zone-level trip counts."""
    zone_counts = df.groupby(zone_col).size().values
    n_zones = len(zone_counts)
    rng = np.random.RandomState(42)
    ginis = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(zone_counts, size=n_zones, replace=True)
        sorted_counts = np.sort(sampled)
        cz = np.arange(1, n_zones + 1) / n_zones
        ct = np.cumsum(sorted_counts) / sorted_counts.sum()
        ginis[b] = 1 - 2 * np.trapezoid(ct, cz)
    return ginis


# %% [markdown]
# ---
# ## 3. Load Zone Centroids, Shapefile, and Spatial Weights

# %%
zone_centroids = pd.read_csv(PATH_CENTROIDS)
print(f"Loaded {len(zone_centroids)} zone centroids")

gdf_raw = gpd.read_file(PATH_SHAPEFILE).to_crs(epsg=4326)
taxi_zones_geo_4326 = json.loads(gdf_raw.to_json())
for f in taxi_zones_geo_4326['features']:
    f['properties']['LocationID'] = str(int(f['properties']['LocationID']))
all_zone_ids = [f['properties']['LocationID'] for f in taxi_zones_geo_4326['features']]
print(f"Loaded {len(all_zone_ids)} taxi zone geometries")

# Spatial weights (used by LISA and Moran's I) - computed once
w = libpysal.weights.KNN.from_dataframe(gdf_raw, k=6)
w.transform = 'r'
gdf_base = gdf_raw.copy()
gdf_base['LocationID_int'] = gdf_base['LocationID'].astype(int)

# Zone centroid lookup (used by many charts)
zc_lookup = zone_centroids.drop_duplicates(subset='zone_id').set_index('zone_id')

# K-nearest zones for localization metric
zc_deduped = zone_centroids.drop_duplicates(subset='zone_id').copy().set_index('zone_id')
zone_ids_all_int = zc_deduped.index.values
zone_coords_arr = zc_deduped[['latitude', 'longitude']].values
zone_dist_matrix = cdist(zone_coords_arr, zone_coords_arr, metric='euclidean')
nearest_zones = {}
for i, zid in enumerate(zone_ids_all_int):
    sorted_idx = np.argsort(zone_dist_matrix[i])
    nearest_zones[zid] = set([zone_ids_all_int[j] for j in sorted_idx[1:K_NEAREST + 1]]) | {zid}

print(f"Spatial weights: KNN k=6, {len(gdf_raw)} geometries")
print(f"K-nearest zones: K={K_NEAREST}, {len(nearest_zones)} zones")


# %% [markdown]
# ---
# ## 4. Load and Process Trip Data (Once)
#
# Both parquet files are read exactly once. Filtering, sampling, temporal features,
# zone merging, and K-means clustering happen here and nowhere else.

# %%
def load_year(path, year, filter_col, filter_val, pu_col, do_col):
    """Load, filter to Uber, sample, add temporal features, merge zones, cluster."""
    print(f"\n[Loading {year}] {os.path.basename(path)}")
    # First pass: count total rows
    table_count = pq.read_table(path, columns=[])
    total = table_count.num_rows
    print(f"  Total rows: {total:,}")

    # Second pass: read needed columns
    cols = ['pickup_datetime', pu_col, do_col, filter_col]
    table = pq.read_table(path, columns=cols)
    df = table.to_pandas()

    # Filter to Uber
    if isinstance(filter_val, list):
        df = df[df[filter_col].isin(filter_val)].copy()
    else:
        df = df[df[filter_col] == filter_val].copy()
    uber_count = len(df)
    print(f"  Uber trips: {uber_count:,} ({100 * uber_count / total:.1f}%)")

    # Sample
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    # Temporal features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_name'] = df['pickup_datetime'].dt.day_name()

    # Pickup zone merge
    df = df.dropna(subset=[pu_col])
    df[pu_col] = df[pu_col].astype(int)
    df = merge_zone_info(df, pu_col, zone_centroids, 'PU')
    df = df.dropna(subset=['PU_lat', 'PU_lon'])

    # Dropoff zone merge
    df[do_col] = pd.to_numeric(df[do_col], errors='coerce')
    df.loc[df[do_col].notna(), do_col] = df.loc[df[do_col].notna(), do_col].astype(int)
    df = merge_zone_info(df, do_col, zone_centroids, 'DO')

    # K-means clustering on pickup coordinates
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['PU_cluster'] = kmeans.fit_predict(df[['PU_lat', 'PU_lon']].values)
    cnames = {i: name_cluster_by_location(*kmeans.cluster_centers_[i], zone_centroids)
              for i in range(N_CLUSTERS)}
    df['PU_cluster_name'] = df['PU_cluster'].map(cnames)

    # Predict dropoff clusters
    do_mask = df['DO_lat'].notna() & df['DO_lon'].notna()
    df.loc[do_mask, 'DO_cluster'] = kmeans.predict(
        df.loc[do_mask, ['DO_lat', 'DO_lon']].values)
    df['DO_cluster_name'] = df['DO_cluster'].map(cnames)

    # Trip distances (haversine between PU and DO centroids)
    valid_dist = df['DO_lat'].notna() & df['DO_lon'].notna()
    df.loc[valid_dist, 'trip_dist_km'] = haversine_km_vectorized(
        df.loc[valid_dist, 'PU_lat'].values, df.loc[valid_dist, 'PU_lon'].values,
        df.loc[valid_dist, 'DO_lat'].values, df.loc[valid_dist, 'DO_lon'].values,
    )

    print(f"  Final sample: {len(df):,} trips")
    for i in range(N_CLUSTERS):
        count = (df['PU_cluster'] == i).sum()
        pct = 100 * count / len(df)
        print(f"    {cnames[i]}: {count:>10,} ({pct:>5.1f}%)")

    del table, table_count
    gc.collect()

    return df, kmeans, cnames, total, uber_count


# Load both years
df_2018, kmeans_2018, cnames_2018, total_2018, uber_n_2018 = load_year(
    PATH_2018, 2018, 'dispatching_base_num', UBER_2018_BASES,
    'PUlocationID', 'DOlocationID'
)
df_2025, kmeans_2025, cnames_2025, total_2025, uber_n_2025 = load_year(
    PATH_2025, 2025, 'hvfhs_license_num', UBER_2025_LICENSE,
    'PULocationID', 'DOLocationID'
)


# %% [markdown]
# ---
# ## 5. Cross-Year Cluster Matching and Derived Quantities
#
# Everything computed here is reused across many charts.

# %%
# ── Cluster matching ─────────────────────────────────────────────────────
centers_2018 = kmeans_2018.cluster_centers_
centers_2025 = kmeans_2025.cluster_centers_
cluster_matches = match_clusters_by_proximity(centers_2018, centers_2025)

print("Cluster matches (2018 -> 2025):")
for i18, i25 in sorted(cluster_matches.items()):
    dist = haversine_km(*centers_2018[i18], *centers_2025[i25])
    print(f"  {cnames_2018[i18]} -> {cnames_2025[i25]} ({dist:.2f} km)")

# ── Short labels ─────────────────────────────────────────────────────────
short_labels_2018 = make_short_labels(cnames_2018, N_CLUSTERS)
short_labels_2025 = make_short_labels(cnames_2025, N_CLUSTERS)

# ── Color maps (matched clusters share same color) ───────────────────────
cluster_color_map_2018 = {
    cnames_2018[i]: STYLE['cluster_colors'][i % len(STYLE['cluster_colors'])]
    for i in range(N_CLUSTERS)
}
cluster_color_map_2025 = {
    cnames_2025[i25]: STYLE['cluster_colors'][i18 % len(STYLE['cluster_colors'])]
    for i18, i25 in cluster_matches.items()
}

# ── Centroid shift data ──────────────────────────────────────────────────
shift_data = []
for i18, i25 in cluster_matches.items():
    lat1, lon1 = centers_2018[i18]
    lat2, lon2 = centers_2025[i25]
    shift_data.append({
        'idx_18': i18, 'idx_25': i25,
        'name_18': cnames_2018[i18], 'name_25': cnames_2025[i25],
        'dist_km': haversine_km(lat1, lon1, lat2, lon2)
    })
avg_shift = np.mean([s['dist_km'] for s in shift_data])

# ── Borough aggregates ───────────────────────────────────────────────────
borough_2018 = df_2018.groupby('PU_borough').size().reset_index(name='trips')
borough_2025 = df_2025.groupby('PU_borough').size().reset_index(name='trips')
borough_2018['pct'] = 100 * borough_2018['trips'] / borough_2018['trips'].sum()
borough_2025['pct'] = 100 * borough_2025['trips'] / borough_2025['trips'].sum()
borough_df = borough_2018[['PU_borough', 'pct']].rename(columns={'pct': 'pct_2018'}).merge(
    borough_2025[['PU_borough', 'pct']].rename(columns={'pct': 'pct_2025'}), on='PU_borough'
)

# ── Zone-level pickup shares ─────────────────────────────────────────────
pu_counts_2018 = df_2018.groupby(['PU_zone_id', 'PU_zone_name', 'PU_borough']).size().reset_index(name='count')
pu_counts_2025 = df_2025.groupby(['PU_zone_id', 'PU_zone_name', 'PU_borough']).size().reset_index(name='count')
pu_counts_2018['share'] = 100 * pu_counts_2018['count'] / pu_counts_2018['count'].sum()
pu_counts_2025['share'] = 100 * pu_counts_2025['count'] / pu_counts_2025['count'].sum()

zone_change = pu_counts_2018[['PU_zone_id', 'PU_zone_name', 'PU_borough', 'share']].rename(
    columns={'share': 'share_2018', 'PU_zone_name': 'zone_name', 'PU_borough': 'borough'}
).merge(
    pu_counts_2025[['PU_zone_id', 'share']].rename(columns={'share': 'share_2025'}),
    on='PU_zone_id', how='outer'
).fillna(0)
zone_change['share_change'] = zone_change['share_2025'] - zone_change['share_2018']
zone_change['zone_id'] = zone_change['PU_zone_id'].astype(float).astype(int).astype(str)
cap_pu = zone_change['share_change'].abs().quantile(0.95)
zone_change['share_change_capped'] = zone_change['share_change'].clip(-cap_pu, cap_pu)
zone_change['zone_name'] = zone_change['zone_id'].astype(int).map(zc_lookup['zone_name'])
zone_change['borough'] = zone_change['zone_id'].astype(int).map(zc_lookup['borough'])

# ── Dropoff share change ─────────────────────────────────────────────────
do_counts_2018 = df_2018.dropna(subset=['DO_zone_id']).groupby('DO_zone_id').size().reset_index(name='count_2018')
do_counts_2025 = df_2025.dropna(subset=['DO_zone_id']).groupby('DO_zone_id').size().reset_index(name='count_2025')
do_comparison = do_counts_2018.merge(do_counts_2025, left_on='DO_zone_id', right_on='DO_zone_id', how='outer').fillna(0)
do_comparison['share_2018'] = 100 * do_comparison['count_2018'] / do_comparison['count_2018'].sum()
do_comparison['share_2025'] = 100 * do_comparison['count_2025'] / do_comparison['count_2025'].sum()
do_comparison['share_change'] = do_comparison['share_2025'] - do_comparison['share_2018']
do_comparison['zone_id'] = do_comparison['DO_zone_id'].astype(float).astype(int).astype(str)
do_comparison['zone_name'] = do_comparison['DO_zone_id'].astype(int).map(zc_lookup['zone_name'])
do_comparison['borough'] = do_comparison['DO_zone_id'].astype(int).map(zc_lookup['borough'])
cap_do = do_comparison['share_change'].abs().quantile(0.95)
do_comparison['share_change_capped'] = do_comparison['share_change'].clip(-cap_do, cap_do)

# ── Gini + Lorenz ────────────────────────────────────────────────────────
pu_zone_counts_2018 = df_2018.groupby('PU_zone_id').size()
pu_zone_counts_2025 = df_2025.groupby('PU_zone_id').size()
cz_18, ct_18 = lorenz_data(pu_zone_counts_2018)
cz_25, ct_25 = lorenz_data(pu_zone_counts_2025)
gini_18 = gini_from_lorenz(cz_18, ct_18)
gini_25 = gini_from_lorenz(cz_25, ct_25)
print(f"\nGini: {gini_18:.4f} (2018) -> {gini_25:.4f} (2025)")

# ── Bootstrap Gini CIs ──────────────────────────────────────────────────
print("Bootstrapping Gini CIs...")
gini_boot_2018 = bootstrap_gini(df_2018)
gini_boot_2025 = bootstrap_gini(df_2025)
ci_2018 = np.percentile(gini_boot_2018, [2.5, 97.5])
ci_2025 = np.percentile(gini_boot_2025, [2.5, 97.5])
print(f"  2018: {gini_18:.4f} [{ci_2018[0]:.4f}, {ci_2018[1]:.4f}]")
print(f"  2025: {gini_25:.4f} [{ci_2025[0]:.4f}, {ci_2025[1]:.4f}]")

# ── Moran's I ────────────────────────────────────────────────────────────
def compute_morans(df_year):
    zone_counts = df_year.groupby('PU_zone_id').size()
    gdf_tmp = gdf_base.copy()
    gdf_tmp['trips'] = gdf_tmp['LocationID_int'].map(zone_counts).fillna(0)
    gdf_tmp['trip_share'] = 100 * gdf_tmp['trips'] / gdf_tmp['trips'].sum()
    mi = Moran(gdf_tmp['trip_share'].values, w)
    return mi.I, mi.p_sim

mi_2018, mi_p_2018 = compute_morans(df_2018)
mi_2025, mi_p_2025 = compute_morans(df_2025)
print(f"Moran's I: {mi_2018:.4f} (p={mi_p_2018:.4f}) -> {mi_2025:.4f} (p={mi_p_2025:.4f})")

# ── Intra-cluster trip shares ────────────────────────────────────────────
intra_2018 = df_2018.dropna(subset=['PU_cluster', 'DO_cluster'])
intra_2018_pct = 100 * (intra_2018['PU_cluster'] == intra_2018['DO_cluster']).mean()
intra_2025 = df_2025.dropna(subset=['PU_cluster', 'DO_cluster'])
intra_2025_pct = 100 * (intra_2025['PU_cluster'] == intra_2025['DO_cluster']).mean()
print(f"Intra-cluster trips: {intra_2018_pct:.1f}% -> {intra_2025_pct:.1f}%")

# ── Mismatch ratios + tests ─────────────────────────────────────────────
mr_2018 = mismatch_ratios(df_2018)
mr_2025 = mismatch_ratios(df_2025)
ks_stat, ks_p = ks_2samp(mr_2018['ratio'].dropna(), mr_2025['ratio'].dropna())
lev_stat, lev_p = levene(mr_2018['ratio'].dropna(), mr_2025['ratio'].dropna())
print(f"KS mismatch: D={ks_stat:.4f}, p={ks_p:.2e}")
print(f"Levene: W={lev_stat:.4f}, p={lev_p:.2e}")

# ── OD matrices ──────────────────────────────────────────────────────────
od_matrix_2018 = pivot_od(df_2018, N_CLUSTERS)
od_matrix_2025 = pivot_od(df_2025, N_CLUSTERS)
od_2025_aligned = np.zeros_like(od_matrix_2025)
for i18, i25 in cluster_matches.items():
    for j18, j25 in cluster_matches.items():
        od_2025_aligned[i18, j18] = od_matrix_2025[i25, j25]
od_diff = od_2025_aligned - od_matrix_2018

# ── Hourly profiles and JSD ──────────────────────────────────────────────
h18_pct = df_2018.groupby('hour').size()
h25_pct = df_2025.groupby('hour').size()
h18_pct = 100 * h18_pct / h18_pct.sum()
h25_pct = 100 * h25_pct / h25_pct.sum()

agg_jsd = jsd(hourly_profile(df_2018), hourly_profile(df_2025))
cluster_jsds = []
for i in range(N_CLUSTERS):
    i25 = cluster_matches[i]
    d = jsd(hourly_profile(df_2018, 'PU_cluster', i),
            hourly_profile(df_2025, 'PU_cluster', i25))
    cluster_jsds.append({'cluster': short_labels_2018[i], 'jsd': d, 'idx': i})
cjsd_df = pd.DataFrame(cluster_jsds)
print(f"Aggregate hourly JSD: {agg_jsd:.6f}")

# ── Localization ─────────────────────────────────────────────────────────
print("Computing zone localization...")
local_2018 = compute_zone_localization(df_2018, nearest_zones)
local_2025 = compute_zone_localization(df_2025, nearest_zones)
mean_local_2018 = local_2018['local_share'].mean()
mean_local_2025 = local_2025['local_share'].mean()
print(f"  Localization: {mean_local_2018:.1f}% -> {mean_local_2025:.1f}%")

# ── Trip distances (filtered) ────────────────────────────────────────────
d18_valid = df_2018[(df_2018['trip_dist_km'] > 0) & (df_2018['trip_dist_km'] < 100)]['trip_dist_km']
d25_valid = df_2025[(df_2025['trip_dist_km'] > 0) & (df_2025['trip_dist_km'] < 100)]['trip_dist_km']
print(f"Trip distances: median {d18_valid.median():.2f} km (2018) vs {d25_valid.median():.2f} km (2025)")

# ── Zone-level cluster assignments (for maps) ────────────────────────────
def zone_cluster_assignment(df):
    zc = df.groupby('PU_zone_id').agg({
        'PU_cluster': lambda x: x.mode()[0],
        'PU_cluster_name': lambda x: x.mode()[0],
        'PU_zone_name': 'first',
        'PU_borough': 'first'
    }).reset_index()
    zc.columns = ['zone_id', 'cluster', 'cluster_name', 'zone_name', 'borough']
    zc['zone_id'] = zc['zone_id'].astype(float).astype(int).astype(str)
    return zc

zone_clusters_2018 = zone_cluster_assignment(df_2018)
zone_clusters_2025 = zone_cluster_assignment(df_2025)

print("\n" + "=" * 70)
print("ALL PRECOMPUTATIONS COMPLETE")
print("=" * 70)
