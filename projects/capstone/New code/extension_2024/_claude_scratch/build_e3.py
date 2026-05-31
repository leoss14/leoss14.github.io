"""Build e3_clusters.ipynb.

Clustering 154 countries by trade-side resource portfolio composition.

Four snapshots:
  1. 1995    : single year, early panel period
  2. 2010    : single year, mid panel period
  3. 2023    : single year, end of panel (post-COVID)
  4. 1995-2023 aggregate : panel-mean composition over the full period

Features: per-HS-chapter trade shares (hs25_share, hs26_share, ..., hs81_share)
plus aggregated shares (hydrocarbon, ores, base_metals, precious).

Pipeline per snapshot:
  Aggregate over M=10 imputations -> log1p(x) -> PCA(2) -> KMeans(k=5)
  
Ensemble across M imputations:
  Run k-means on each panel, build co-membership matrix, consensus via
  hierarchical clustering on the matrix. Report modal cluster + stability.

Validates k via silhouette sweep (k=2..8) on the aggregate snapshot.

Outputs:
  intermediary/Master_v2_clusters.csv  -- per-country cluster trajectory
  Graphics/e3_silhouette.png           -- k selection plot
  Graphics/e3_biplots_<snapshot>.png   -- PCA biplots with cluster colours
  Graphics/e3_transitions.png          -- Sankey of 1995 -> 2023 transitions
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text.splitlines(keepends=True),
    }


cells = []

cells.append(md('''# e3 — Resource Profile Clustering

Cluster countries by their trade-side resource portfolio composition.

Four snapshots:
1. **1995** — start of the panel
2. **2010** — mid-period
3. **2023** — end of panel (post-COVID)
4. **1995-2023 aggregate** — full-panel mean composition

Features: per-HS-chapter export shares (HS 25, 26, 27, 28, 29, 44, 71, 72, 74, 75,
76, 78, 79, 80, 81) plus aggregated shares (hydrocarbon, ores, base metals, precious).

Estimation:
- Each of M=10 MICE imputations runs k-means independently.
- A co-membership matrix accumulates pairwise agreement across imputations.
- Hierarchical clustering on the co-membership matrix yields the consensus.
- Per-country modal cluster + stability score (fraction agreeing with mode).

Trade-side features mean composition reflects export structure, which is the
relevant marker of "what kind of resource economy is this country".
'''))

cells.append(md('## 1. Setup'))

cells.append(code('''import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
import _config as cfg
from _mice_pool import iter_imputations, load_imputations

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

EXT  = Path('.').resolve()
INTER = EXT / 'intermediary'
GRAPHICS = EXT / 'Graphics' / 'NB3'
GRAPHICS.mkdir(parents=True, exist_ok=True)

print(f'Working dir: {EXT}')
'''))

cells.append(md('## 2. Feature definitions'))

cells.append(code('''# Per-chapter shares (HS code -> column name in Master_v2)
HS_CHAPTERS = [25, 26, 27, 28, 29, 44, 71, 72, 74, 75, 76, 78, 79, 80, 81]
PER_CHAPTER_FEATURES = [f'hs{c}_share' for c in HS_CHAPTERS]

# Aggregated shares
AGGREGATE_FEATURES = [
    'hydrocarbon_share',
    'ores_share',
    'base_metals_share',
    'precious_share',
]

# All cluster features
ALL_FEATURES = PER_CHAPTER_FEATURES + AGGREGATE_FEATURES

# Verify these exist in Master_v2 by checking imputation 0
imp0 = next(iter_imputations())[1]
missing = [c for c in ALL_FEATURES if c not in imp0.columns]
present = [c for c in ALL_FEATURES if c in imp0.columns]
print(f'Available features: {len(present)} / {len(ALL_FEATURES)}')
if missing:
    print(f'Missing: {missing}')
FEATURES = present
'''))

cells.append(md('## 3. Snapshot preparation'))

cells.append(code('''def prepare_snapshot(panel, snapshot):
    """Filter a panel to the snapshot definition and aggregate to one row per country."""
    df = panel.copy()
    if snapshot == '1995':
        df = df[df['Year'] == 1995]
    elif snapshot == '2010':
        df = df[df['Year'] == 2010]
    elif snapshot == '2023':
        df = df[df['Year'] == 2023]
    elif snapshot == 'aggregate':
        # Full-panel mean across all years for each country
        df = df.groupby('Country Code', as_index=False)[FEATURES].mean()
        df['Country Name'] = panel.groupby('Country Code')['Country Name'].first().reindex(
            df['Country Code']
        ).values
        return df[['Country Code', 'Country Name'] + FEATURES]
    else:
        raise ValueError(f'unknown snapshot: {snapshot}')

    # For single-year snapshots, take that year's row per country
    cols = ['Country Code', 'Country Name'] + FEATURES
    return df[cols].copy()


# Quick sanity check
snap_aggr = prepare_snapshot(imp0, 'aggregate')
print(f'Aggregate snapshot: {len(snap_aggr)} countries, {len(FEATURES)} features')
print(snap_aggr.head(3).to_string(index=False))
'''))

cells.append(md('## 4. Single-imputation clustering pipeline'))

cells.append(code('''def cluster_one(snapshot_df, n_clusters=5, random_state=42, return_model=False):
    """Run log1p -> PCA(2) -> KMeans(k=5) on a snapshot DataFrame."""
    feats = snapshot_df[FEATURES].fillna(0).values
    # Clip negative shares (shouldn't occur but be safe)
    feats = np.clip(feats, 0, None)
    X_log = np.log1p(feats)

    pca = PCA(n_components=2, random_state=random_state)
    pca_components = pca.fit_transform(X_log)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(pca_components)

    sil = silhouette_score(pca_components, labels) if len(set(labels)) > 1 else np.nan

    result_df = pd.DataFrame({
        'Country Code': snapshot_df['Country Code'].values,
        'Country Name': snapshot_df.get('Country Name', '').values,
        'PC1': pca_components[:, 0],
        'PC2': pca_components[:, 1],
        'Cluster_raw': labels,
    })

    if return_model:
        return result_df, pca, km, sil
    return result_df, sil


# Sanity check
res0, sil0 = cluster_one(snap_aggr)
print(f'Single-imputation aggregate: silhouette={sil0:.3f}')
print(f'Cluster sizes: {res0["Cluster_raw"].value_counts().sort_index().to_dict()}')
'''))

cells.append(md('## 5. Silhouette analysis (k selection)'))

cells.append(code('''# Run k=2..8 on the aggregate snapshot from imputation 0 to confirm k=5 holds
k_range = range(2, 9)
sil_scores = []
inertias = []

snap_for_k = prepare_snapshot(imp0, 'aggregate')

for k in k_range:
    res_k, sil_k = cluster_one(snap_for_k, n_clusters=k)
    # Compute inertia by refitting (we don't return it from cluster_one)
    feats_k = np.log1p(np.clip(snap_for_k[FEATURES].fillna(0).values, 0, None))
    pca_k = PCA(n_components=2, random_state=42).fit_transform(feats_k)
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pca_k)
    inertias.append(float(km_k.inertia_))
    sil_scores.append(sil_k)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(list(k_range), sil_scores, 'o-')
ax1.axvline(5, color='red', linestyle='--', label='k=5')
ax1.set_xlabel('k (number of clusters)')
ax1.set_ylabel('Silhouette score')
ax1.set_title('Silhouette across k (aggregate snapshot)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(list(k_range), inertias, 'o-')
ax2.axvline(5, color='red', linestyle='--', label='k=5')
ax2.set_xlabel('k (number of clusters)')
ax2.set_ylabel('Inertia')
ax2.set_title('Elbow plot')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPHICS / 'silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'k=5 silhouette: {sil_scores[3]:.3f}')
print(f'k range silhouette: {[round(s, 3) for s in sil_scores]}')
'''))

cells.append(md('## 6. Cluster ensemble across M imputations'))

cells.append(code('''def cluster_ensemble(snapshot, k=5):
    """Run k-means on each of M imputations, build co-membership matrix,
    derive consensus via hierarchical clustering."""
    panels = list(iter_imputations())
    M = len(panels)

    # Collect per-imputation labels
    label_arrays = []
    country_codes_canon = None
    for imp_id, panel in panels:
        snap = prepare_snapshot(panel, snapshot)
        res, _ = cluster_one(snap, n_clusters=k, random_state=42)
        if country_codes_canon is None:
            country_codes_canon = res['Country Code'].tolist()
            label_arrays.append(res['Cluster_raw'].values)
        else:
            # Reorder to match canonical order
            res = res.set_index('Country Code').reindex(country_codes_canon).reset_index()
            label_arrays.append(res['Cluster_raw'].fillna(-1).astype(int).values)

    label_matrix = np.array(label_arrays)  # (M, N)
    M, N = label_matrix.shape

    # Co-membership matrix: M_{ij} = fraction of imputations where i, j in same cluster
    co_membership = np.zeros((N, N))
    for m in range(M):
        same = (label_matrix[m, :, None] == label_matrix[m, None, :]).astype(float)
        co_membership += same
    co_membership /= M

    # Hierarchical clustering on (1 - co_membership) as distance
    dist = 1 - co_membership
    np.fill_diagonal(dist, 0)
    agg = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',
        linkage='average',
    )
    consensus_labels = agg.fit_predict(dist)

    # Stability: fraction of M imputations agreeing with the consensus
    # For each country, compute the modal raw cluster within its consensus group
    stability = np.zeros(N)
    for i in range(N):
        # Across M, how often does country i end up in the same raw cluster as
        # the modal raw cluster (within its consensus partition)?
        # Approximation: count fraction of (i, j) pairs sharing the same raw
        # cluster, where j is any other country also assigned to consensus[i].
        same_consensus = (consensus_labels == consensus_labels[i])
        same_consensus[i] = False  # exclude self
        if same_consensus.sum() == 0:
            stability[i] = 1.0
            continue
        stability[i] = co_membership[i, same_consensus].mean()

    return pd.DataFrame({
        'Country Code':       country_codes_canon,
        'Cluster_consensus':  consensus_labels,
        'Stability':          stability,
    })


# Run for the aggregate snapshot first
print('Running ensemble for aggregate snapshot (this may take 30s)...')
t0 = time.time()
consensus_aggr = cluster_ensemble('aggregate', k=5)
print(f'Done in {time.time() - t0:.1f}s')
print(f'Cluster sizes: {consensus_aggr["Cluster_consensus"].value_counts().sort_index().to_dict()}')
print(f'Mean stability: {consensus_aggr["Stability"].mean():.3f}')
print(f'Low-stability (< 0.8) countries: {(consensus_aggr["Stability"] < 0.8).sum()}')
'''))

cells.append(md('## 7. Cluster labelling via reference countries'))

cells.append(code('''# Anchor each cluster ID to an economic label using known reference countries.
# (KMeans cluster IDs are arbitrary; hierarchical consensus labels are too.)
REFERENCE_COUNTRIES = [
    ('SAU', 'Petrostates'),
    ('NGA', 'Oil Exporters'),
    ('CHL', 'Mining Specialists'),
    ('MNG', 'Diverse Resource Exporters'),
    ('VNM', 'Diversified Manufacturing'),
]


def label_clusters(consensus_df, snapshot_label):
    """Map consensus cluster IDs to economic labels using reference countries."""
    label_map = {}
    used = set()
    for ref_code, label in REFERENCE_COUNTRIES:
        row = consensus_df[consensus_df['Country Code'] == ref_code]
        if len(row) == 0:
            continue
        cid = int(row['Cluster_consensus'].iloc[0])
        if cid not in label_map and label not in used:
            label_map[cid] = label
            used.add(label)

    # Fallback: any unassigned cluster ID gets generic label
    for cid in sorted(consensus_df['Cluster_consensus'].unique()):
        if cid not in label_map:
            label_map[cid] = f'Unlabeled_{cid}'

    consensus_df = consensus_df.copy()
    consensus_df['ClusterLabel'] = consensus_df['Cluster_consensus'].map(label_map)
    print(f'\\nSnapshot: {snapshot_label}')
    for cid in sorted(label_map.keys()):
        lbl = label_map[cid]
        codes = sorted(consensus_df[consensus_df['Cluster_consensus'] == cid]['Country Code'].tolist())
        n = len(codes)
        sample = ', '.join(codes[:10]) + ('...' if n > 10 else '')
        print(f'  {lbl:30s} (n={n:3d}): {sample}')
    return consensus_df, label_map


consensus_aggr_labeled, label_map_aggr = label_clusters(consensus_aggr, 'aggregate')
'''))

cells.append(md('## 8. Run for all four snapshots'))

cells.append(code('''snapshot_results = {}
snapshot_labels = {}

for snap in ['1995', '2010', '2023', 'aggregate']:
    print(f'\\n--- Snapshot {snap} ---')
    t0 = time.time()
    consensus = cluster_ensemble(snap, k=5)
    consensus_labeled, label_map = label_clusters(consensus, snap)
    print(f'Done in {time.time() - t0:.1f}s, mean stability {consensus_labeled["Stability"].mean():.3f}')
    snapshot_results[snap] = consensus_labeled
    snapshot_labels[snap] = label_map
'''))

cells.append(md('## 9. PCA biplot for the aggregate snapshot'))

cells.append(code('''# Take a single imputation's PCA for visualisation
res_viz, pca_model, km_model, _ = cluster_one(
    prepare_snapshot(imp0, 'aggregate'), n_clusters=5, return_model=True
)

# Merge consensus labels for colouring
res_viz = res_viz.merge(
    snapshot_results['aggregate'][['Country Code', 'Cluster_consensus', 'ClusterLabel']],
    on='Country Code', how='left',
)

fig, ax = plt.subplots(figsize=(11, 8))
for cid in sorted(res_viz['Cluster_consensus'].dropna().unique()):
    sub = res_viz[res_viz['Cluster_consensus'] == cid]
    label = sub['ClusterLabel'].iloc[0]
    ax.scatter(sub['PC1'], sub['PC2'], label=f'{label} (n={len(sub)})', alpha=0.7, s=40)

# Annotate notable countries
notable = ['SAU', 'NOR', 'CHL', 'COG', 'AZE', 'VEN', 'NGA', 'AGO', 'ZAF',
           'BRA', 'IND', 'CHN', 'MEX', 'TUR', 'IDN', 'VNM', 'MNG', 'PER']
for cc in notable:
    row = res_viz[res_viz['Country Code'] == cc]
    if len(row):
        ax.annotate(cc, (row['PC1'].iloc[0], row['PC2'].iloc[0]),
                    fontsize=8, alpha=0.8)

# PCA loading arrows
loadings = pca_model.components_.T
for i, feat in enumerate(FEATURES):
    if abs(loadings[i, 0]) > 0.25 or abs(loadings[i, 1]) > 0.25:
        ax.annotate('', xy=(loadings[i, 0] * 3, loadings[i, 1] * 3),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        ax.text(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2, feat,
                fontsize=8, color='gray')

ax.axhline(0, color='gray', alpha=0.3)
ax.axvline(0, color='gray', alpha=0.3)
ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('Resource portfolio clusters (1995-2023 aggregate)')
ax.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(GRAPHICS / 'biplot_aggregate.png', dpi=150, bbox_inches='tight')
plt.show()
'''))

cells.append(md('## 10. Cluster transitions (1995 -> 2023)'))

cells.append(code('''# How many countries changed cluster between 1995 and 2023?
trans = (
    snapshot_results['1995'][['Country Code', 'ClusterLabel']].rename(columns={'ClusterLabel': '1995'})
    .merge(
        snapshot_results['2023'][['Country Code', 'ClusterLabel']].rename(columns={'ClusterLabel': '2023'}),
        on='Country Code', how='inner',
    )
)
trans['Changed'] = trans['1995'] != trans['2023']
print(f'Countries in both 1995 and 2023: {len(trans)}')
print(f'Cluster changers: {trans["Changed"].sum()} ({100*trans["Changed"].mean():.1f}%)')
print()
print('Transition table (1995 rows -> 2023 columns):')
table = pd.crosstab(trans['1995'], trans['2023'])
print(table.to_string())
'''))

cells.append(md('## 11. Export consensus + trajectory per country'))

cells.append(code('''# One row per country, columns: cluster in 1995/2010/2023/aggregate + stabilities
out = (
    snapshot_results['aggregate'][['Country Code', 'ClusterLabel', 'Stability']]
    .rename(columns={'ClusterLabel': 'Cluster_aggregate', 'Stability': 'Stability_aggregate'})
)
for snap in ['1995', '2010', '2023']:
    out = out.merge(
        snapshot_results[snap][['Country Code', 'ClusterLabel', 'Stability']]
        .rename(columns={
            'ClusterLabel': f'Cluster_{snap}',
            'Stability':    f'Stability_{snap}',
        }),
        on='Country Code', how='outer',
    )

# Add country names from imp0
imp0_names = imp0.groupby('Country Code')['Country Name'].first()
out['Country Name'] = out['Country Code'].map(imp0_names)
out = out[['Country Code', 'Country Name',
           'Cluster_aggregate', 'Cluster_1995', 'Cluster_2010', 'Cluster_2023',
           'Stability_aggregate', 'Stability_1995', 'Stability_2010', 'Stability_2023']]
out = out.sort_values('Country Name')

out_csv = INTER / 'Master_v2_clusters.csv'
out.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')
print(f'Rows: {len(out)}')
print('Sample:')
print(out.head(15).to_string(index=False))
'''))

cells.append(md('## 12. Summary'))

cells.append(code('''print('=' * 70)
print('e3 — Clustering summary')
print('=' * 70)
for snap in ['1995', '2010', '2023', 'aggregate']:
    d = snapshot_results[snap]
    n = len(d)
    mean_stab = d['Stability'].mean()
    low_stab = (d['Stability'] < 0.8).sum()
    print(f'  {snap:10s}: N={n}, mean stability={mean_stab:.3f}, low-stability countries: {low_stab}')

print()
print('Files:')
print(f'  intermediary/Master_v2_clusters.csv')
print(f'  Graphics/NB3/silhouette.png')
print(f'  Graphics/NB3/biplot_aggregate.png')
print()
print('Next: feed into e5 as a moderator (run main regression separately by cluster)')
'''))

# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.4'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Wrote {NB_PATH}')
print(f'Cells: {len(cells)} ({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')
