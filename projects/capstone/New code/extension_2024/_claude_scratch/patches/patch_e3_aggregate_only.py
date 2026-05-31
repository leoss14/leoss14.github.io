"""Re-run resource-rich clustering using only the 4 aggregate share variables.

Replaces the prior 19-feature design with a clean 4-feature design:
  hydrocarbon_share, ores_share, base_metals_share, precious_share

Same threshold (mean wide_resource_share > 0.30), same M-imputation ensemble,
same k values (3, 4, 5). Updates the columns added to Master_v2_clusters.csv.

The original e3 section 13 cells are patched in-place. Maps and inventory
(cells 30+) keep working since column names didn't change.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb'

# New version of cell 27 (was: cluster_ensemble_subset using FEATURES list)
NEW_CELL_27 = '''# Clustering features for the resource-rich subsample.
# Use only the 4 aggregate shares (not the per-chapter shares) for two reasons:
#  1. Avoids redundancy: hs27_share=hydrocarbon_share, hs26_share=ores_share,
#     hs71_share=precious_share are exact duplicates; including both would
#     give those dimensions double weight in PCA.
#  2. Direct interpretability: cluster boundaries are read in the same space
#     the regressions use.
RR_FEATURES = ['hydrocarbon_share', 'ores_share', 'base_metals_share', 'precious_share']
print(f'Resource-rich clustering features: {RR_FEATURES}')


def cluster_one_aggshares(snapshot_df, n_clusters=3, random_state=42, return_model=False):
    """Cluster pipeline using only the 4 aggregate shares.
    log1p + PCA(2) + KMeans. Returns same fields as cluster_one()."""
    feats = snapshot_df[RR_FEATURES].fillna(0).values
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


def cluster_ensemble_subset(snapshot, k, country_codes):
    """Ensemble across M MICE imputations on the resource-rich subset,
    using the aggregate-share feature set."""
    panels = list(iter_imputations())
    M = len(panels)

    label_arrays = []
    country_codes_canon = None
    silhouettes_per_imp = []

    for imp_id, panel in panels:
        snap = prepare_snapshot(panel, snapshot)
        snap = snap[snap['Country Code'].isin(country_codes)].reset_index(drop=True)
        if len(snap) < k:
            print(f'  WARNING: snapshot {snapshot}, k={k}, imp {imp_id}: only {len(snap)} countries')
            return None
        res, sil = cluster_one_aggshares(snap, n_clusters=k, random_state=42)
        if country_codes_canon is None:
            country_codes_canon = res['Country Code'].tolist()
            label_arrays.append(res['Cluster_raw'].values)
        else:
            res = res.set_index('Country Code').reindex(country_codes_canon).reset_index()
            label_arrays.append(res['Cluster_raw'].fillna(-1).astype(int).values)
        silhouettes_per_imp.append(sil)

    label_matrix = np.array(label_arrays)
    M_, N = label_matrix.shape
    co_membership = np.zeros((N, N))
    for m in range(M_):
        same = (label_matrix[m, :, None] == label_matrix[m, None, :]).astype(float)
        co_membership += same
    co_membership /= M_

    dist = 1 - co_membership
    np.fill_diagonal(dist, 0)
    agg = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    consensus_labels = agg.fit_predict(dist)

    stability = np.zeros(N)
    for i in range(N):
        same_consensus = (consensus_labels == consensus_labels[i])
        same_consensus[i] = False
        if same_consensus.sum() == 0:
            stability[i] = 1.0
            continue
        stability[i] = co_membership[i, same_consensus].mean()

    return {
        'codes':       country_codes_canon,
        'cluster':     consensus_labels,
        'stability':   stability,
        'silhouette_mean': float(np.mean(silhouettes_per_imp)),
    }


# Run all combinations
SUB_K_VALUES = [3, 4, 5]
SUB_SNAPSHOTS = ['1995', '2010', '2023', 'aggregate']

sub_results = {}
silhouette_table = []

print('\\nRunning resource-rich sub-clustering (aggregate shares only):')
for snap in SUB_SNAPSHOTS:
    for k in SUB_K_VALUES:
        t0 = time.time()
        res = cluster_ensemble_subset(snap, k, resource_rich_codes)
        if res is None:
            sub_results[(k, snap)] = None
            silhouette_table.append({'snapshot': snap, 'k': k, 'silhouette': np.nan, 'n_countries': 0})
            continue
        sub_results[(k, snap)] = res
        silhouette_table.append({
            'snapshot': snap, 'k': k,
            'silhouette': res['silhouette_mean'],
            'n_countries': len(res['codes']),
        })
        print(f'  {snap:10s} k={k}: silhouette={res["silhouette_mean"]:.3f}, '
              f'N={len(res["codes"])}, time={time.time() - t0:.1f}s')

silhouette_df = pd.DataFrame(silhouette_table)
print('\\nSilhouette scores by (snapshot, k):')
print(silhouette_df.pivot(index='snapshot', columns='k', values='silhouette').round(3).to_string())
'''


def patch():
    with open(NB) as f:
        nb = json.load(f)

    # Find the cell that currently defines `cluster_ensemble_subset` and runs the loop
    target = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'cluster_ensemble_subset' in s and 'SUB_K_VALUES' in s and 'def cluster_ensemble_subset' in s:
            target = i
            break

    if target is None:
        print('ERROR: existing resource-rich clustering cell not found.')
        sys.exit(1)

    nb['cells'][target]['source'] = NEW_CELL_27.splitlines(keepends=True)
    nb['cells'][target]['outputs'] = []
    nb['cells'][target]['execution_count'] = None

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'cell {target}: replaced with aggregate-shares-only clustering.')


if __name__ == '__main__':
    patch()
