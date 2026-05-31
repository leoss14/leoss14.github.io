"""Add resource-rich subsample clustering as a new section to e3_clusters.ipynb.

Definition: country is resource-rich if mean(wide_resource_share) > 0.30
across 1995-2023.

For each of 4 snapshots (1995, 2010, 2023, aggregate) and 3 k values (3, 4, 5):
  - Subset to resource-rich countries only
  - Run k-means with M-imputation ensemble
  - Record cluster ID + silhouette + stability

Appends to intermediary/Master_v2_clusters.csv with new columns.
"""
import json, sys
from pathlib import Path

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb'

NEW_CELLS_MARKDOWN = '''## 13. Resource-Rich Subsample Clustering (Robustness)

Subsample restricted to countries where mean `wide_resource_share` > 0.30 over
1995-2023. Yields ~50-70 resource-dependent economies.

For each of the four snapshots (1995, 2010, 2023, aggregate) and each k in
{3, 4, 5}, we run the same MICE-ensemble clustering pipeline as the headline
analysis above. Three k values reported side by side so cluster granularity
can be chosen post-hoc.

No reference-country labelling here. Cluster IDs are integers (0..k-1) and can
be re-mapped later via dominant-feature analysis or labels assigned in a paper
write-up.

Output columns added to `Master_v2_clusters.csv`:
- `Resource_Rich` -- True if mean wide_resource_share > 0.30
- `RR_kX_<snapshot>` -- cluster ID (0..k-1) at each (k, snapshot)
- `RR_silhouette_kX_<snapshot>` -- silhouette score for that (k, snapshot)
- `RR_stability_kX_<snapshot>` -- per-country stability score across MICE imputations
'''

NEW_CELLS_CODE_1 = '''# Define the resource-rich subset based on aggregate (full-panel) wide_resource_share mean.
# Use the first imputation only since wide_resource_share is observed data, not imputed.

panel0 = next(iter_imputations())[1]
mean_share = panel0.groupby('Country Code')['wide_resource_share'].mean()

THRESHOLD = 0.30
resource_rich_codes = mean_share[mean_share > THRESHOLD].index.tolist()
print(f'Resource-rich threshold: mean wide_resource_share > {THRESHOLD}')
print(f'Resource-rich countries (N = {len(resource_rich_codes)}):')
print('  ' + ', '.join(sorted(resource_rich_codes)))
print()
print(f'Mean wide_resource_share distribution:')
print(f'  min: {mean_share.min():.3f}')
print(f'  25%: {mean_share.quantile(0.25):.3f}')
print(f'  median: {mean_share.median():.3f}')
print(f'  75%: {mean_share.quantile(0.75):.3f}')
print(f'  max: {mean_share.max():.3f}')
print(f'  > 0.30: {(mean_share > 0.30).sum()} countries')
print(f'  > 0.20: {(mean_share > 0.20).sum()} countries')
print(f'  > 0.40: {(mean_share > 0.40).sum()} countries')
'''

NEW_CELLS_CODE_2 = '''def cluster_ensemble_subset(snapshot, k, country_codes):
    """Like cluster_ensemble but restricts to a specified country list."""
    panels = list(iter_imputations())
    M = len(panels)

    label_arrays = []
    country_codes_canon = None
    silhouettes_per_imp = []

    for imp_id, panel in panels:
        snap = prepare_snapshot(panel, snapshot)
        # Restrict to the resource-rich subset
        snap = snap[snap['Country Code'].isin(country_codes)].reset_index(drop=True)
        if len(snap) < k:
            print(f'  WARNING: snapshot {snapshot}, k={k}, imp {imp_id}: only {len(snap)} countries')
            return None
        res, sil = cluster_one(snap, n_clusters=k, random_state=42)
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

sub_results = {}  # (k, snapshot) -> dict
silhouette_table = []

print('Running resource-rich sub-clustering for all (k, snapshot) combinations:')
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
            'snapshot': snap,
            'k': k,
            'silhouette': res['silhouette_mean'],
            'n_countries': len(res['codes']),
        })
        print(f'  {snap:10s} k={k}: silhouette={res["silhouette_mean"]:.3f}, '
              f'N={len(res["codes"])}, time={time.time() - t0:.1f}s')

silhouette_df = pd.DataFrame(silhouette_table)
print('\\nSilhouette scores by (snapshot, k):')
print(silhouette_df.pivot(index='snapshot', columns='k', values='silhouette').round(3).to_string())
'''

NEW_CELLS_CODE_3 = '''# Build the resource-rich addition to Master_v2_clusters.csv
rr_df = pd.DataFrame({'Country Code': sorted(set().union(*[set(r['codes']) for r in sub_results.values() if r is not None]))})
rr_df['Resource_Rich'] = True

for (k, snap), res in sub_results.items():
    if res is None:
        col_cluster = f'RR_k{k}_{snap}'
        col_stab    = f'RR_stability_k{k}_{snap}'
        rr_df[col_cluster] = np.nan
        rr_df[col_stab]    = np.nan
        continue
    df_temp = pd.DataFrame({
        'Country Code': res['codes'],
        f'RR_k{k}_{snap}': res['cluster'],
        f'RR_stability_k{k}_{snap}': res['stability'],
    })
    rr_df = rr_df.merge(df_temp, on='Country Code', how='left')

# Merge into the existing clusters CSV
existing = pd.read_csv(INTER / 'Master_v2_clusters.csv')
merged = existing.merge(rr_df, on='Country Code', how='left')
merged['Resource_Rich'] = merged['Resource_Rich'].fillna(False)

out_csv = INTER / 'Master_v2_clusters.csv'
merged.to_csv(out_csv, index=False)
print(f'Wrote: {out_csv}')
print(f'Total rows: {len(merged)}, total cols: {len(merged.columns)}')
print(f'Resource-Rich countries: {merged["Resource_Rich"].sum()}')

# Save silhouette summary as separate file
silhouette_df.to_csv(INTER / 'Master_v2_clusters_RR_silhouette.csv', index=False)
print(f'Wrote: {INTER / "Master_v2_clusters_RR_silhouette.csv"}')
'''

NEW_CELLS_CODE_4 = '''# Quick visualization: silhouette across (k, snapshot)
fig, ax = plt.subplots(figsize=(8, 5))
pivoted = silhouette_df.pivot(index='snapshot', columns='k', values='silhouette')
pivoted.plot(kind='bar', ax=ax, color=['#4a6fa5', '#c23a3a', '#2e7d4a'], width=0.7)
ax.set_title(f'Resource-Rich sub-clustering silhouette across k\\n(threshold: mean wide_resource_share > {THRESHOLD})')
ax.set_xlabel('Snapshot')
ax.set_ylabel('Silhouette score')
ax.legend(title='k')
ax.set_xticklabels(pivoted.index, rotation=0)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(GRAPHICS / 'silhouette_resource_rich.png', dpi=150, bbox_inches='tight')
plt.show()
'''


def patch():
    with open(NB) as f:
        nb = json.load(f)

    # Idempotency: check if already added
    for c in nb['cells']:
        if c['cell_type'] != 'markdown':
            continue
        if '13. Resource-Rich' in ''.join(c.get('source', [])):
            print('Section 13 already present. No-op.')
            return

    cells_to_add = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': NEW_CELLS_MARKDOWN.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': NEW_CELLS_CODE_1.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': NEW_CELLS_CODE_2.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': NEW_CELLS_CODE_3.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
         'source': NEW_CELLS_CODE_4.splitlines(keepends=True)},
    ]

    nb['cells'].extend(cells_to_add)

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'Added {len(cells_to_add)} cells. Total cells now: {len(nb["cells"])}')


if __name__ == '__main__':
    patch()
