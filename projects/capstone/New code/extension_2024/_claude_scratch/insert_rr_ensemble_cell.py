"""
Insert the missing RR-clustering loop cell into e3_clusters.ipynb.

The notebook references `sub_results` and `silhouette_df` in cell 28 but the
cell that builds them is missing. Re-create that cell using the same ensemble
pattern as cluster_ensemble() in cell 12, but using cluster_one_aggshares() and
restricted to resource_rich_codes.
"""
import json
from pathlib import Path

NB = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb')

with open(NB) as f:
    nb = json.load(f)

# The missing cell: run cluster_one_aggshares ensemble for (k, snapshot)
# combinations, build sub_results dict and silhouette_df.
MISSING_CELL = '''\
# Build sub_results: {(k, snapshot): {'codes', 'cluster', 'stability'}} via
# ensemble clustering on the resource-rich subsample only.
# silhouette_df: long-form table of mean silhouette by (k, snapshot).
#
# Pattern mirrors cluster_ensemble() in cell 12 but uses cluster_one_aggshares
# (the RR_FEATURES set) and restricts each imputation to resource_rich_codes
# before clustering.

def cluster_ensemble_rr(snapshot, k, codes):
    """RR-subset ensemble clustering for one (k, snapshot) combination.

    Returns dict with 'codes' (canonical order), 'cluster' (consensus labels),
    'stability' (per-country agreement fraction).
    """
    panels = list(iter_imputations())
    M = len(panels)

    label_arrays = []
    country_codes_canon = None
    sil_scores = []

    for imp_id, panel in panels:
        # Restrict to resource-rich codes
        panel_rr = panel[panel['Country Code'].isin(codes)]
        if panel_rr['Country Code'].nunique() < k + 1:
            # Too few countries for k clusters at this snapshot
            return None
        snap = prepare_snapshot(panel_rr, snapshot)
        # Drop countries with all-zero (or all-NaN) feature vectors that
        # would collapse the PCA. log1p of zero is zero so they collapse
        # to the origin and create degenerate clusters.
        feat_sum = snap[RR_FEATURES].fillna(0).clip(lower=0).sum(axis=1)
        snap = snap[feat_sum > 1e-6].reset_index(drop=True)
        if len(snap) < k + 1:
            return None

        res, sil = cluster_one_aggshares(snap, n_clusters=k, random_state=42)
        sil_scores.append(sil)

        if country_codes_canon is None:
            country_codes_canon = res['Country Code'].tolist()
            label_arrays.append(res['Cluster_raw'].values)
        else:
            res = res.set_index('Country Code').reindex(country_codes_canon).reset_index()
            label_arrays.append(res['Cluster_raw'].fillna(-1).astype(int).values)

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
        'codes':     country_codes_canon,
        'cluster':   consensus_labels,
        'stability': stability,
        'mean_silhouette': float(np.nanmean(sil_scores)),
    }


# Run for k in {3, 4, 5} across all four snapshots
sub_results = {}
silhouette_rows = []

print(f'Running RR ensemble clustering ({MODE}) for k in (3, 4, 5) and 4 snapshots...')
print()
for k in [3, 4, 5]:
    for snap in ['aggregate', '1995', '2010', '2023']:
        t0 = time.time()
        res = cluster_ensemble_rr(snap, k=k, codes=resource_rich_codes)
        dt = time.time() - t0
        sub_results[(k, snap)] = res
        if res is None:
            print(f'  k={k} {snap:>9s}: SKIPPED (not enough countries)')
            silhouette_rows.append({'k': k, 'snapshot': snap, 'silhouette': np.nan})
        else:
            sil = res['mean_silhouette']
            print(f'  k={k} {snap:>9s}: silhouette={sil:.3f}, '
                  f'N={len(res["codes"])}, mean stability={res["stability"].mean():.3f} '
                  f'({dt:.1f}s)')
            silhouette_rows.append({'k': k, 'snapshot': snap, 'silhouette': sil})

silhouette_df = pd.DataFrame(silhouette_rows)
print()
print('Best k per snapshot (max silhouette):')
print(silhouette_df.loc[silhouette_df.groupby('snapshot')['silhouette'].idxmax()].to_string(index=False))
'''


def md_cell(s):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': s.splitlines(keepends=True)}


def code_cell(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': s.splitlines(keepends=True)}


# Insert at position 28 (right after the RR_FEATURES + cluster_one_aggshares def,
# right before the cell that uses sub_results)
INSERT_AT = 28
nb['cells'].insert(INSERT_AT, code_cell(MISSING_CELL))

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Inserted RR ensemble loop cell at index {INSERT_AT}')
print(f'Total cells now: {len(nb["cells"])}')
print()
print('Cell inventory after patch:')
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    head = src.split('\n')[0][:100] if src else '(empty)'
    print(f'  [{i:2d}] {c["cell_type"]:8s} | {head}')
