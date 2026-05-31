"""
Patch e3_clusters.ipynb with a MODE switch (gross vs net).

Default MODE = 'gross' to preserve existing behavior on accidental kernel restart.
Run with MODE = 'net' to produce the net-share version of the clustering.

Changes:
  - Cell 2: introduce MODE constant, output paths route through MODE subfolder.
  - Cell 4: feature names route through a column-name resolver (resolve_col)
            that maps 'hsXX_share' -> 'hsXX_share_net' etc. when MODE='net'.
  - Cell 8: cluster_one uses resolve_col list.
  - Cell 26: threshold applied against wide_resource_share[_net] depending on MODE.
  - Cell 27: cluster_one_aggshares uses resolve_col list.
  - Save paths get _net suffix when MODE='net'.

Backup the notebook first.
"""
import json
from pathlib import Path
import shutil

NB = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e3_clusters.ipynb')
BAK = NB.with_suffix('.ipynb.bak_before_mode_switch')

if not BAK.exists():
    shutil.copy2(NB, BAK)
    print(f'Backup: {BAK}')
else:
    print(f'Backup already exists: {BAK}')

with open(NB) as f:
    nb = json.load(f)

def set_cell(idx, src):
    nb['cells'][idx]['source'] = src.splitlines(keepends=True)
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None


# ============================================================
# Cell 2: Setup. Add MODE switch + resolve_col helper. Route GRAPHICS path through MODE.
# ============================================================
CELL_2 = """\
import sys, time, warnings
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

# ----------------------------------------------------------------
# MODE switch: 'gross' uses *_share columns, 'net' uses *_share_net.
# Override by setting CLUSTER_MODE env var or editing this line.
# ----------------------------------------------------------------
import os
MODE = os.environ.get('CLUSTER_MODE', 'gross')
assert MODE in ('gross', 'net'), f"Unknown MODE={MODE!r}; must be 'gross' or 'net'"
print(f'MODE = {MODE}')

def resolve_col(name):
    \"\"\"Map a gross share column name to its net counterpart when MODE='net'.\"\"\"
    if MODE == 'gross':
        return name
    # Net column naming: <stem>_net
    return name + '_net'

EXT  = Path('.').resolve()
INTER = EXT / 'intermediary'
# Route outputs through MODE subfolder so gross/net coexist.
GRAPHICS = EXT / 'Graphics' / 'NB3' / MODE
GRAPHICS.mkdir(parents=True, exist_ok=True)

# Save-path helpers for downstream CSV outputs.
def _suffix(name):
    \"\"\"Append _net before the file extension when MODE='net'.\"\"\"
    if MODE == 'gross':
        return name
    p = Path(name)
    return f'{p.stem}_net{p.suffix}'

print(f'Working dir: {EXT}')
print(f'Output dir:  {GRAPHICS}')
"""
set_cell(2, CELL_2)
print('Cell 2 patched (MODE switch + path routing)')


# ============================================================
# Cell 4: Feature definitions. Apply resolve_col to the feature lists.
# ============================================================
CELL_4 = """\
# Per-chapter shares (HS code -> column name in Master_v2)
HS_CHAPTERS = [25, 26, 27, 28, 29, 44, 71, 72, 74, 75, 76, 78, 79, 80, 81]
PER_CHAPTER_FEATURES = [resolve_col(f'hs{c}_share') for c in HS_CHAPTERS]

# Aggregated shares
AGGREGATE_FEATURES = [resolve_col(c) for c in [
    'hydrocarbon_share',
    'ores_share',
    'base_metals_share',
    'precious_share',
]]

# All cluster features
ALL_FEATURES = PER_CHAPTER_FEATURES + AGGREGATE_FEATURES

# Verify these exist in Master_v2 by checking imputation 0
imp0 = next(iter_imputations())[1]
missing = [c for c in ALL_FEATURES if c not in imp0.columns]
present = [c for c in ALL_FEATURES if c in imp0.columns]
print(f'Available features ({MODE}): {len(present)} / {len(ALL_FEATURES)}')
if missing:
    print(f'Missing: {missing}')
FEATURES = present
"""
set_cell(4, CELL_4)
print('Cell 4 patched (FEATURES through resolve_col)')


# ============================================================
# Cell 6: Snapshot prep. EXTRA_FEATURES routed through resolve_col.
# ============================================================
CELL_6 = """\
def prepare_snapshot(panel, snapshot):
    \"\"\"Filter a panel to the snapshot definition and aggregate to one row per country.

    Retains FEATURES plus any HS27 sub-code share columns present in the panel.
    The sub-codes are used by the RR-subset clustering pipeline (cluster_one_aggshares);
    the full-panel cluster_one() pipeline still references FEATURES explicitly and is
    not affected by the extra columns being present.\"\"\"
    df = panel.copy()

    # Sub-code share columns added in the e2 re-run; retain them so the RR-subset
    # clustering can use them via RR_FEATURES. Route through resolve_col so they
    # pick up the _net suffix when MODE='net'.
    EXTRA_FEATURES = [resolve_col(c) for c in
                      ['coal_share', 'crude_oil_share', 'refined_oil_share', 'gas_share']]
    extra_present = [c for c in EXTRA_FEATURES if c in panel.columns]
    feats = FEATURES + extra_present

    if snapshot == '1995':
        df = df[df['Year'] == 1995]
    elif snapshot == '2010':
        df = df[df['Year'] == 2010]
    elif snapshot == '2023':
        df = df[df['Year'] == 2023]
    elif snapshot == 'aggregate':
        # Full-panel mean across all years for each country
        df = df.groupby('Country Code', as_index=False)[feats].mean()
        df['Country Name'] = panel.groupby('Country Code')['Country Name'].first().reindex(
            df['Country Code']
        ).values
        return df[['Country Code', 'Country Name'] + feats]
    else:
        raise ValueError(f'unknown snapshot: {snapshot}')

    # For single-year snapshots, take that year's row per country
    cols = ['Country Code', 'Country Name'] + feats
    return df[cols].copy()


# Quick sanity check
snap_aggr = prepare_snapshot(imp0, 'aggregate')
print(f'Aggregate snapshot: {len(snap_aggr)} countries, {len(snap_aggr.columns) - 2} feature columns')
print(snap_aggr.head(3).to_string(index=False))
"""
set_cell(6, CELL_6)
print('Cell 6 patched (EXTRA_FEATURES through resolve_col)')


# ============================================================
# Cell 26: RR threshold filter. Apply against wide_resource_share[_net].
# ============================================================
CELL_26 = """\
# Define the resource-rich subset based on aggregate (full-panel) wide_resource_share mean.
# Use the first imputation only since wide_resource_share is observed data, not imputed.
#
# Under MODE='net' the threshold is applied to wide_resource_share_net. Countries with
# negative net wide-share automatically fail the > THRESHOLD test, which is the
# "cluster only on positive net exports" rule.

panel0 = next(iter_imputations())[1]
share_col_for_threshold = resolve_col('wide_resource_share')
mean_share = panel0.groupby('Country Code')[share_col_for_threshold].mean()

THRESHOLD = 0.40
resource_rich_codes = mean_share[mean_share > THRESHOLD].index.tolist()
print(f'Resource-rich threshold: mean {share_col_for_threshold} > {THRESHOLD}')
print(f'Resource-rich countries (N = {len(resource_rich_codes)}):')
print('  ' + ', '.join(sorted(resource_rich_codes)))
print()
print(f'Mean {share_col_for_threshold} distribution:')
print(f'  min: {mean_share.min():.3f}')
print(f'  25%: {mean_share.quantile(0.25):.3f}')
print(f'  median: {mean_share.median():.3f}')
print(f'  75%: {mean_share.quantile(0.75):.3f}')
print(f'  max: {mean_share.max():.3f}')
print(f'  > 0.30: {(mean_share > 0.30).sum()} countries')
print(f'  > 0.20: {(mean_share > 0.20).sum()} countries')
print(f'  > 0.40: {(mean_share > 0.40).sum()} countries')
"""
set_cell(26, CELL_26)
print('Cell 26 patched (threshold against resolved column)')


# ============================================================
# Cell 27: RR_FEATURES through resolve_col.
# ============================================================
CELL_27 = """\
# Clustering features for the resource-rich subsample.
# Routes through resolve_col so MODE='net' picks up _net columns.
# Under MODE='net': features can be negative for net-importer country-years.
# The np.clip(feats, 0, None) line in cluster_one_aggshares handles this by
# treating net imports as zero contribution, which is the "specialisation in
# net exports only" interpretation.
RR_FEATURES = [resolve_col(c) for c in
               ['coal_share', 'crude_oil_share', 'refined_oil_share', 'gas_share',
                'ores_share', 'base_metals_share', 'precious_share']]
print(f'Resource-rich clustering features ({MODE}): {RR_FEATURES}')


def cluster_one_aggshares(snapshot_df, n_clusters=3, random_state=42, return_model=False):
    \"\"\"Cluster pipeline using the RR_FEATURES set (4 HS27 sub-codes plus
    ores, base_metals, precious). log1p + PCA(2) + KMeans. Returns same
    fields as cluster_one(). Function name retained for backward
    compatibility with downstream callers.\"\"\"
    feats = snapshot_df[RR_FEATURES].fillna(0).values
    # Clip negatives at zero. Under MODE='net' this means net-importer
    # country-years contribute zero in the affected category, which is
    # the cleanest interpretation for clustering specialisation.
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
"""
set_cell(27, CELL_27)
print('Cell 27 patched (RR_FEATURES through resolve_col)')


# ============================================================
# Cells with hard-coded save paths: Master_v2_clusters.csv and
# Master_v2_clusters_RR_silhouette.csv. Need to add _net suffix when MODE='net'.
# Find them by string-matching.
# ============================================================
patched_save_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'Master_v2_clusters' in src and "to_csv" in src:
        # Replace literal filenames with _suffix() wrapped versions
        new_src = src
        for fname in ['Master_v2_clusters.csv', 'Master_v2_clusters_RR.csv',
                       'Master_v2_clusters_RR_silhouette.csv']:
            # Match patterns like: INTER / 'Master_v2_clusters.csv'
            # or: cfg.INTERMEDIARY / 'Master_v2_clusters.csv'
            # Replace with: INTER / _suffix('Master_v2_clusters.csv')
            new_src = new_src.replace(f"'{fname}'", f"_suffix('{fname}')")
            new_src = new_src.replace(f'"{fname}"', f'_suffix("{fname}")')
        if new_src != src:
            set_cell(i, new_src)
            patched_save_cells.append(i)

print(f'Save cells patched: {patched_save_cells}')


# ============================================================
# Hard-coded Graphics file paths in the visualization cells:
# Already routed via GRAPHICS which is now MODE-aware. No change needed.
# ============================================================

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'\\nSaved: {NB}')
print()
print('======================================================')
print('To run gross: open notebook, restart kernel, Run All.')
print('To run net:   set CLUSTER_MODE=net in environment, OR')
print('              edit cell 2 line: MODE = \"net\"')
print('              then restart kernel, Run All.')
print('Outputs:')
print('  Graphics/NB3/gross/   (existing maps + biplot + silhouette)')
print('  Graphics/NB3/net/     (new, populated after net re-run)')
print('  intermediary/Master_v2_clusters.csv')
print('  intermediary/Master_v2_clusters_net.csv')
print('  intermediary/Master_v2_clusters_RR_silhouette.csv')
print('  intermediary/Master_v2_clusters_RR_silhouette_net.csv')
print('======================================================')
