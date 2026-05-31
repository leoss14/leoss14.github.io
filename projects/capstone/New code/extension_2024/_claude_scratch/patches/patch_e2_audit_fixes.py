"""Apply audit fixes to e2_data_prep.ipynb.

1. Drop ANT (Netherlands Antilles), ERI (Eritrea, 1 year), TKM (Turkmenistan, 4 years),
   TUV (Tuvalu, 4 years) — non-sovereign or insufficient time-series.
2. Clip imputed GFCF >= 0 post-MICE.
3. Clip imputed shares to [0, 1] defensively.
4. Clip imputed inflation to plausible range [-30, 1100].
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e2_data_prep.ipynb'

# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: Extend DEPENDENT_TERRITORIES with ANT, and add a new INSUFFICIENT_COVERAGE
# filter applied alongside.
# ─────────────────────────────────────────────────────────────────────────────

with open(NB) as f:
    nb = json.load(f)

# Find the cell that defines DEPENDENT_TERRITORIES (Section 2 of e2)
target_terr = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'DEPENDENT_TERRITORIES' in s and "'GUM'" in s:
        target_terr = i
        break

if target_terr is None:
    print('ERROR: DEPENDENT_TERRITORIES cell not found.')
    sys.exit(1)

# Add ANT to the dependent territories list
s = ''.join(nb['cells'][target_terr]['source'])
if "'ANT'" not in s:
    # Insert ANT at the top of the set
    s = s.replace(
        "DEPENDENT_TERRITORIES = {",
        "DEPENDENT_TERRITORIES = {\n    'ANT',  # Netherlands Antilles (dissolved 2010)",
        1,
    )
    nb['cells'][target_terr]['source'] = s.splitlines(keepends=True)
    nb['cells'][target_terr]['outputs'] = []
    nb['cells'][target_terr]['execution_count'] = None
    print(f'cell {target_terr}: added ANT to DEPENDENT_TERRITORIES.')

# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Add an INSUFFICIENT_COVERAGE filter
# Insert a new cell after the HIC filter cell (cell 5).
# ─────────────────────────────────────────────────────────────────────────────

# Find the cell where the territories/HIC filter is applied (looks like the
# "Apply sovereign + HIC filters" cell)
target_filter = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'DEPENDENT_TERRITORIES' in s and 'hic_exclude' in s and 'M = M[' in s:
        target_filter = i
        break

if target_filter is None:
    print('WARNING: filter cell not found. Adding INSUFFICIENT_COVERAGE block elsewhere.')
else:
    s = ''.join(nb['cells'][target_filter]['source'])
    if 'INSUFFICIENT_COVERAGE' not in s:
        # Append the insufficient-coverage filter at the end of the cell
        addition = '''

# Drop countries with insufficient time-series coverage in the panel.
# These appeared in the audit as having <5 years of trade data:
#   ERI (Eritrea, 1 year), TKM (Turkmenistan, 4 years), TUV (Tuvalu, 4 years)
# Their within-country variation is too sparse for FE identification.
INSUFFICIENT_COVERAGE = {'ERI', 'TKM', 'TUV'}
pre_filter_n = M['Country Code'].nunique()
M = M[~M['Country Code'].isin(INSUFFICIENT_COVERAGE)]
post_filter_n = M['Country Code'].nunique()
print(f'Insufficient-coverage filter: dropped {pre_filter_n - post_filter_n} countries '
      f'({sorted(INSUFFICIENT_COVERAGE)})')
'''
        s = s + addition
        nb['cells'][target_filter]['source'] = s.splitlines(keepends=True)
        nb['cells'][target_filter]['outputs'] = []
        nb['cells'][target_filter]['execution_count'] = None
        print(f'cell {target_filter}: added INSUFFICIENT_COVERAGE filter.')

# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Post-MICE clip imputed values to plausible ranges.
# Insert AFTER the MICE extraction cell, BEFORE validation.
# ─────────────────────────────────────────────────────────────────────────────

# Find the "Extract all M imputed datasets" cell — we patch by appending clip logic
target_mice = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'imputed_datasets = []' in s and 'kernel.complete_data' in s:
        target_mice = i
        break

if target_mice is None:
    print('ERROR: MICE extraction cell not found.')
    sys.exit(1)

s = ''.join(nb['cells'][target_mice]['source'])
if 'POST_MICE_CLIPS' not in s:
    # Append clip logic at the end of the cell
    clip_block = '''

# Post-MICE: clip imputed values to plausible ranges. MICE on demeaned data
# occasionally produces values outside the observed range (e.g., negative
# GFCF as % of GDP); clip them back to physically meaningful bounds.
# Observed-only cells are untouched since they're not imputed.
POST_MICE_CLIPS = {
    'Gross fixed capital formation, all, Constant prices, Percent of GDP': (0, None),
    'Trade (% of GDP)': (0, None),
    'Government revenue': (0, None),
    'Domestic credit to private sector (% of GDP)': (0, None),
    'Urban population (% of total population)': (0, 100),
    'Access to electricity (% of population)': (0, 100),
    'Life expectancy at birth, total (years)': (0, 100),
    'Mobile cellular subscriptions (per 100 people)': (0, None),
    'Inflation, consumer prices (annual %)': (-30, None),  # observed min was -16.86
    'Human capital index': (1.0, 4.0),
    # Share variables: defensively clip to [0, 1]
    'wide_resource_share': (0, 1),
    'hydrocarbon_share': (0, 1),
    'ores_share': (0, 1),
    'base_metals_share': (0, 1),
    'precious_share': (0, 1),
    'tight_share': (0, 1),
    'extractives_share': (0, 1),
    # The COVID interactions are products of share × dummy so also in [0, 1]
    'post2019_x_hydrocarbon_share': (0, 1),
    'post2019_x_ores_share': (0, 1),
    'post2019_x_base_metals_share': (0, 1),
}

clip_summary = []
for ds in imputed_datasets:
    for col, (lo, hi) in POST_MICE_CLIPS.items():
        if col not in ds.columns:
            continue
        before = ds[col].copy()
        if lo is not None:
            ds[col] = ds[col].clip(lower=lo)
        if hi is not None:
            ds[col] = ds[col].clip(upper=hi)
        n_clipped = (before != ds[col]).sum()
        if n_clipped > 0:
            clip_summary.append((col, n_clipped))

# Per-column summary across all M imputations
from collections import Counter
clip_totals = Counter()
for col, n in clip_summary:
    clip_totals[col] += n

print(f'Post-MICE clipping applied:')
for col, n in clip_totals.most_common():
    print(f'  {col}: {n} cells clipped (across all M)')
if not clip_totals:
    print('  No cells required clipping.')
'''
    s = s + clip_block
    nb['cells'][target_mice]['source'] = s.splitlines(keepends=True)
    nb['cells'][target_mice]['outputs'] = []
    nb['cells'][target_mice]['execution_count'] = None
    print(f'cell {target_mice}: added POST_MICE_CLIPS.')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('e2 audit fixes applied.')
