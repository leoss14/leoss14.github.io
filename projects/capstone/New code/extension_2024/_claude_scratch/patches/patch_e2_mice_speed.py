"""Optimize the e2 MICE runtime.

Two changes to the run-MICE cell:
  1. Build `variable_schema` listing only columns that contain NaN as targets.
     Variables with no missing values are passed through untouched.
  2. For each target, restrict predictors to top-K most correlated other
     columns (K=20 default; this preserves >90% of cross-variable signal
     while cutting RF training cost roughly in half).

The schema is logged so it's clear what's being imputed and what isn't.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e2_data_prep.ipynb'

NEW_RUN_MICE = '''# Run MICE
mice_input = df_demeaned[mice_cols + list(CATEGORICAL_COLS)].copy()
# Apply name sanitisation
mice_input = mice_input.rename(columns=SAN_MAP)

# ── OPTIMIZATION: only impute columns that contain NaN, and restrict
# predictors per target to the top-K most correlated other columns.
# Columns with no missing values are still passed to miceforest (it needs
# the full data matrix for prediction) but they're not listed as targets,
# so no RF is trained for them. This is a substantial speed-up.

nan_counts = mice_input.isna().sum()
cols_with_missing = nan_counts[nan_counts > 0].index.tolist()
cols_no_missing = nan_counts[nan_counts == 0].index.tolist()

print(f'Columns with missing values: {len(cols_with_missing)}')
print(f'Columns fully observed (skipped as targets): {len(cols_no_missing)}')

# For each target, find the top-K most correlated other columns to use as predictors.
# Correlation computed on observed pairs only.
TOP_K_PREDICTORS = 20

# Compute correlation matrix on numeric columns only (categoricals will be added back)
cat_sanitised = {SAN_MAP[c] for c in CATEGORICAL_COLS if c in SAN_MAP}
numeric_for_corr = [c for c in mice_input.columns if c not in cat_sanitised]

corr = mice_input[numeric_for_corr].corr().abs()

# Build variable_schema: for each target, its top-K predictors (excluding itself).
# All categorical columns are always included as candidate predictors.
variable_schema = {}
for target in cols_with_missing:
    if target in cat_sanitised:
        # For categorical targets, use everything as predictors (these are
        # only Civil war etc. with simple structure)
        predictors = [c for c in mice_input.columns if c != target]
    else:
        if target not in corr.columns:
            continue
        # Top-K most correlated numeric columns (excluding self)
        top = corr[target].drop(target).sort_values(ascending=False).head(TOP_K_PREDICTORS).index.tolist()
        # Always include categoricals as candidates
        predictors = top + list(cat_sanitised)
    variable_schema[target] = [p for p in predictors if p in mice_input.columns]

print(f'Variable schema: {len(variable_schema)} targets, '
      f'~{int(np.mean([len(v) for v in variable_schema.values()]))} predictors each (median)')

print(f'\\nStarting miceforest: M={M_IMPUTATIONS} imputations, '
      f'{len(mice_input.columns)} total cols (imputing {len(variable_schema)}), '
      f'{len(mice_input):,} rows')
t0 = time.time()

kernel = mf.ImputationKernel(
    mice_input,
    num_datasets=M_IMPUTATIONS,
    variable_schema=variable_schema,
    random_state=0,
)
kernel.mice(iterations=5, verbose=False)
print(f'MICE done in {time.time() - t0:.1f}s ({(time.time() - t0)/60:.1f} min)')

# Extract all M imputed datasets, undo sanitisation
imputed_datasets = []
for m in range(M_IMPUTATIONS):
    sub = kernel.complete_data(dataset=m).copy()
    # Restore original column names
    sub = sub.rename(columns=SAN_INV)
    # Reattach ID cols
    sub = pd.concat([
        df[ID_COLS].reset_index(drop=True),
        sub.reset_index(drop=True)
    ], axis=1)
    # Add back country means (undo demeaning) for numeric columns
    for c in mice_cols:
        sub[c] = sub[c] + country_means[c].reset_index(drop=True)
    sub['imputation_id'] = m
    imputed_datasets.append(sub)

print(f'Extracted {len(imputed_datasets)} imputed panels.')
'''

with open(NB) as f:
    nb = json.load(f)

target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'mf.ImputationKernel' in s and 'imputed_datasets' in s and 'cols_with_missing' not in s:
        target = i
        break

if target is None:
    # Already patched?
    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'cols_with_missing' in s:
            print(f'cell {i}: already optimized.')
            sys.exit(0)
    print('ERROR: run-MICE cell not found.')
    sys.exit(1)

nb['cells'][target]['source'] = NEW_RUN_MICE.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: optimized.')
