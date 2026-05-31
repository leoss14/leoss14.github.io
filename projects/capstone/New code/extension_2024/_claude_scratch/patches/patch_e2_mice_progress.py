"""Add per-iteration progress to the MICE run cell, drop M from 10 to 5,
keep iterations at 5 per user instruction.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e2_data_prep.ipynb'

NEW_RUN_MICE = '''# Run MICE
# M_IMPUTATIONS was 10; reduced to 5 to halve runtime. Rubin's rules still
# apply with M=5; SE noise is modest when FMI < 0.3. Re-run with M=10 for
# paper-quality numbers once the pipeline is validated.
M_IMPUTATIONS = 5
ITERATIONS = 5

mice_input = df_demeaned[mice_cols + list(CATEGORICAL_COLS)].copy()
# Apply name sanitisation
mice_input = mice_input.rename(columns=SAN_MAP)

# ── OPTIMIZATION: only impute columns that contain NaN, and restrict
# predictors per target to the top-K most correlated other columns.
# Columns with no missing values are still passed to miceforest (it needs
# the full data matrix for prediction) but they're not listed as targets,
# so no RF is trained for them. Substantial speed-up.

nan_counts = mice_input.isna().sum()
cols_with_missing = nan_counts[nan_counts > 0].index.tolist()
cols_no_missing = nan_counts[nan_counts == 0].index.tolist()

print(f'Columns with missing values: {len(cols_with_missing)}')
print(f'Columns fully observed (skipped as targets): {len(cols_no_missing)}')

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
        predictors = [c for c in mice_input.columns if c != target]
    else:
        if target not in corr.columns:
            continue
        top = corr[target].drop(target).sort_values(ascending=False).head(TOP_K_PREDICTORS).index.tolist()
        predictors = top + list(cat_sanitised)
    variable_schema[target] = [p for p in predictors if p in mice_input.columns]

print(f'Variable schema: {len(variable_schema)} targets, '
      f'~{int(np.mean([len(v) for v in variable_schema.values()]))} predictors each (median)')

print(f'\\nStarting miceforest: M={M_IMPUTATIONS} imputations, '
      f'{len(mice_input.columns)} total cols (imputing {len(variable_schema)}), '
      f'{len(mice_input):,} rows, {ITERATIONS} iterations')

kernel = mf.ImputationKernel(
    mice_input,
    num_datasets=M_IMPUTATIONS,
    variable_schema=variable_schema,
    random_state=0,
)

# Per-iteration progress: call .mice(iterations=1) in a loop so we get
# timing and ETA printed between each pass. miceforest preserves state
# across multiple .mice() calls so this is equivalent to a single call
# with iterations=N but with visibility.
t0 = time.time()
for it in range(1, ITERATIONS + 1):
    t_iter = time.time()
    kernel.mice(iterations=1, verbose=False)
    iter_time = time.time() - t_iter
    elapsed = time.time() - t0
    remaining_eta = (ITERATIONS - it) * iter_time
    print(f'  iter {it}/{ITERATIONS}  ({iter_time:5.1f}s, '
          f'elapsed {elapsed/60:4.1f} min, ETA {remaining_eta/60:4.1f} min)')

print(f'\\nMICE done in {(time.time() - t0)/60:.1f} min')

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

# Locate the run-MICE cell (find by content)
target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'mf.ImputationKernel' in s and 'imputed_datasets' in s:
        target = i
        break

if target is None:
    print('ERROR: run-MICE cell not found.')
    sys.exit(1)

nb['cells'][target]['source'] = NEW_RUN_MICE.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

# Also: e2's M_IMPUTATIONS=10 is set earlier in a different cell. Override it
# in this cell to ensure consistency, OR find and patch that cell. Let's
# patch the earlier reference too to keep things consistent.
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'M_IMPUTATIONS = 10' in s and i != target:
        s_new = s.replace('M_IMPUTATIONS = 10',
                          'M_IMPUTATIONS = 5  # was 10; reduced for speed, re-run with 10 for paper')
        nb['cells'][i]['source'] = s_new.splitlines(keepends=True)
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        print(f'cell {i}: M_IMPUTATIONS init reduced to 5.')

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: run-MICE replaced with per-iteration progress + M=5, iter=5.')
