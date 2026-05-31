"""Apply three fixes to e2_data_prep.ipynb:
  A. Replace current HIC list with hardcoded 1995 HIC (sovereign states only).
  B. Remove the ECI filter (let trade-merge inner join cut countries).
  C. Sanitise column names before MICE to bypass LightGBM JSON-char restriction.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e2_data_prep.ipynb'

# ─────────────────────────────────────────────────────────────────────────────
# Fix A: Replace current-HIC block with hardcoded 1995 HIC list
# ─────────────────────────────────────────────────────────────────────────────
HIC_OLD_SUBSTR = """try:
    import wbgapi as wb"""

HIC_NEW_CODE = '''# Hardcoded 1995 World Bank high-income country (HIC) classification.
# Threshold at the time: 1993 GNI per capita > USD 8,955.
# Includes only sovereign states; dependent territories are excluded by the
# preceding territories filter regardless.
# Source: WB Country and Lending Groups historical classification, 1995.
HIC_1995 = {
    'AUS',  # Australia
    'AUT',  # Austria
    'BEL',  # Belgium
    'BHS',  # Bahamas
    'BRN',  # Brunei
    'CAN',  # Canada
    'CHE',  # Switzerland
    'CYP',  # Cyprus
    'DEU',  # Germany
    'DNK',  # Denmark
    'ESP',  # Spain
    'FIN',  # Finland
    'FRA',  # France
    'GBR',  # United Kingdom
    'GRC',  # Greece
    'IRL',  # Ireland
    'ISL',  # Iceland
    'ISR',  # Israel
    'ITA',  # Italy
    'JPN',  # Japan
    'KOR',  # Korea, Rep.
    'LIE',  # Liechtenstein (microstate; will also be dropped by territory list if present)
    'LUX',  # Luxembourg
    'MCO',  # Monaco
    'NLD',  # Netherlands
    'NOR',  # Norway
    'NZL',  # New Zealand
    'PRT',  # Portugal
    'SGP',  # Singapore
    'SMR',  # San Marino
    'SWE',  # Sweden
    'USA',  # United States
    # Gulf states 1995-HIC (kept by Gulf override below)
    'ARE',  # United Arab Emirates
    'BHR',  # Bahrain
    'KWT',  # Kuwait
    'QAT',  # Qatar
    # Other small high-income economies in 1995
    'ATG',  # Antigua and Barbuda
    'BRB',  # Barbados
}

hic_codes = HIC_1995
print(f'1995 HIC list (sovereign): {len(hic_codes)} countries')
'''

# Find the start of the wbgapi block and the end of the fallback else branch
# We replace the entire HIC-lookup block with the static list above.
WB_LOOKUP_BLOCK_OLD = '''# Current WB high-income classification (via wbgapi).
# Faithful to v2: applies present-day income groups, not 1995. Combined with the
# Gulf override, this yields the same effective filter as v2 produced.'''

WB_LOOKUP_BLOCK_NEW = '''# 1995 World Bank high-income classification (sovereign states only).
# This is hardcoded because WB does not expose historical income classifications
# via API. The list captures countries that were already high-income at the
# start of the panel; post-1995 promotions to HIC (e.g. Czechia, Estonia,
# Lithuania, Chile, Uruguay) remain in the sample so their full development
# trajectories enter the resource-curse analysis.'''


# ─────────────────────────────────────────────────────────────────────────────
# Fix B: Remove ECI filter cell
# ─────────────────────────────────────────────────────────────────────────────
# Locate the cell containing the ECI filter and replace its content with a
# comment so the cell structure stays consistent (but does nothing).


# ─────────────────────────────────────────────────────────────────────────────
# Fix C: Sanitise column names for MICE
# ─────────────────────────────────────────────────────────────────────────────
# The "Run MICE" cell needs column-name sanitisation before passing data to
# miceforest/LightGBM, and restoration after extracting imputed datasets.


def patch():
    with open(NB) as f:
        nb = json.load(f)

    # ── Fix A: HIC block ──
    hic_cell_idx = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'wbgapi' in s and 'HIC' in s and 'incomeLevel' in s:
            hic_cell_idx = i
            break

    if hic_cell_idx is None:
        print('ERROR: could not find HIC cell.')
        sys.exit(1)

    old_src = ''.join(nb['cells'][hic_cell_idx]['source'])
    new_src = HIC_NEW_CODE + '''
# Effective exclusion: HIC minus Gulf override
hic_exclude = hic_codes - GULF_STATES
print(f'HIC after Gulf override: excluding {len(hic_exclude)} countries')
'''

    nb['cells'][hic_cell_idx]['source'] = new_src.splitlines(keepends=True)
    nb['cells'][hic_cell_idx]['outputs'] = []
    nb['cells'][hic_cell_idx]['execution_count'] = None
    print(f'Fix A: HIC cell {hic_cell_idx} replaced with 1995 HIC list.')

    # ── Fix B: Remove ECI filter cell ──
    eci_cell_idx = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'eci_avail = M.groupby' in s:
            eci_cell_idx = i
            break

    if eci_cell_idx is None:
        print('WARNING: ECI filter cell not found (already removed?).')
    else:
        nb['cells'][eci_cell_idx]['source'] = [
            '# ECI filter removed: countries without trade data are dropped at the\n',
            '# inner merge in the next cell, which is the more principled cut.\n',
            "print('ECI filter skipped — trade-merge inner join handles country coverage')\n",
        ]
        nb['cells'][eci_cell_idx]['outputs'] = []
        nb['cells'][eci_cell_idx]['execution_count'] = None
        print(f'Fix B: ECI filter cell {eci_cell_idx} neutralised.')

    # ── Fix C: Sanitise column names for MICE ──
    # Find the cell that does "Columns to pass to MICE" + demeaning
    mice_prep_idx = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'country_means = df.groupby' in s and 'df_demeaned[mice_cols]' in s:
            mice_prep_idx = i
            break

    if mice_prep_idx is None:
        print('ERROR: MICE prep cell not found.')
        sys.exit(1)

    new_mice_prep = '''# Columns to pass to MICE
ID_COLS = ['Country Code', 'Country Name', 'Year']
SKIP_FROM_MICE = {
    # Don't synthesize the forecast benchmark
    'Atlas growth projection',
    # Already filled in pre-fill stage
    'Civil war', 'Production_Hydrocarbons', 'Production_Metals', 'Production_Others',
    'Use of IMF credit (DOD, current US$)',
    # Trade USD aggregates: scale-dependent, leave NaN; share columns will be imputed
}
SKIP_FROM_MICE.update(TRADE_USD_COLS)

CATEGORICAL_COLS = {
    'Landlocked', 'post_2019',
    'Hydrocarbons_Dominant', 'Subsoil_Metals_Dominant', 'Precious_Metals_Dominant',
}
CATEGORICAL_COLS &= set(df.columns)

mice_cols = [c for c in df.columns
             if c not in ID_COLS
             and c not in SKIP_FROM_MICE
             and c not in CATEGORICAL_COLS]
print(f'MICE numeric columns: {len(mice_cols)}')
print(f'MICE categorical columns: {len(CATEGORICAL_COLS)}')

# Pre-step: country-demeaning on numeric columns
country_means = df.groupby('Country Code')[mice_cols].transform('mean')
df_demeaned = df.copy()
df_demeaned[mice_cols] = df[mice_cols] - country_means

# Cast categorical columns
for c in CATEGORICAL_COLS:
    df_demeaned[c] = df_demeaned[c].astype('category')

# LightGBM (under miceforest) does not accept JSON special characters in
# column names: ":", ",", "(", ")", "[", "]", "{", "}", quotes, slashes.
# Sanitise on the way in, restore on the way out.
import re as _re
def _sanitise(name):
    s = _re.sub(r'[^A-Za-z0-9_]', '_', name)
    return _re.sub(r'_+', '_', s).strip('_')

SAN_MAP = {c: _sanitise(c) for c in mice_cols + list(CATEGORICAL_COLS)}
# Check for sanitisation collisions
collisions = [k for k, v in SAN_MAP.items() if list(SAN_MAP.values()).count(v) > 1]
if collisions:
    print(f'WARNING: {len(collisions)} name collisions after sanitisation. Adding hash suffix.')
    # Add hash to disambiguate
    seen = {}
    for k in list(SAN_MAP.keys()):
        v = SAN_MAP[k]
        if v in seen.values():
            SAN_MAP[k] = f'{v}_{abs(hash(k)) % 10000:04d}'
        seen[k] = SAN_MAP[k]
SAN_INV = {v: k for k, v in SAN_MAP.items()}

print(f'Demeaning: country means computed. Shape: {df_demeaned.shape}')
print(f'Sanitised {len(SAN_MAP)} column names for MICE')
'''
    nb['cells'][mice_prep_idx]['source'] = new_mice_prep.splitlines(keepends=True)
    nb['cells'][mice_prep_idx]['outputs'] = []
    nb['cells'][mice_prep_idx]['execution_count'] = None
    print(f'Fix C.1: MICE prep cell {mice_prep_idx} updated with sanitisation.')

    # ── Fix C.2: Run-MICE cell needs to use sanitised names ──
    run_mice_idx = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'mf.ImputationKernel' in s and 'imputed_datasets' in s:
            run_mice_idx = i
            break

    if run_mice_idx is None:
        print('ERROR: run-MICE cell not found.')
        sys.exit(1)

    new_run_mice = '''# Run MICE
mice_input = df_demeaned[mice_cols + list(CATEGORICAL_COLS)].copy()
# Apply name sanitisation
mice_input = mice_input.rename(columns=SAN_MAP)

print(f'Starting miceforest: M={M_IMPUTATIONS} imputations, '
      f'{len(mice_cols) + len(CATEGORICAL_COLS)} columns, {len(mice_input):,} rows')
t0 = time.time()

kernel = mf.ImputationKernel(
    mice_input,
    num_datasets=M_IMPUTATIONS,
    random_state=0,
)
kernel.mice(iterations=5, verbose=False)
print(f'MICE done in {time.time() - t0:.1f}s')

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
    nb['cells'][run_mice_idx]['source'] = new_run_mice.splitlines(keepends=True)
    nb['cells'][run_mice_idx]['outputs'] = []
    nb['cells'][run_mice_idx]['execution_count'] = None
    print(f'Fix C.2: run-MICE cell {run_mice_idx} updated with rename().')

    # ── Fix C.3: Validation cell also needs sanitisation since it runs another MICE ──
    val_idx = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c.get('source', []))
        if 'val_kernel = mf.ImputationKernel' in s and 'masked' in s:
            val_idx = i
            break

    if val_idx is None:
        print('WARNING: validation cell not found.')
    else:
        val_src = ''.join(nb['cells'][val_idx]['source'])
        # Insert sanitisation rename + restore around the val_kernel block
        old_val_block = '''val_kernel = mf.ImputationKernel(mask_input, num_datasets=1, random_state=42)
val_kernel.mice(iterations=5, verbose=False)
val_filled = val_kernel.complete_data(dataset=0)'''
        new_val_block = '''mask_input_san = mask_input.rename(columns=SAN_MAP)
val_kernel = mf.ImputationKernel(mask_input_san, num_datasets=1, random_state=42)
val_kernel.mice(iterations=5, verbose=False)
val_filled = val_kernel.complete_data(dataset=0).rename(columns=SAN_INV)'''
        if old_val_block in val_src:
            val_src = val_src.replace(old_val_block, new_val_block)
            nb['cells'][val_idx]['source'] = val_src.splitlines(keepends=True)
            nb['cells'][val_idx]['outputs'] = []
            nb['cells'][val_idx]['execution_count'] = None
            print(f'Fix C.3: validation cell {val_idx} updated.')
        else:
            print(f'WARNING: validation cell {val_idx} pattern not found.')

    # Write back
    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Saved {NB}')


if __name__ == '__main__':
    patch()
