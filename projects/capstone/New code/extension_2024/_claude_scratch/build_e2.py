"""Build e2_data_prep.ipynb from scratch.

This script constructs the notebook programmatically as JSON, which is cleaner
than dumping a long string with all the JSON-quoting headaches. Once written,
the notebook is a normal .ipynb that can be opened, run, and edited in Jupyter.

Sections:
  1. Setup and imports
  2. Sovereign-state and HIC filtering
  3. Load Master_extended and trade_metrics
  4. Merge, then population merge
  5. Drop deprecated columns (5 rents + 2 adjusted savings + 2 reserves)
  6. Pre-fill: Civil war=0, Production_*=0, Use of IMF credit=0
  7. post_2019 dummy and pre-computed interactions
  8. Stage 1 imputation: linear interp interior, ffill trailing, bfill leading
  9. Stage 2 imputation: miceforest M=10 with country-demeaning
  10. Validation: random-mask + per-variable median % error
  11. Error-weighted reliance score
  12. Write Master_v2_imputations.parquet + diagnostics CSV
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e2_data_prep.ipynb')


def md(text: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text.splitlines(keepends=True),
    }


cells = []

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('''# e2 — Master_v2 construction

Merges:
- `intermediary/master_data_wide.csv` (Master_extended, from e1)
- `intermediary/trade_metrics.csv` (from e1b)
- `rawdata/PopulationWDI.csv`

Applies:
- Sovereign-states filter (drops dependent territories and regional aggregates)
- Current WB high-income classification exclusion (with Gulf-state override)
- Variable cleanup: drops WB rents (replaced by trade shares), 2 Adjusted Savings, 2 Reserves
- Pre-fills: `Civil war = 0`, `Production_* = 0`, `Use of IMF credit = 0`
- `post_2019` dummy and three pre-computed COVID interaction columns
- Two-stage imputation:
  - Stage 1: linear interpolation interior gaps (MAX_GAP=3), ffill trailing (limit=3), bfill leading (limit=3)
  - Stage 2: miceforest with M=10 random-forest imputations, country-demeaned to bias toward within-country variation
- Validation: random-cell masking + per-variable median % error
- Error-weighted reliance score per country

Outputs:
- `intermediary/Master_v2_imputations.parquet` — M=10 stacked panels, keyed by `imputation_id`
- `intermediary/Master_v2_diagnostics.csv` — per-variable validation accuracy + per-country reliance
- `intermediary/Master_v2_observed.csv` — observed cells only (no imputation), for robustness checks

Downstream (e5/e6) uses `_mice_pool.py` for Rubin's-rules pooling.
'''))

cells.append(md('## 1. Setup'))

cells.append(code('''import os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='miceforest')

sys.path.insert(0, '.')
import _config as cfg
from standardize_country import ISO3_TO_WB

EXT_DIR  = Path('.').resolve()
INTER    = EXT_DIR / 'intermediary'
RAWDATA  = EXT_DIR / 'rawdata'

print(f'Working dir: {EXT_DIR}')
print(f'YEAR_MIN, YEAR_MAX = {cfg.YEAR_MIN}, {cfg.YEAR_MAX}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 2. Sovereign-state filter'))

cells.append(code('''# ISO3 codes that are dependent territories or non-sovereign entities.
# Exclude: HKG, MAC (SARs of China); VAT (microstate with no economy).
# Include: TWN (de facto sovereign), PSE/XKX (UN observers / partial recognition).
DEPENDENT_TERRITORIES = {
    'ABW',  # Aruba
    'AIA',  # Anguilla
    'ASM',  # American Samoa
    'BES',  # Bonaire, Sint Eustatius and Saba
    'BMU',  # Bermuda
    'BVT',  # Bouvet Island
    'CCK',  # Cocos (Keeling) Islands
    'COK',  # Cook Islands
    'CUW',  # Curaçao
    'CXR',  # Christmas Island
    'CYM',  # Cayman Islands
    'ESH',  # Western Sahara
    'FLK',  # Falkland Islands
    'FRO',  # Faroe Islands
    'GGY',  # Guernsey
    'GIB',  # Gibraltar
    'GLP',  # Guadeloupe
    'GRL',  # Greenland
    'GUF',  # French Guiana
    'GUM',  # Guam
    'HKG',  # Hong Kong SAR (excluded per sovereignty principle)
    'IMN',  # Isle of Man
    'IOT',  # British Indian Ocean Territory
    'JEY',  # Jersey
    'MAC',  # Macao SAR
    'MAF',  # Saint Martin (French)
    'MNP',  # Northern Mariana Islands
    'MSR',  # Montserrat
    'MTQ',  # Martinique
    'MYT',  # Mayotte
    'NCL',  # New Caledonia
    'NFK',  # Norfolk Island
    'NIU',  # Niue
    'PCN',  # Pitcairn
    'PRI',  # Puerto Rico
    'PYF',  # French Polynesia
    'REU',  # Réunion
    'SGS',  # South Georgia
    'SHN',  # Saint Helena
    'SJM',  # Svalbard and Jan Mayen
    'SPM',  # Saint Pierre and Miquelon
    'SXM',  # Sint Maarten
    'TCA',  # Turks and Caicos
    'TKL',  # Tokelau
    'UMI',  # US Minor Outlying Islands
    'VAT',  # Vatican City (no economy)
    'VGB',  # British Virgin Islands
    'VIR',  # US Virgin Islands
    'WLF',  # Wallis and Futuna
}

# Gulf state guarantee override (kept despite HIC classification)
GULF_STATES = {'ARE', 'BHR', 'KWT', 'OMN', 'QAT', 'SAU', 'IRQ', 'IRN', 'YEM'}

print(f'Dependent territories blacklist: {len(DEPENDENT_TERRITORIES)} codes')
print(f'Gulf-state override: {len(GULF_STATES)} codes')
'''))

cells.append(code('''# Current WB high-income classification (via wbgapi).
# Faithful to v2: applies present-day income groups, not 1995. Combined with the
# Gulf override, this yields the same effective filter as v2 produced.
try:
    import wbgapi as wb
    _eco = wb.economy.DataFrame()
    hic_codes = set(_eco[_eco['incomeLevel'] == 'HIC'].index)
    print(f'WB HIC list: {len(hic_codes)} countries')
except Exception as e:
    print(f'WARNING: wbgapi unavailable ({e}). Falling back to hardcoded HIC list.')
    # Hardcoded fallback (WB HIC as of 2024). Update if WB reclassifies.
    hic_codes = {
        'AND', 'ARE', 'ATG', 'AUS', 'AUT', 'BEL', 'BHR', 'BHS', 'BMU', 'BRB',
        'BRN', 'CAN', 'CHE', 'CHI', 'CHL', 'CYM', 'CYP', 'CZE', 'DEU', 'DNK',
        'ESP', 'EST', 'FIN', 'FRA', 'FRO', 'GBR', 'GIB', 'GRC', 'GRL', 'GUM',
        'GUY', 'HKG', 'HRV', 'HUN', 'IMN', 'IRL', 'ISL', 'ISR', 'ITA', 'JPN',
        'KNA', 'KOR', 'KWT', 'LIE', 'LTU', 'LUX', 'LVA', 'MAC', 'MCO', 'MLT',
        'NCL', 'NLD', 'NOR', 'NRU', 'NZL', 'OMN', 'PAN', 'POL', 'PRI', 'PRT',
        'PYF', 'QAT', 'ROU', 'SAU', 'SGP', 'SMR', 'SVK', 'SVN', 'SWE', 'SXM',
        'SYC', 'TCA', 'TTO', 'TWN', 'URY', 'USA', 'VIR',
    }
    print(f'HIC fallback: {len(hic_codes)} countries')

# Effective exclusion: HIC minus Gulf override
hic_exclude = hic_codes - GULF_STATES
print(f'HIC after Gulf override: excluding {len(hic_exclude)} countries')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 3. Load Master_extended and trade_metrics'))

cells.append(code('''# Master_extended (from e1)
M = pd.read_csv(INTER / 'master_data_wide.csv')
print(f'Master_extended: {len(M):,} rows, {len(M.columns)} cols, '
      f'{M["Country Code"].nunique()} countries, {M["Year"].min()}-{M["Year"].max()}')

# Trade metrics (from e1b)
T = pd.read_csv(INTER / 'trade_metrics.csv')
T = T.rename(columns={'reporterISO': 'Country Code', 'period': 'Year'})
T['Year'] = T['Year'].astype(int)
print(f'Trade metrics: {len(T):,} rows, {len(T.columns)} cols, '
      f'{T["Country Code"].nunique()} countries, {T["Year"].min()}-{T["Year"].max()}')

# Clip both to YEAR_MIN..YEAR_MAX
M = M[(M['Year'] >= cfg.YEAR_MIN) & (M['Year'] <= cfg.YEAR_MAX)]
T = T[(T['Year'] >= cfg.YEAR_MIN) & (T['Year'] <= cfg.YEAR_MAX)]
print(f'After year clip: M={len(M):,}, T={len(T):,}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 4. Sovereign filter + HIC filter, then merge'))

cells.append(code('''# Apply sovereign + HIC filters to Master_extended
pre_n = M['Country Code'].nunique()
M = M[~M['Country Code'].isin(DEPENDENT_TERRITORIES)]
mid_n = M['Country Code'].nunique()
M = M[~M['Country Code'].isin(hic_exclude)]
post_n = M['Country Code'].nunique()

print(f'Country count: {pre_n} -> {mid_n} after territories -> {post_n} after HIC (Gulf override applied)')
print(f'Master after filters: {len(M):,} rows, {post_n} countries')
'''))

cells.append(code('''# Atlas ECI coverage: countries with at least one non-NaN eci_hs92
eci_avail = M.groupby('Country Code')['Economic Complexity Index'].apply(
    lambda x: x.notna().any()
)
no_eci = eci_avail[~eci_avail].index.tolist()
print(f'Dropping {len(no_eci)} countries with no ECI data: {sorted(no_eci)[:20]}'
      + ('...' if len(no_eci) > 20 else ''))

M = M[M['Country Code'].isin(eci_avail[eci_avail].index)]
print(f'After ECI filter: {len(M):,} rows, {M["Country Code"].nunique()} countries')
'''))

cells.append(code('''# Merge Master + Trade (inner join on Country Code, Year)
# Inner because we need both sides for the headline analysis
df = pd.merge(M, T, on=['Country Code', 'Year'], how='inner',
              suffixes=('', '_trade'))
print(f'Merged: {len(df):,} rows, {len(df.columns)} cols, {df["Country Code"].nunique()} countries')

# Diagnostic: countries in M but not in T (and vice versa)
m_only = set(M['Country Code'].unique()) - set(T['Country Code'].unique())
t_only = set(T['Country Code'].unique()) - set(M['Country Code'].unique())
print(f'In Master only (no trade data): {len(m_only)}')
print(f'In Trade only (no Master, expected — trade has 207 reporters): {len(t_only)}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 5. Population merge'))

cells.append(code('''pop_raw = pd.read_csv(RAWDATA / 'PopulationWDI.csv')
year_cols = [c for c in pop_raw.columns if c.startswith(tuple(str(y) for y in range(1990, 2030)))]
pop_long = pop_raw.melt(
    id_vars=['Series Name', 'Series Code', 'Country Name', 'Country Code'],
    value_vars=year_cols,
    var_name='Year_raw',
    value_name='Population',
)
pop_long['Year'] = pop_long['Year_raw'].str.extract(r'(\\d{4})').astype(int)
pop_long['Population'] = pd.to_numeric(pop_long['Population'], errors='coerce')
pop_long = pop_long[['Country Code', 'Year', 'Population']]
pop_long = pop_long[
    (pop_long['Year'] >= cfg.YEAR_MIN) & (pop_long['Year'] <= cfg.YEAR_MAX)
]

df = df.merge(pop_long, on=['Country Code', 'Year'], how='left')
print(f'After population merge: {len(df):,} rows, {len(df.columns)} cols')
print(f'Population NaN: {df["Population"].isna().sum()}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 6. Drop deprecated columns'))

cells.append(code('''# Replaced by trade_metrics shares (wide_resource_share, hydrocarbon_share, ores_share)
WB_RENTS_TO_DROP = [
    'Total natural resources rents (% of GDP)',
    'Oil rents (% of GDP)',
    'Natural gas rents (% of GDP)',
    'Mineral rents (% of GDP)',
    'Forestry rents (% of GDP)',
]

# Source-data cliff at 2022; replaced by no current variable
ADJUSTED_SAVINGS_TO_DROP = [
    'Adjusted savings: gross savings (% of GNI)',
    'Adjusted savings: total (current US$)',
    'Adjusted savings: natural resources depletion (% of GNI)',  # also drop the depletion column
]

# >98% missing; not usable as a panel variable
RESERVES_TO_DROP = [
    'Reserves_Metals',
    'Reserves_Others',
]

to_drop = WB_RENTS_TO_DROP + ADJUSTED_SAVINGS_TO_DROP + RESERVES_TO_DROP
existing = [c for c in to_drop if c in df.columns]
missing  = [c for c in to_drop if c not in df.columns]
df = df.drop(columns=existing)
print(f'Dropped {len(existing)} columns: {existing}')
if missing:
    print(f'(not found in df: {missing})')
print(f'After drops: {len(df.columns)} cols')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 7. Pre-fills: zero where NaN means absence'))

cells.append(code('''# Civil war: V-Dem binary. NaN ~ no recorded conflict in country-year -> 0.
if 'Civil war' in df.columns:
    pre = df['Civil war'].isna().sum()
    df['Civil war'] = df['Civil war'].fillna(0)
    print(f'Civil war: filled {pre} NaN with 0')

# Production_* columns: countries with no production in this category genuinely
# produce zero, not "unknown". Distinguishing requires a coverage flag.
for col in ['Production_Hydrocarbons', 'Production_Metals', 'Production_Others']:
    if col in df.columns:
        pre = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        print(f'{col}: filled {pre} NaN with 0')

# Use of IMF credit: NaN ~ not in IMF database = no outstanding IMF debt = 0.
# (Faithful to v2 logic.)
if 'Use of IMF credit (DOD, current US$)' in df.columns:
    pre = df['Use of IMF credit (DOD, current US$)'].isna().sum()
    df['Use of IMF credit (DOD, current US$)'] = df['Use of IMF credit (DOD, current US$)'].fillna(0)
    print(f'Use of IMF credit: filled {pre} NaN with 0')

print()
print('NB: Death rates pre-fill from v2 is NOT applied — NaN means missing, not zero deaths.')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 8. `post_2019` dummy + pre-computed COVID interactions'))

cells.append(code('''df['post_2019'] = (df['Year'] >= 2020).astype(int)

# Pre-compute interactions: post_2019 × {hydrocarbon, ores, base_metals}_share
for share_col, name in [
    ('hydrocarbon_share', 'post2019_x_hydrocarbon_share'),
    ('ores_share',         'post2019_x_ores_share'),
    ('base_metals_share',  'post2019_x_base_metals_share'),
]:
    if share_col in df.columns:
        df[name] = df['post_2019'] * df[share_col]
        print(f'  Added {name}')
    else:
        print(f'  WARNING: {share_col} not in df, skipping {name}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('''## 9. Stage 1 imputation: per-country interpolation

- Interior gaps (within a country's time series): linear interpolation, MAX_GAP=3
- Leading gaps (first observed value > 3 years into panel): bfill, limit=3
- Trailing gaps (last observed value > 3 years before panel end): ffill, limit=3

Avoids COVID-era linear extrapolation artifacts. Variables with no observations
at all in a country are left NaN for Stage 2 (MICE) to handle cross-country.'''))

cells.append(code('''MAX_GAP = 3

# Variables to EXCLUDE from interpolation (time-invariant, dummies, pre-filled)
EXCLUDE_FROM_INTERP = {
    'Country Code', 'Country Name', 'Year', 'Population',
    'Landlocked', 'post_2019', 'Civil war',
    'Hydrocarbons_Dominant', 'Subsoil_Metals_Dominant', 'Precious_Metals_Dominant',
    'Production_Hydrocarbons', 'Production_Metals', 'Production_Others',
    'Use of IMF credit (DOD, current US$)',
}

# Trade USD aggregates: large, scale-dependent. Better imputed cross-country
# only if absolutely necessary; for now, exclude from Stage 1 (can leave NaN).
TRADE_USD_COLS = [c for c in df.columns if c.endswith('_usd')] + [
    c for c in df.columns if c.isdigit()  # HS chapter raw value cols if any survived
]
EXCLUDE_FROM_INTERP.update(TRADE_USD_COLS)

interp_cols = [c for c in df.columns if c not in EXCLUDE_FROM_INTERP]
print(f'Interpolating {len(interp_cols)} columns')

# Track pre-Stage-1 NaN count
pre_nan = df[interp_cols].isna().sum().sum()

df = df.sort_values(['Country Code', 'Year']).reset_index(drop=True)

for col in interp_cols:
    # Per-country interpolation with three modes
    def _stage1(group):
        s = group[col].copy()
        # Interior: linear, limited gap
        s = s.interpolate(method='linear', limit=MAX_GAP, limit_area='inside')
        # Trailing: ffill, limit MAX_GAP
        s = s.ffill(limit=MAX_GAP)
        # Leading: bfill, limit MAX_GAP
        s = s.bfill(limit=MAX_GAP)
        return s

    df[col] = df.groupby('Country Code', group_keys=False)[col].transform(
        lambda s: (
            s.interpolate(method='linear', limit=MAX_GAP, limit_area='inside')
             .ffill(limit=MAX_GAP)
             .bfill(limit=MAX_GAP)
        )
    )

post_nan = df[interp_cols].isna().sum().sum()
print(f'Stage 1: NaN reduced {pre_nan:,} -> {post_nan:,} ({pre_nan - post_nan:,} filled)')
print(f'Remaining NaN cells: {post_nan:,} (for Stage 2 MICE to handle)')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('''## 10. Stage 2 imputation: miceforest M=10 with country-demeaning

- Library: `miceforest` (random forest conditional model, native multiple imputation)
- M = 10 imputations with deterministic seeds 0..9
- Pre-step: subtract country mean from each numeric column → run MICE on
  deviations-from-country-mean → add country mean back. This bias toward
  within-country variation addresses the pooled-panel-KNN critique.
- Categorical columns (`Landlocked`, dominance dummies, `post_2019`, `Civil war`)
  are passed as `category` dtype so miceforest uses appropriate splits.'''))

cells.append(code('''# Check for miceforest
try:
    import miceforest as mf
    print(f'miceforest version: {mf.__version__}')
except ImportError:
    print('miceforest not installed. Install with:')
    print('  pip install miceforest --break-system-packages')
    raise

M_IMPUTATIONS = 10
'''))

cells.append(code('''# Columns to pass to MICE
ID_COLS = ['Country Code', 'Country Name', 'Year']
SKIP_FROM_MICE = {
    # Don't synthesize the forecast benchmark
    'growth_proj',
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

print(f'Demeaning: country means computed. Shape: {df_demeaned.shape}')
'''))

cells.append(code('''# Pre-MICE diagnostic: where are NaNs now?
pre_mice_nan = df_demeaned[mice_cols + list(CATEGORICAL_COLS)].isna().sum()
pre_mice_nan = pre_mice_nan[pre_mice_nan > 0].sort_values(ascending=False)
print('Remaining NaN cells per column (top 20, going into MICE):')
print(pre_mice_nan.head(20).to_string())
print(f'\\nTotal NaN: {pre_mice_nan.sum():,}')
print(f'Total cells: {len(df_demeaned) * (len(mice_cols) + len(CATEGORICAL_COLS)):,}')
print(f'Missing rate: {100 * pre_mice_nan.sum() / (len(df_demeaned) * (len(mice_cols) + len(CATEGORICAL_COLS))):.2f}%')
'''))

cells.append(code('''# Run MICE
mice_input = df_demeaned[mice_cols + list(CATEGORICAL_COLS)].copy()

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

# Extract all M imputed datasets
imputed_datasets = []
for m in range(M_IMPUTATIONS):
    sub = kernel.complete_data(dataset=m).copy()
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
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 11. Validation: random-mask + per-variable median % error'))

cells.append(code('''# For each numeric column, mask 5% of known values, predict, compute median % error.
# Uses a single MICE run for speed; this is the MAR-style validation. MNAR is the
# next iteration's work.

np.random.seed(0)
MASK_FRAC = 0.05
mask_input = mice_input.copy()
masked = {}  # col -> [(row_idx, true_value), ...]

for col in mice_cols:
    obs_idx = mask_input[mask_input[col].notna()].index.tolist()
    n_mask = int(len(obs_idx) * MASK_FRAC)
    if n_mask == 0:
        continue
    sample = np.random.choice(obs_idx, size=n_mask, replace=False)
    truth = mask_input.loc[sample, col].copy()
    mask_input.loc[sample, col] = np.nan
    masked[col] = (sample, truth)

print(f'Masked {sum(len(v[0]) for v in masked.values()):,} cells across {len(masked)} columns')

# Run single-imputation validation
val_kernel = mf.ImputationKernel(mask_input, num_datasets=1, random_state=42)
val_kernel.mice(iterations=5, verbose=False)
val_filled = val_kernel.complete_data(dataset=0)

# Compute median % error per column
validation_rows = []
for col, (idx, truth) in masked.items():
    pred = val_filled.loc[idx, col]
    # Median absolute percent error
    mask_nonzero = truth.abs() > 1e-8
    if mask_nonzero.sum() == 0:
        median_pct_error = np.nan
    else:
        pct_err = np.abs(pred[mask_nonzero] - truth[mask_nonzero]) / np.abs(truth[mask_nonzero]) * 100
        median_pct_error = float(pct_err.median())
    mae = float((pred - truth).abs().median())
    validation_rows.append({
        'variable': col,
        'n_masked': len(idx),
        'median_pct_error': median_pct_error,
        'median_abs_error': mae,
    })

validation_df = pd.DataFrame(validation_rows).sort_values('median_pct_error', ascending=False)
print('\\nValidation (top 15 by error):')
print(validation_df.head(15).to_string(index=False))
print('\\nValidation (bottom 10 by error, most reliable):')
print(validation_df.tail(10).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 12. Error-weighted reliance score'))

cells.append(code('''# Per country, what fraction of cells were filled by Stage 2, and weight by
# per-variable validation error. High weighted reliance = country's data is
# more synthetic AND that synthesis is unreliable.

# Recompute: which cells were NaN before MICE?
pre_mice_mask = df_demeaned[mice_cols + list(CATEGORICAL_COLS)].isna()

# Per-country, per-variable count of imputed cells
fill_counts = pd.concat([
    df[['Country Code']].reset_index(drop=True),
    pre_mice_mask.reset_index(drop=True),
], axis=1).groupby('Country Code').sum()

# Per-variable median % error map (numeric vars only; categoricals have NaN)
err_map = dict(zip(validation_df['variable'], validation_df['median_pct_error']))

# Weighted reliance: sum over variables of (filled_cells × pct_error_for_that_var)
# Then divide by total cells in that country.
total_cells_per_country = df.groupby('Country Code').size() * (len(mice_cols) + len(CATEGORICAL_COLS))

reliance_records = []
for cc in fill_counts.index:
    row = fill_counts.loc[cc]
    weighted_score = 0.0
    raw_count = 0
    for col, n in row.items():
        if n == 0:
            continue
        raw_count += n
        err = err_map.get(col, np.nan)
        if pd.notna(err):
            weighted_score += n * err
    raw_pct = 100 * raw_count / total_cells_per_country[cc]
    weighted_pct = weighted_score / total_cells_per_country[cc] if total_cells_per_country[cc] else 0
    reliance_records.append({
        'Country Code': cc,
        'raw_imputed_pct':    raw_pct,
        'weighted_score':      weighted_pct,
    })

reliance_df = pd.DataFrame(reliance_records).sort_values('weighted_score', ascending=False)
print('Top 15 countries by weighted reliance (most-synthetic data):')
print(reliance_df.head(15).to_string(index=False))
print()
print('Bottom 10 countries by weighted reliance (most-observed data):')
print(reliance_df.tail(10).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 13. Write outputs'))

cells.append(code('''# Stack the M datasets into one parquet
stacked = pd.concat(imputed_datasets, ignore_index=True)
print(f'Stacked: {len(stacked):,} rows ({M_IMPUTATIONS} x {len(stacked) // M_IMPUTATIONS} country-years), '
      f'{len(stacked.columns)} cols')

out_parquet = INTER / 'Master_v2_imputations.parquet'
stacked.to_parquet(out_parquet, index=False)
print(f'Wrote {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)')

# Diagnostic CSV: per-variable accuracy + per-country reliance
diag = {
    'validation': validation_df,
    'reliance': reliance_df,
}
out_csv = INTER / 'Master_v2_diagnostics.csv'
with open(out_csv, 'w') as f:
    f.write('# Per-variable validation (MAR-style random masking, 5%)\\n')
    validation_df.to_csv(f, index=False)
    f.write('\\n# Per-country reliance (raw + error-weighted)\\n')
    reliance_df.to_csv(f, index=False)
print(f'Wrote {out_csv}')

# Observed-only panel (no imputation) for downstream robustness
observed_only = df.copy()
out_observed = INTER / 'Master_v2_observed.csv'
observed_only.to_csv(out_observed, index=False)
print(f'Wrote {out_observed} ({len(observed_only):,} rows, {len(observed_only.columns)} cols, '
      f'NaN cells: {observed_only.isna().sum().sum():,})')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 14. Summary'))

cells.append(code('''print('=' * 70)
print('Master_v2 construction summary')
print('=' * 70)
print(f'Year range:           {cfg.YEAR_MIN}-{cfg.YEAR_MAX}')
print(f'Countries:            {df["Country Code"].nunique()}')
print(f'Country-years:        {len(df):,}')
print(f'Columns:              {len(df.columns)}')
print(f'Imputations (M):      {M_IMPUTATIONS}')
print(f'Stacked rows:         {len(stacked):,}')
print()
print(f'Files:')
print(f'  intermediary/Master_v2_imputations.parquet  (M=10 stacked)')
print(f'  intermediary/Master_v2_diagnostics.csv      (validation + reliance)')
print(f'  intermediary/Master_v2_observed.csv         (observed-only, for robustness)')
print()
print('Next: e3_clusters / e4_ml / e5_regressions / e6_forecast — all need to')
print('loop over imputation_id and pool results via _mice_pool.py utility.')
'''))

# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.4',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Wrote {NB_PATH}')
print(f'Cells: {len(cells)} ({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')
