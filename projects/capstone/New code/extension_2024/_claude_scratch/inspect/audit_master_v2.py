"""Audit Master_v2_imputations.parquet for problems before re-running.

Looks for:
  1. Countries that shouldn't be in the panel (territories, defunct states,
     non-sovereign entities) — e.g. ANT (Netherlands Antilles)
  2. Negative shares where they shouldn't be
  3. Out-of-range values: share > 1, share < 0
  4. Anomalously extreme imputed values vs the observed range per variable
  5. Year coverage gaps per country
  6. Variable name issues (encoding, accidental duplicates)
  7. Cross-imputation variance per cell (where M=5 imputations diverge most)
  8. Patterns of NaN in observed-only data that the imputation may have papered over
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

EXT  = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
INTER = EXT / 'intermediary'

print('=' * 70)
print('Master_v2 audit')
print('=' * 70)

# Load all data
imputed = pd.read_parquet(INTER / 'Master_v2_imputations.parquet')
observed = pd.read_csv(INTER / 'Master_v2_observed.csv')
diag = pd.read_csv(INTER / 'Master_v2_diagnostics.csv', comment='#')

print(f'Imputed:   {len(imputed):,} rows, {len(imputed.columns)} cols, M={imputed["imputation_id"].nunique()}')
print(f'Observed:  {len(observed):,} rows, {len(observed.columns)} cols')
print()

# ── 1. Countries that shouldn't be there ──
print('=' * 70)
print('1. Country list audit')
print('=' * 70)

countries = sorted(observed['Country Code'].unique())
print(f'Total countries: {len(countries)}')
print()

# Known to-be-dropped codes
SUSPICIOUS = {
    'ANT': 'Netherlands Antilles (dissolved 2010)',
    'YUG': 'Yugoslavia (defunct)',
    'CSK': 'Czechoslovakia (defunct)',
    'DDR': 'East Germany (defunct)',
    'SUN': 'USSR (defunct)',
    'PSE': 'Palestine (borderline; UN observer)',
    'XKX': 'Kosovo (borderline; partial recognition)',
    'TWN': 'Taiwan (sovereignty question; usually kept)',
    'VAT': 'Vatican (microstate)',
    'MCO': 'Monaco (microstate, HIC 1995)',
    'SMR': 'San Marino (microstate)',
    'AND': 'Andorra (microstate; HIC borderline)',
    'LIE': 'Liechtenstein (microstate; HIC 1995)',
}

print('Suspicious country codes present:')
for code, reason in SUSPICIOUS.items():
    if code in countries:
        print(f'  {code}: {reason}')
print()

# Microstates with very low population that might be in by accident
pop_per_country = observed.groupby('Country Code')['Population'].mean()
print('Countries with mean population < 500,000 (possibly micro-states):')
small = pop_per_country[pop_per_country < 500_000].sort_values()
for code, pop in small.items():
    print(f'  {code}: {pop/1000:6.1f}k')

# ── 2. Share value range checks ──
print()
print('=' * 70)
print('2. Share variable range audit')
print('=' * 70)

share_cols = [c for c in observed.columns if c.endswith('_share') or c in ['Resource_HHI_trade']]
share_issues = []
for c in share_cols:
    if observed[c].isna().all():
        continue
    n_neg = (observed[c] < 0).sum()
    n_high = (observed[c] > 1).sum()
    if n_neg > 0 or n_high > 0:
        share_issues.append({
            'variable': c, 'n_negative': int(n_neg), 'n_above_1': int(n_high),
            'min': float(observed[c].min()), 'max': float(observed[c].max()),
        })

if share_issues:
    print('Share variables with out-of-range values:')
    print(pd.DataFrame(share_issues).to_string(index=False))
else:
    print('All share variables are in [0, 1].')

# Check imputed values for share violations
print()
print('Same audit on IMPUTED data (5x rows):')
imputed_share_issues = []
for c in share_cols:
    if c not in imputed.columns: continue
    sub = imputed[c]
    n_neg = (sub < 0).sum()
    n_high = (sub > 1).sum()
    if n_neg > 0 or n_high > 0:
        imputed_share_issues.append({
            'variable': c, 'n_negative': int(n_neg), 'n_above_1': int(n_high),
            'min': float(sub.min()), 'max': float(sub.max()),
        })

if imputed_share_issues:
    print('IMPUTED share variables with out-of-range values:')
    print(pd.DataFrame(imputed_share_issues).to_string(index=False))
else:
    print('All imputed share variables are in [0, 1].')

# ── 3. Anomalous imputed values relative to observed range ──
print()
print('=' * 70)
print('3. Anomalous imputed values per variable')
print('=' * 70)

key_vars = [
    'Economic Complexity Index', 'Human capital index',
    'GDP per capita (constant prices, PPP)',
    'Gross fixed capital formation, all, Constant prices, Percent of GDP',
    'Inflation, consumer prices (annual %)', 'Real interest rate (%)',
    'Political stability — estimate', 'Rule of law index',
    'Government revenue', 'Trade (% of GDP)',
    'wide_resource_share', 'hydrocarbon_share',
]

print(f'{"Variable":<50s} {"Obs min":>10s} {"Obs max":>10s} {"Imp min":>10s} {"Imp max":>10s} {"Anomaly":>10s}')
print('-' * 95)
for v in key_vars:
    if v not in imputed.columns or v not in observed.columns:
        continue
    obs_min, obs_max = observed[v].min(), observed[v].max()
    imp_min, imp_max = imputed[v].min(), imputed[v].max()
    # Anomaly: any imputed value outside [obs_min - 50%*range, obs_max + 50%*range]
    rng = obs_max - obs_min if pd.notna(obs_max) and pd.notna(obs_min) else 0
    threshold_lo = obs_min - 0.5 * rng if rng else obs_min
    threshold_hi = obs_max + 0.5 * rng if rng else obs_max
    n_below = (imputed[v] < threshold_lo).sum() if pd.notna(threshold_lo) else 0
    n_above = (imputed[v] > threshold_hi).sum() if pd.notna(threshold_hi) else 0
    anomaly_str = f'{n_below + n_above}' if (n_below + n_above) > 0 else ''
    print(f'{v[:50]:<50s} {obs_min:>10.2f} {obs_max:>10.2f} {imp_min:>10.2f} {imp_max:>10.2f} {anomaly_str:>10s}')

# ── 4. Year coverage per country ──
print()
print('=' * 70)
print('4. Year coverage audit')
print('=' * 70)

# Year range per country
yr_range = observed.groupby('Country Code')['Year'].agg(['min', 'max', 'count'])
yr_range['expected'] = yr_range['max'] - yr_range['min'] + 1
yr_range['gaps'] = yr_range['expected'] - yr_range['count']
gappy = yr_range[yr_range['gaps'] > 0].sort_values('gaps', ascending=False)
if len(gappy) > 0:
    print(f'Countries with year gaps (n={len(gappy)}):')
    print(gappy.head(20).to_string())
else:
    print('No year gaps. All countries are continuous.')
print()
print('Countries not covering 1995-2023 fully:')
incomplete = yr_range[(yr_range['min'] > 1995) | (yr_range['max'] < 2023)]
if len(incomplete) > 0:
    print(incomplete.to_string())
else:
    print('All countries cover full 1995-2023 range.')

# ── 5. Variable name issues ──
print()
print('=' * 70)
print('5. Variable name issues')
print('=' * 70)

# Look for special chars in column names that might cause problems
problematic_chars = []
for c in observed.columns:
    if any(ch in c for ch in ['  ', '\t', '\n', '\r', '\xa0']):
        problematic_chars.append(c)
    if c != c.strip():
        problematic_chars.append(c)
if problematic_chars:
    print('Columns with whitespace issues:')
    for c in problematic_chars:
        print(f'  {repr(c)}')
else:
    print('Column names: no whitespace issues.')

# Encoding check: look for em-dash, en-dash
em_dash_cols = [c for c in observed.columns if '—' in c or '–' in c or '‐' in c]
if em_dash_cols:
    print(f'Columns with dashes (UTF-8 special chars): {len(em_dash_cols)}')
    for c in em_dash_cols:
        print(f'  {c}')

# ── 6. Cross-imputation variance ──
print()
print('=' * 70)
print('6. Cross-imputation variance (which cells diverge most across M)')
print('=' * 70)

# For numeric columns, compute coefficient of variation across M imputations per cell
imp_numeric = [c for c in imputed.columns if c not in {'Country Code', 'Country Name', 'Year', 'imputation_id'}]
imp_numeric = [c for c in imp_numeric if imputed[c].dtype.kind in 'fiu']

# Group by (Country Code, Year), compute std across M imputations per column
cv_per_var = []
for v in key_vars[:6]:  # Just the top 6 for performance
    if v not in imputed.columns: continue
    grouped = imputed.groupby(['Country Code', 'Year'])[v].agg(['mean', 'std'])
    # Coefficient of variation where mean isn't ~0
    cv = (grouped['std'] / grouped['mean'].abs()).replace([np.inf, -np.inf], np.nan).dropna()
    if len(cv) == 0: continue
    cv_per_var.append({
        'variable': v,
        'cv_median': float(cv.median()),
        'cv_p95':    float(cv.quantile(0.95)),
        'cv_max':    float(cv.max()),
        'n_cells':   int(len(cv)),
    })
print('Coefficient of variation across M imputations (per cell):')
print(pd.DataFrame(cv_per_var).to_string(index=False))

# Cells where M=5 imputations diverge most (highest std)
print()
print('Top 10 most-divergent imputed cells (highest std across M, for ECI):')
eci_std = imputed.groupby(['Country Code', 'Year'])['Economic Complexity Index'].agg(['mean', 'std'])
eci_std = eci_std.sort_values('std', ascending=False).head(10)
print(eci_std.to_string())

# ── 7. Variables that should be observed but are imputed ──
print()
print('=' * 70)
print('7. Trade-side variables: should be observed everywhere')
print('=' * 70)

trade_vars = ['wide_resource_share', 'hydrocarbon_share', 'ores_share',
              'base_metals_share', 'precious_share']
for v in trade_vars:
    if v not in observed.columns: continue
    n_nan_obs = observed[v].isna().sum()
    n_nan_imp = imputed[v].isna().sum()
    print(f'  {v}: NaN in observed = {n_nan_obs}, NaN in imputed = {n_nan_imp}')

# ── 8. Population merge sanity ──
print()
print('=' * 70)
print('8. Population variable sanity')
print('=' * 70)

if 'Population' in observed.columns:
    pop_zero = (observed['Population'] == 0).sum()
    pop_nan = observed['Population'].isna().sum()
    print(f'  Population = 0: {pop_zero} rows')
    print(f'  Population NaN: {pop_nan} rows')
    print(f'  Top-5 smallest avg population countries:')
    print(observed.groupby('Country Code')['Population'].mean().sort_values().head(5).to_string())

print()
print('=' * 70)
print('Audit complete.')
print('=' * 70)
