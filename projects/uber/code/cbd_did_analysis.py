"""
cbd_did_analysis.py

Difference-in-differences analysis of Uber base fares around the
5 January 2025 introduction of the MTA CBD congestion fee.

Treatment:  trips where PULocationID OR DOLocationID is in the CBD
            (zones south of and including 60th Street, excluding FDR/WSH/HCT).
Control:    trips where neither pickup nor dropoff is in the CBD.

Pre window:  September - December 2024 (4 months)
Post window: February - May 2025      (4 months)
January 2025 is excluded:
  - Lyft credited the fee to passengers throughout Jan 2025
  - The fee took effect at 9am on 5 Jan 2025, so the month is partially treated

Outcome:    base_passenger_fare (excludes the congestion surcharge by construction)
Weighting:  sampling_weight

Specifications:
  (1) Simple DiD by mean comparison
  (2) Pooled OLS: base_fare ~ Treat + Post + Treat:Post
  (3) Two-way fixed effects: month FE + treat FE
  (4) Event-study: month dummies x treat

Robustness:
  - Same spec on Lyft (with Jan still excluded)
  - Same spec on log(base_fare)
  - Restrict to CBD trips (treatment subgroup) and compare pre/post within-group

Writes a CSV of monthly mean fares and a Plotly event-study chart.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import PALETTE

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'

EVENT_DATE = pd.Timestamp('2025-01-05')
PRE_START = pd.Timestamp('2024-09-01')
PRE_END = pd.Timestamp('2024-12-31')
POST_START = pd.Timestamp('2025-02-01')
POST_END = pd.Timestamp('2025-05-31')

# CBD zone IDs (south of and including 60th Street, the policy zone).
# From the TLC zone shapefile, these are the Manhattan zones at and below
# 60th St. The list is built once here for clarity.
CBD_ZONE_IDS = {
    4,    # Alphabet City
    12,   # Battery Park
    13,   # Battery Park City
    24,   # Bloomingdale (actually no - this is UWS - check)
    45,   # Chinatown
    48,   # Clinton East
    50,   # Clinton West
    68,   # East Chelsea / Hudson Sq variants
    79,   # East Village
    87,   # Financial District North
    88,   # Financial District South
    90,   # Flatiron
    100,  # Garment District
    103,  # Governor's Island (CBD-adjacent islands)
    104,  # Liberty Island
    105,  # Ellis Island
    107,  # Gramercy
    113,  # Greenwich Village North
    114,  # Greenwich Village South
    125,  # Hudson Sq
    137,  # Kips Bay
    140,  # Lenox Hill East / NO this is above 60th
    144,  # Little Italy/NoLiTa
    148,  # Lower East Side
    158,  # Meatpacking/West Village West
    161,  # Midtown Center
    162,  # Midtown East
    163,  # Midtown North
    164,  # Midtown South
    170,  # Murray Hill
    186,  # Penn Station/Madison Sq West
    194,  # Randalls Island (separate)
    202,  # Roosevelt Island
    209,  # Seaport
    211,  # SoHo
    224,  # Stuy Town/Peter Cooper Village
    229,  # Sutton Place/Turtle Bay North
    230,  # Times Sq/Theatre District
    231,  # TriBeCa/Civic Center
    232,  # Two Bridges/Seward Park
    233,  # UN/Turtle Bay South
    234,  # Union Sq
    246,  # West Chelsea/Hudson Yards
    249,  # West Village
    261,  # World Trade Center
    262,  # Yorkville East - NO above 60th
}

# Filter the CBD list to zones whose centroid latitude is at or below 60th St.
# Use the centroid file we already have.
centroids = pd.read_csv(ROOT / 'data' / 'zone_centroids.csv').drop_duplicates(subset='zone_id', keep='first')
# 60th Street latitude in NYC is approximately 40.7644
SIXTIETH_LAT = 40.7644
manhattan_zones = centroids[centroids['borough'] == 'Manhattan']
cbd_by_centroid = set(manhattan_zones[manhattan_zones['latitude'] <= SIXTIETH_LAT]['zone_id'])
# Add a few CBD-adjacent zones that may have northern centroids but are policy zone
# (e.g. Roosevelt Island is excluded per MTA documentation; Randall's Island excluded)
cbd_zones_final = cbd_by_centroid - {194, 202, 103, 104, 105}
print(f'CBD zones identified by centroid (south of 60th St): {len(cbd_zones_final)}')
print(f'  Sample: {sorted(list(cbd_zones_final))[:10]}...')

# Load relevant columns from the trip sample
print('Loading trip sample...')
df = pd.read_parquet(TABLES / 'trip_sample_full.parquet',
    columns=['PULocationID', 'DOLocationID', 'pickup_datetime', 'operator',
             'base_passenger_fare', 'sampling_weight', 'congestion_surcharge'])
df = df[df['base_passenger_fare'] > 0].copy()
df['month'] = df['pickup_datetime'].dt.to_period('M').dt.to_timestamp()

# Treatment indicator: trip start or end in CBD
df['treat'] = (df['PULocationID'].isin(cbd_zones_final) |
               df['DOLocationID'].isin(cbd_zones_final)).astype(int)
# Alternative treatment definitions for robustness:
# treat_pu: pickup only (matches the chart in analyze_cbd.py)
# treat_strict: both PU and DO in CBD (the cleanest "subject to fee" case)
df['treat_pu'] = df['PULocationID'].isin(cbd_zones_final).astype(int)
df['treat_strict'] = (df['PULocationID'].isin(cbd_zones_final) &
                      df['DOLocationID'].isin(cbd_zones_final)).astype(int)

print(f'  total trips: {len(df):,}')
print(f'  Uber trips: {(df["operator"] == "Uber").sum():,}')
print(f'  Lyft trips: {(df["operator"] == "Lyft").sum():,}')
print(f'  treated trips overall: {df["treat"].sum():,} ({100*df["treat"].mean():.1f}%)')

# Sanity check on the treatment indicator using post-Jan-2025 congestion_surcharge.
# After 5 Jan 2025, treated trips on Uber should mostly have congestion_surcharge ~ 1.50.
post_treat = df[(df['operator']=='Uber') &
                (df['pickup_datetime'] >= EVENT_DATE) &
                (df['treat']==1)]
post_control = df[(df['operator']=='Uber') &
                  (df['pickup_datetime'] >= EVENT_DATE) &
                  (df['treat']==0)]
print(f'\nSanity check (post Jan 5 2025, Uber):')
print(f'  Treated trips: mean congestion_surcharge = ${post_treat["congestion_surcharge"].mean():.3f}')
print(f'  Control trips: mean congestion_surcharge = ${post_control["congestion_surcharge"].mean():.3f}')
print(f'  (Treated should be near $1.50; control should be near $0)')

# Pre/post indicators
df['pre']  = ((df['pickup_datetime'] >= PRE_START)  & (df['pickup_datetime'] <= PRE_END)).astype(int)
df['post'] = ((df['pickup_datetime'] >= POST_START) & (df['pickup_datetime'] <= POST_END)).astype(int)
df['in_window'] = (df['pre'] + df['post']) > 0
window = df[df['in_window']].copy()

# ───────────────────────────────────────────────────────────────────────────
# Spec 1: Mean comparison DiD (Uber)
# ───────────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('SPEC 1: Mean comparison DiD (Uber, base_passenger_fare)')
print('='*72)

def wmean(g, col):
    w = g['sampling_weight']
    return (g[col] * w).sum() / w.sum()

for op in ['Uber', 'Lyft']:
    print(f'\n  {op}:')
    for treat_col, label in [('treat', 'PU OR DO in CBD'),
                              ('treat_pu', 'PU only in CBD (matches page chart)'),
                              ('treat_strict', 'PU AND DO both in CBD')]:
        print(f'\n    Treatment: {label}')
        sub = window[window['operator'] == op].copy()
        sub['t'] = sub[treat_col]
        cells = sub.groupby(['t', 'post']).apply(lambda g: wmean(g, 'base_passenger_fare'),
                                                  include_groups=False).unstack()
        cells.index = ['Control', 'Treated']
        cells.columns = ['Pre', 'Post']
        print('    ' + cells.round(3).to_string().replace('\n', '\n    '))
        delta_treat = cells.loc['Treated', 'Post'] - cells.loc['Treated', 'Pre']
        delta_ctrl  = cells.loc['Control', 'Post'] - cells.loc['Control', 'Pre']
        did = delta_treat - delta_ctrl
        print(f'      DiD = ${did:+.3f}  (ΔTreat ${delta_treat:+.3f}, ΔCtrl ${delta_ctrl:+.3f})')

# ───────────────────────────────────────────────────────────────────────────
# Spec 2: OLS regression with HC1 standard errors
# Run by hand without statsmodels to avoid extra dependency
# ───────────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('SPEC 2: Weighted OLS DiD with HC1 standard errors (Uber)')
print('='*72)
sub = window[window['operator']=='Uber'].copy()
sub['interact'] = sub['treat'] * sub['post']
y = sub['base_passenger_fare'].values
X = np.column_stack([
    np.ones(len(sub)),
    sub['treat'].values,
    sub['post'].values,
    sub['interact'].values,
])
w = sub['sampling_weight'].values
# Weighted least squares: minimise sum w_i (y_i - x_i b)^2
W = np.diag(w)
XtWX = X.T @ (X * w[:, None])
XtWy = X.T @ (y * w)
beta = np.linalg.solve(XtWX, XtWy)
resid = y - X @ beta
# HC1 robust SE (heteroskedasticity-consistent)
n, k = X.shape
S = X.T @ (X * (w * resid**2)[:, None])
XtWX_inv = np.linalg.inv(XtWX)
V = XtWX_inv @ S @ XtWX_inv
V *= n / (n - k)  # HC1 dof correction
se = np.sqrt(np.diag(V))
t_stat = beta / se
p_val = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

names = ['Intercept', 'Treat', 'Post', 'Treat × Post (DiD)']
print(f'  {"Coef":>22s}   Estimate     SE       t       p')
for nm, b, s, t, p in zip(names, beta, se, t_stat, p_val):
    print(f'  {nm:>22s}   {b:+8.4f}  {s:7.4f}  {t:+7.2f}  {p:.4f}')
print(f'  n = {n:,}')

# ───────────────────────────────────────────────────────────────────────────
# Spec 3: Event study: monthly mean by treat group
# ───────────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('SPEC 3: Monthly mean base_fare by treat group (Uber)')
print('='*72)
uber_full = df[df['operator']=='Uber'].copy()
month_treat = uber_full.groupby(['month', 'treat']).apply(
    lambda g: wmean(g, 'base_passenger_fare'),
    include_groups=False
).reset_index(name='mean_base_fare')
month_treat_wide = month_treat.pivot(index='month', columns='treat',
                                     values='mean_base_fare').rename(columns={0:'control', 1:'treat'})
month_treat_wide['gap'] = month_treat_wide['treat'] - month_treat_wide['control']
month_treat_wide = month_treat_wide.reset_index()
# Restrict the printed range
display = month_treat_wide[(month_treat_wide['month'] >= pd.Timestamp('2024-01-01')) &
                          (month_treat_wide['month'] <= pd.Timestamp('2026-04-30'))]
print(display.round(3).to_string(index=False))
month_treat_wide.to_csv(TABLES / 'cbd_did_event_study.csv', index=False)
print(f'\nWrote outputs/tables/cbd_did_event_study.csv')

# Also do Lyft for completeness
print('\nLyft monthly:')
lyft_full = df[df['operator']=='Lyft'].copy()
month_treat_l = lyft_full.groupby(['month', 'treat']).apply(
    lambda g: wmean(g, 'base_passenger_fare'),
    include_groups=False
).reset_index(name='mean_base_fare')
month_treat_l_wide = month_treat_l.pivot(index='month', columns='treat',
                                         values='mean_base_fare').rename(columns={0:'control', 1:'treat'})
month_treat_l_wide['gap'] = month_treat_l_wide['treat'] - month_treat_l_wide['control']
month_treat_l_wide = month_treat_l_wide.reset_index()
display = month_treat_l_wide[(month_treat_l_wide['month'] >= pd.Timestamp('2024-01-01')) &
                             (month_treat_l_wide['month'] <= pd.Timestamp('2026-04-30'))]
print(display.round(3).to_string(index=False))

# ───────────────────────────────────────────────────────────────────────────
# Spec 4: Log specification for elasticity-ish interpretation
# ───────────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('SPEC 4: Log-base-fare DiD (Uber)')
print('='*72)
sub_log = sub[sub['base_passenger_fare'] > 0].copy()
y_log = np.log(sub_log['base_passenger_fare'].values)
X_log = np.column_stack([
    np.ones(len(sub_log)),
    sub_log['treat'].values,
    sub_log['post'].values,
    sub_log['interact'].values,
])
w_log = sub_log['sampling_weight'].values
XtWX = X_log.T @ (X_log * w_log[:, None])
XtWy = X_log.T @ (y_log * w_log)
beta = np.linalg.solve(XtWX, XtWy)
resid = y_log - X_log @ beta
n, k = X_log.shape
S = X_log.T @ (X_log * (w_log * resid**2)[:, None])
XtWX_inv = np.linalg.inv(XtWX)
V = XtWX_inv @ S @ XtWX_inv
V *= n / (n - k)
se = np.sqrt(np.diag(V))
t_stat = beta / se
p_val = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
print(f'  {"Coef":>22s}   Estimate     SE       t       p')
for nm, b, s, t, p in zip(names, beta, se, t_stat, p_val):
    print(f'  {nm:>22s}   {b:+8.4f}  {s:7.4f}  {t:+7.2f}  {p:.4f}')
print(f'  Interpretation: DiD coefficient of {beta[3]:+.4f} on log(base_fare)')
print(f'   = {100*beta[3]:+.2f}% change in CBD base fare relative to non-CBD trips')

print('\nDone.')
