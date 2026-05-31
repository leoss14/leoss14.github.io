"""
rev_weighted_and_verify.py

(1) Regenerate op_margin_monthly.html and op_margin_by_zone.html using
    revenue-weighted margins (1 - sum w*pay / sum w*fare) instead of the
    trip-weighted definition currently in sample_charts.py. This makes the
    Part 3 charts consistent with each other and with industry definitions
    of platform take rate.

(2) Verify the substantive claims in Part 3 viz-notes against the
    regenerated chart data:
      - pay_per_hour: trip-only vs effective by cluster
      - response_time: dispatch waits by operator x cluster
      - pool_by_zone: Pool request and match rates by cluster
      - tip_geography: average tip and zero-tip rate by cluster

(3) Recompute the Part 1 time-trend claim ("Manhattan share declined,
    outer clusters gained") under the new C clustering, returning the
    monthly per-cluster share series so the prose can be updated with
    correct magnitudes.

Each section reports plain numbers to stdout. The page can be updated
manually from those numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from trip_queries import save_chart, PALETTE

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
DATA = ROOT / 'data'
CHART_DIR = ROOT / 'outputs' / 'sample'

OP_COLORS = {'Uber': PALETTE['navy'], 'Lyft': PALETTE['rose']}
CLUSTER_COLORS = [PALETTE['navy'], PALETTE['rose'], PALETTE['gold'], PALETTE['sage']]
LBL = ['Manhattan', 'Brooklyn', 'Bronx + Upper Manhattan', 'Queens']

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print('Loading canonical clusters and sample...')
clusters = pd.read_csv(TABLES / 'zone_clusters_canonical.csv')
z2c = dict(zip(clusters['zone_id'], clusters['cluster']))

df = pd.read_parquet(TABLES / 'trip_sample_full.parquet', columns=[
    'PULocationID','pickup_datetime','operator',
    'base_passenger_fare','driver_pay','sampling_weight',
    'tips','response_sec','trip_time','effective_pay_per_hour' if False else 'pay_per_hour',
    'is_shared_requested','is_shared_matched',
])
df = df[df['base_passenger_fare'] > 0].copy()
df['cluster'] = df['PULocationID'].map(z2c)
df = df.dropna(subset=['cluster']).copy()
df['cluster'] = df['cluster'].astype(int)
df['month'] = df['pickup_datetime'].dt.to_period('M').dt.to_timestamp()
df['year'] = df['pickup_datetime'].dt.year
print(f'  {len(df):,} trips after cluster join')


def rw_margin(g):
    """Revenue-weighted margin = 1 - sum(w*pay) / sum(w*fare)."""
    return 1 - (g['driver_pay']*g['sampling_weight']).sum() / (g['base_passenger_fare']*g['sampling_weight']).sum()


# ---------------------------------------------------------------------------
# (1a) Regenerate op_margin_monthly with revenue-weighted margins
# ---------------------------------------------------------------------------
print('\n[1a] Regenerating op_margin_monthly with revenue-weighted definition...')

monthly = df.groupby(['operator','month']).apply(rw_margin, include_groups=False).reset_index(name='margin_rw')

fig1 = go.Figure()
for op in ['Uber','Lyft']:
    d = monthly[monthly['operator']==op].sort_values('month')
    fig1.add_trace(go.Scatter(
        x=d['month'], y=d['margin_rw']*100,
        mode='lines', name=op,
        line=dict(color=OP_COLORS[op], width=2.2),
        hovertemplate='%{x|%b %Y}<br>'+op+': %{y:.1f}%<extra></extra>',
    ))
fig1.add_hline(y=0, line_color='#aab0bd', line_width=1, line_dash='dot')
fig1.update_layout(
    xaxis_title='Month',
    yaxis_title='Operator margin (revenue-weighted, %)',
    yaxis_ticksuffix='%',
    legend=dict(yanchor='bottom', y=0.05, xanchor='right', x=0.98),
    height=420,
)
save_chart(fig1, CHART_DIR / 'op_margin_monthly.html')
print('  wrote op_margin_monthly.html')

# Headline numbers: full-panel weighted means
for op in ['Uber','Lyft']:
    sub = df[df['operator']==op]
    m = rw_margin(sub)
    print(f'    {op} full-panel revenue-weighted: {m*100:.1f}%')

# ---------------------------------------------------------------------------
# (1b) Regenerate op_margin_by_zone with revenue-weighted margins
# ---------------------------------------------------------------------------
print('\n[1b] Regenerating op_margin_by_zone with revenue-weighted definition...')

by_zone = df.groupby(['operator','cluster']).apply(rw_margin, include_groups=False).reset_index(name='margin_rw')

fig2 = go.Figure()
for op in ['Uber','Lyft']:
    d = by_zone[by_zone['operator']==op].sort_values('cluster')
    fig2.add_trace(go.Bar(
        x=[LBL[c] for c in d['cluster']],
        y=d['margin_rw']*100,
        name=op,
        marker_color=OP_COLORS[op],
        hovertemplate='%{x}<br>'+op+': %{y:.1f}%<extra></extra>',
    ))
fig2.update_layout(
    barmode='group',
    xaxis_title='Pickup cluster',
    yaxis_title='Operator margin (revenue-weighted, %)',
    yaxis_ticksuffix='%',
    height=420,
    legend=dict(yanchor='top', y=0.98, xanchor='right', x=0.98),
)
save_chart(fig2, CHART_DIR / 'op_margin_by_zone.html')
print('  wrote op_margin_by_zone.html')
print('  cluster x operator (revenue-weighted, %):')
piv = by_zone.pivot(index='cluster', columns='operator', values='margin_rw') * 100
piv.index = [LBL[i] for i in piv.index]
piv['gap_Lyft_minus_Uber'] = piv['Lyft'] - piv['Uber']
print(piv.round(2).to_string())

# ---------------------------------------------------------------------------
# (2a) pay_per_hour by cluster (Uber only, 2024-26)
# ---------------------------------------------------------------------------
print('\n[2a] pay_per_hour by cluster (Uber, 2024-26):')
u_recent = df[(df['operator']=='Uber') & (df['year']>=2024)].copy()
resp = u_recent['response_sec'].clip(lower=0, upper=1800).fillna(0)
u_recent['effective_hours'] = (u_recent['trip_time'] + resp) / 3600.0
u_recent['effective_pay_per_hour'] = u_recent['driver_pay'] / u_recent['effective_hours'].replace(0, np.nan)

def wmean(g, col):
    s = g.dropna(subset=[col])
    return (s[col]*s['sampling_weight']).sum() / s['sampling_weight'].sum() if len(s) else float('nan')

print(f'  {"cluster":24s}  trip-only $  effective $  gap %')
for c in range(4):
    g = u_recent[u_recent['cluster']==c]
    trip_pph = wmean(g, 'pay_per_hour')
    eff_pph = wmean(g, 'effective_pay_per_hour')
    gap = 100 * (trip_pph - eff_pph) / trip_pph
    print(f'  {LBL[c]:24s}  ${trip_pph:6.2f}      ${eff_pph:6.2f}      {gap:5.1f}%')

# ---------------------------------------------------------------------------
# (2b) response_time by operator x cluster (2024-26)
# ---------------------------------------------------------------------------
print('\n[2b] response_time (dispatch wait, sec) by operator x cluster (2024-26):')
recent = df[df['year']>=2024].copy()
recent['response_sec'] = recent['response_sec'].clip(lower=0, upper=1800)
recent = recent.dropna(subset=['response_sec'])

resp_table = recent.groupby(['operator','cluster']).apply(
    lambda g: wmean(g, 'response_sec'), include_groups=False).reset_index(name='resp_mean')
piv = resp_table.pivot(index='cluster', columns='operator', values='resp_mean')
piv.index = [LBL[i] for i in piv.index]
piv['gap_seconds'] = piv['Lyft'] - piv['Uber']
print(piv.round(1).to_string())

# ---------------------------------------------------------------------------
# (2c) pool by cluster (Uber, 2024-26)
# ---------------------------------------------------------------------------
print('\n[2c] Pool requested and matched rates by cluster (Uber, 2024-26):')
u_recent_pool = df[(df['operator']=='Uber') & (df['year']>=2024)].copy()
u_recent_pool['is_shared_requested'] = u_recent_pool['is_shared_requested'].fillna(0)
u_recent_pool['is_shared_matched'] = u_recent_pool['is_shared_matched'].fillna(0)

for c in range(4):
    g = u_recent_pool[u_recent_pool['cluster']==c]
    req = wmean(g, 'is_shared_requested') * 100
    matched = wmean(g, 'is_shared_matched') * 100
    match_rate = matched / req * 100 if req > 0 else float('nan')
    print(f'  {LBL[c]:24s}  request %: {req:5.2f}%   match %: {matched:5.2f}%   match-rate-of-requests: {match_rate:5.1f}%')

# ---------------------------------------------------------------------------
# (2d) tip by operator x cluster
# ---------------------------------------------------------------------------
print('\n[2d] Tips by operator x cluster (full panel):')
df['tip_dollars'] = df['tips']
df['zero_tip'] = (df['tips'] == 0).astype(float)
tip_table = df.groupby(['operator','cluster']).apply(
    lambda g: pd.Series({
        'avg_tip_$': wmean(g, 'tip_dollars'),
        'zero_tip_pct': wmean(g, 'zero_tip') * 100,
    }), include_groups=False).reset_index()
print(tip_table.assign(cluster_name=lambda d: d['cluster'].map(lambda c: LBL[c])).round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# (3) Part 1 time-trend: per-cluster share of Uber pickups, by year
# ---------------------------------------------------------------------------
print('\n[3] Per-cluster share of Uber pickups by year (the Part 1 time-trend claim):')
u_all = df[df['operator']=='Uber'].copy()
u_all['n_w'] = u_all['sampling_weight']
yearly = u_all.groupby(['year','cluster'])['n_w'].sum().reset_index()
yearly_total = u_all.groupby('year')['n_w'].sum().reset_index().rename(columns={'n_w':'total'})
yearly = yearly.merge(yearly_total, on='year')
yearly['share'] = yearly['n_w'] / yearly['total']
piv = yearly.pivot(index='year', columns='cluster', values='share') * 100
piv.columns = [LBL[i] for i in piv.columns]
print(piv.round(2).to_string())

print('\nChanges from 2019 to 2026 (percentage-point shift):')
for c_name in piv.columns:
    series = piv[c_name].dropna()
    if len(series) >= 2:
        change = series.iloc[-1] - series.iloc[0]
        print(f'  {c_name:24s}  {series.iloc[0]:5.2f}%  ->  {series.iloc[-1]:5.2f}%   ({change:+.2f} pp)')

print('\nDone.')
