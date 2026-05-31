"""
regen_k4_charts.py

Regenerate Part 2 + Part 3 charts using canonical k=4 clusters.
Run from project root or anywhere; uses absolute paths.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber/code')
from trip_queries import save_chart, PALETTE

import pandas as pd
import numpy as np
import plotly.graph_objects as go

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'

LABELS = ['Manhattan', 'Brooklyn', 'Bronx + Upper Manhattan', 'Queens']
COLORS = [PALETTE['navy'], PALETTE['rose'], PALETTE['gold'], PALETTE['sage']]
OP_COLORS = {'Uber': PALETTE['navy'], 'Lyft': PALETTE['rose']}
LEGEND = dict(orientation='h', yanchor='bottom', y=1.0,
              xanchor='right', x=1.0, font=dict(size=11),
              bgcolor='rgba(255,255,255,0.85)',
              bordercolor='rgba(0,0,0,0)')

clusters = pd.read_csv(TABLES / 'zone_clusters_canonical.csv')
z2c = dict(zip(clusters['zone_id'], clusters['cluster']))
print(f'Loaded {len(clusters)} zone->cluster mappings')

print('Loading sample (this takes ~30s)...')
df = pd.read_parquet(TABLES / 'trip_sample_full.parquet',
    columns=['PULocationID','pickup_datetime','operator',
             'base_passenger_fare','driver_pay','sampling_weight',
             'tips','trip_time','trip_miles','on_scene_datetime',
             'request_datetime','shared_request_flag','shared_match_flag'])
df = df[df['base_passenger_fare'] > 0].copy()
df['cluster'] = df['PULocationID'].map(z2c)
df = df.dropna(subset=['cluster']).copy()
df['cluster'] = df['cluster'].astype(int)
df['month'] = df['pickup_datetime'].dt.to_period('M').dt.to_timestamp()
df['year']  = df['pickup_datetime'].dt.year
df['margin'] = (df['base_passenger_fare'] - df['driver_pay']) / df['base_passenger_fare']
print(f'Sample after cluster join: {len(df):,} rows')

# ============================================================
# CHART 1: Part 2 - margin by cluster monthly (Uber)
# ============================================================
print('\n[1/5] Margin by cluster monthly (Uber)...')
u = df[df['operator']=='Uber']
g = u.groupby(['month','cluster']).apply(
    lambda x: pd.Series({
        'rev_w': 1 - (x['driver_pay']*x['sampling_weight']).sum() /
                     (x['base_passenger_fare']*x['sampling_weight']).sum(),
    }), include_groups=False
).reset_index()
g = g.sort_values(['cluster','month'])
g['smooth'] = g.groupby('cluster')['rev_w'].transform(
    lambda s: s.rolling(3, center=True, min_periods=1).mean())

fig = go.Figure()
for c in range(4):
    sub = g[g['cluster']==c]
    fig.add_trace(go.Scatter(
        x=sub['month'], y=sub['smooth'], mode='lines',
        line=dict(color=COLORS[c], width=2.2),
        name=LABELS[c],
        hovertemplate=LABELS[c]+'<br>%{x|%b %Y}<br>%{y:.1%}<extra></extra>',
    ))
fig.add_annotation(x='2020-08-01', y=0.21,
    text='Convergence (Aug 2020)<br>cross-cluster gap narrows',
    showarrow=True, arrowhead=2, arrowsize=0.9, ax=-30, ay=-50,
    font=dict(size=10, color=PALETTE['ink']),
    bgcolor='rgba(255,255,255,0.85)',
    bordercolor=PALETTE['grey'], borderwidth=1)
fig.add_annotation(x='2023-03-01', y=0.27,
    text='Divergence (early 2023)<br>Manhattan pulls ahead',
    showarrow=True, arrowhead=2, arrowsize=0.9, ax=20, ay=-40,
    font=dict(size=10, color=PALETTE['ink']),
    bgcolor='rgba(255,255,255,0.85)',
    bordercolor=PALETTE['grey'], borderwidth=1)
fig.update_layout(
    xaxis_title='', yaxis_title='Revenue-weighted operator margin',
    yaxis=dict(tickformat='.0%'),
    height=440, legend=LEGEND, margin=dict(t=60, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'margin'/'trip_margin_by_zone_class_p50.html')
print('  wrote trip_margin_by_zone_class_p50.html')

# ============================================================
# CHART 2: Part 3 - op_margin_by_zone (Uber vs Lyft margin by cluster)
# ============================================================
print('\n[2/5] Uber vs Lyft margin by cluster...')
both = df.groupby(['operator','cluster']).apply(
    lambda x: pd.Series({
        'rev_w': 1 - (x['driver_pay']*x['sampling_weight']).sum() /
                     (x['base_passenger_fare']*x['sampling_weight']).sum(),
    }), include_groups=False).reset_index()

fig = go.Figure()
for op in ['Uber','Lyft']:
    s = both[both['operator']==op]
    vals = []
    for c in range(4):
        row = s[s['cluster']==c]
        vals.append(row['rev_w'].iloc[0] if len(row) else None)
    fig.add_trace(go.Bar(
        x=LABELS, y=vals, name=op,
        marker_color=OP_COLORS[op],
        hovertemplate=op+': %{x}<br>Margin: %{y:.1%}<extra></extra>',
        text=[f'{v:.1%}' for v in vals],
        textposition='outside', textfont=dict(size=10),
    ))
fig.update_layout(
    barmode='group', xaxis_title='', yaxis_title='Revenue-weighted operator margin',
    yaxis=dict(tickformat='.0%', range=[0, 0.35]),
    height=420, legend=LEGEND, margin=dict(t=60, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'sample'/'op_margin_by_zone.html')
print('  wrote op_margin_by_zone.html')

# ============================================================
# CHART 3: Part 3 - subsidised share by year x cluster (Uber)
# ============================================================
print('\n[3/5] Subsidised share by year x cluster (Uber)...')
u2 = df[df['operator']=='Uber'].copy()
u2['neg'] = (u2['margin'] < 0).astype(int)
ss = u2.groupby(['year','cluster']).apply(
    lambda x: (x['neg']*x['sampling_weight']).sum() / x['sampling_weight'].sum(),
    include_groups=False).reset_index(name='neg_share')

fig = go.Figure()
for c in range(4):
    s = ss[ss['cluster']==c].sort_values('year')
    fig.add_trace(go.Scatter(
        x=s['year'], y=s['neg_share'], mode='lines+markers',
        line=dict(color=COLORS[c], width=2.2),
        marker=dict(size=8),
        name=LABELS[c],
        hovertemplate=LABELS[c]+'<br>%{x}: %{y:.1%}<extra></extra>',
    ))
fig.update_layout(
    xaxis_title='', yaxis_title='Share of Uber trips with negative margin',
    yaxis=dict(tickformat='.0%'),
    height=440, legend=LEGEND, margin=dict(t=60, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'sample'/'op_subsidised_share.html')
print('  wrote op_subsidised_share.html')

# ============================================================
# CHART 4: Part 3 - pay per hour (Uber, 2024-2026, by cluster)
# ============================================================
print('\n[4/5] Pay per hour by cluster (Uber 2024-26)...')
r = df[(df['pickup_datetime']>='2024-01-01') & (df['trip_time']>0)].copy()
r['dispatch_s'] = (pd.to_datetime(r['on_scene_datetime']) -
                   pd.to_datetime(r['request_datetime'])).dt.total_seconds()
r['dispatch_s'] = r['dispatch_s'].fillna(0).clip(lower=0, upper=1800)
r['eff_time'] = r['trip_time'] + r['dispatch_s']

def metrics(x):
    w = x['sampling_weight']
    pay = (x['driver_pay']*w).sum()
    th  = (x['trip_time']*w).sum() / 3600
    eh  = (x['eff_time']*w).sum() / 3600
    return pd.Series({
        'pph_trip': pay/th if th>0 else None,
        'pph_eff':  pay/eh if eh>0 else None,
    })

pph = r.groupby(['operator','cluster']).apply(metrics, include_groups=False).reset_index()
upph = pph[pph['operator']=='Uber'].set_index('cluster').reindex(range(4)).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=LABELS, y=upph['pph_trip'], name='Trip-only',
    marker_color=PALETTE['steel'],
    text=[f'${v:.0f}' if pd.notna(v) else '' for v in upph['pph_trip']],
    textposition='outside', textfont=dict(size=10),
    hovertemplate='Trip-only: %{x}<br>$%{y:.2f}/hr<extra></extra>',
))
fig.add_trace(go.Bar(
    x=LABELS, y=upph['pph_eff'], name='Effective (incl. dispatch)',
    marker_color=PALETTE['navy'],
    text=[f'${v:.0f}' if pd.notna(v) else '' for v in upph['pph_eff']],
    textposition='outside', textfont=dict(size=10),
    hovertemplate='Effective: %{x}<br>$%{y:.2f}/hr<extra></extra>',
))
fig.update_layout(
    barmode='group', xaxis_title='', yaxis_title='Uber driver pay ($/hour)',
    height=420, legend=LEGEND, margin=dict(t=60, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'sample'/'op_pay_per_hour.html')
print('  wrote op_pay_per_hour.html')

# ============================================================
# CHART 5: Part 3 - response time by operator x cluster (2024-2026)
# ============================================================
print('\n[5/5] Response time by cluster (2024-26)...')
rt = df[(df['pickup_datetime']>='2024-01-01') &
        df['on_scene_datetime'].notna() &
        df['request_datetime'].notna()].copy()
rt['dispatch_min'] = ((pd.to_datetime(rt['on_scene_datetime']) -
                       pd.to_datetime(rt['request_datetime'])).dt.total_seconds() / 60)
rt = rt[(rt['dispatch_min']>0) & (rt['dispatch_min']<30)]

agg = rt.groupby(['operator','cluster'])['dispatch_min'].agg([
    ('p10', lambda x: x.quantile(0.10)),
    ('p50', lambda x: x.median()),
    ('p90', lambda x: x.quantile(0.90)),
]).reset_index()

fig = go.Figure()
for op in ['Uber','Lyft']:
    s = agg[agg['operator']==op].set_index('cluster').reindex(range(4)).reset_index()
    fig.add_trace(go.Bar(
        x=LABELS, y=s['p50'], name=op+' median',
        marker_color=OP_COLORS[op],
        error_y=dict(
            type='data', symmetric=False,
            array=s['p90']-s['p50'], arrayminus=s['p50']-s['p10'],
            color=PALETTE['ink'], thickness=1,
        ),
        text=[f'{v:.1f} min' if pd.notna(v) else '' for v in s['p50']],
        textposition='outside', textfont=dict(size=10),
        hovertemplate=op+': %{x}<br>Median: %{y:.1f} min<extra></extra>',
    ))
fig.update_layout(
    barmode='group', xaxis_title='', yaxis_title='Dispatch time (minutes)',
    height=440, legend=LEGEND, margin=dict(t=60, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'sample'/'op_response_time.html')
print('  wrote op_response_time.html')
print('\nAll done.')
