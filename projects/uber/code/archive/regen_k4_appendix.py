"""
Regenerate op_pool_by_zone.html and op_tip_geography.html at k=4 clusters.
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

print('Loading sample...')
df = pd.read_parquet(TABLES / 'trip_sample_full.parquet',
    columns=['PULocationID','pickup_datetime','operator',
             'base_passenger_fare','driver_pay','sampling_weight',
             'tips','shared_request_flag','shared_match_flag'])
df = df[df['base_passenger_fare'] > 0].copy()
df['cluster'] = df['PULocationID'].map(z2c)
df = df.dropna(subset=['cluster']).copy()
df['cluster'] = df['cluster'].astype(int)
print(f'Sample after cluster join: {len(df):,} rows')

# =============================================================
# CHART: op_pool_by_zone — Uber Pool request rate + match rate by cluster (2024-26)
# =============================================================
print('\n[1/2] Pool by cluster (Uber 2024-26)...')
u = df[(df['operator']=='Uber') &
       (df['pickup_datetime'] >= '2024-01-01')].copy()
u['req'] = (u['shared_request_flag'] == 'Y').astype(int)
u['mat'] = (u['shared_match_flag'] == 'Y').astype(int)

def pool_metrics(x):
    w = x['sampling_weight']
    total_w = w.sum()
    req_rate = (x['req'] * w).sum() / total_w
    requested = x[x['req']==1]
    if len(requested):
        wr = requested['sampling_weight']
        match_rate = (requested['mat'] * wr).sum() / wr.sum()
    else:
        match_rate = None
    return pd.Series({'request_rate': req_rate, 'match_rate': match_rate})

pool = u.groupby('cluster').apply(pool_metrics, include_groups=False).reset_index()
pool = pool.set_index('cluster').reindex(range(4)).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=LABELS, y=pool['request_rate'], name='Pool request rate',
    marker_color=PALETTE['steel'],
    text=[f'{v:.1%}' if pd.notna(v) else '' for v in pool['request_rate']],
    textposition='outside', textfont=dict(size=10),
    hovertemplate='Request rate: %{x}<br>%{y:.1%}<extra></extra>',
))
fig.add_trace(go.Bar(
    x=LABELS, y=pool['match_rate'], name='Match rate (given request)',
    marker_color=PALETTE['navy'],
    text=[f'{v:.1%}' if pd.notna(v) else '' for v in pool['match_rate']],
    textposition='outside', textfont=dict(size=10),
    hovertemplate='Match rate: %{x}<br>%{y:.1%}<extra></extra>',
))
fig.update_layout(
    barmode='group', xaxis_title='',
    yaxis_title='Share of Uber trips',
    yaxis=dict(tickformat='.0%'),
    height=420, legend=LEGEND,
    margin=dict(t=60, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'sample'/'op_pool_by_zone.html')
print('  wrote op_pool_by_zone.html')

# =============================================================
# CHART: op_tip_geography — Average tip and zero-tip rate by cluster x operator
# =============================================================
print('\n[2/2] Tip geography by cluster...')
t = df.copy()
t['tips'] = t['tips'].fillna(0)
t['zero_tip'] = (t['tips'] == 0).astype(int)

def tip_metrics(x):
    w = x['sampling_weight']
    avg_tip = (x['tips'] * w).sum() / w.sum()
    zero_rate = (x['zero_tip'] * w).sum() / w.sum()
    return pd.Series({'avg_tip': avg_tip, 'zero_rate': zero_rate})

tip = t.groupby(['operator','cluster']).apply(tip_metrics, include_groups=False).reset_index()

# Build chart: 2 panels (avg tip + zero rate), 2 operators x 4 clusters bars each
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Average tip per trip ($)',
                                    'Share of trips with zero tip'),
                    horizontal_spacing=0.12)

for op in ['Uber','Lyft']:
    s = tip[tip['operator']==op].set_index('cluster').reindex(range(4)).reset_index()
    fig.add_trace(go.Bar(
        x=LABELS, y=s['avg_tip'], name=op,
        marker_color=OP_COLORS[op],
        text=[f'${v:.2f}' for v in s['avg_tip']],
        textposition='outside', textfont=dict(size=9),
        hovertemplate=op+' %{x}<br>Avg tip: $%{y:.2f}<extra></extra>',
        legendgroup=op, showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=LABELS, y=s['zero_rate'], name=op,
        marker_color=OP_COLORS[op],
        text=[f'{v:.0%}' for v in s['zero_rate']],
        textposition='outside', textfont=dict(size=9),
        hovertemplate=op+' %{x}<br>Zero-tip rate: %{y:.1%}<extra></extra>',
        legendgroup=op, showlegend=False,
    ), row=1, col=2)

fig.update_yaxes(title_text='$', row=1, col=1)
fig.update_yaxes(tickformat='.0%', row=1, col=2)
fig.update_layout(
    barmode='group',
    height=420,
    legend=LEGEND,
    margin=dict(t=80, b=40, l=60, r=20),
)
save_chart(fig, ROOT/'outputs'/'sample'/'op_tip_geography.html')
print('  wrote op_tip_geography.html')

print('\nDone.')
