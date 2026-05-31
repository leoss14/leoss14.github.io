"""
cbd_overlay_charts.py

Produces the dual-line (pickup-only vs either-end) overlay variant of the
CBD share chart. The two definitions differ by ~8 percentage points on
this metric (relative gap ~33%), large enough that showing both is
informative.

Overlays were also explored for the volume, buffer-share, and pass-through
charts. They were dropped:
  • volume_by_zone_class: stacked-area decomposition is pickup-class based
    by construction; an either-end overlay confuses the visual.
  • buffer_share: gap is real but the chart is a tidy single-line metric
    and a dual-line version adds clutter without changing the story.
  • passthrough: PU-only and either-end fare/pay lines sit within $1 of
    each other, i.e. nearly parallel; the second pair adds visual noise
    without analytical content.

Original four charts (analyze_cbd.py) are unchanged. This script only
writes outputs/cbd/cbd_share_inside_both.html.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
from _panel_loader import classify_zones, base_layout, save_chart, PALETTE

EVENT_DATE = pd.Timestamp('2025-01-05')
OP_COLOR = {'Uber': PALETTE['navy'], 'Lyft': PALETTE['rose']}

zones = classify_zones()
cbd_zones = set(zones.loc[zones['zone_class'] == 'cbd', 'zone_id'].astype(int))
print(f'CBD zones: {len(cbd_zones)}')

print('Loading trip sample...')
df = pd.read_parquet(
    ROOT / 'outputs' / 'tables' / 'trip_sample_full.parquet',
    columns=['operator', 'pickup_datetime', 'PULocationID',
             'DOLocationID', 'sampling_weight'])
df['month'] = df['pickup_datetime'].dt.to_period('M').dt.to_timestamp()
df['w'] = df['sampling_weight'].astype(float)
df['pu_cbd'] = df['PULocationID'].isin(cbd_zones)
df['either_cbd'] = df['pu_cbd'] | df['DOLocationID'].isin(cbd_zones)
print(f'  {len(df):,} trips')

fig = go.Figure()
for op in ['Uber', 'Lyft']:
    sub = df[df['operator'] == op]
    m = sub.groupby('month').apply(lambda x: pd.Series({
        't': x['w'].sum(),
        'e': x.loc[x['either_cbd'], 'w'].sum(),
    })).reset_index()
    color = OP_COLOR[op]
    fig.add_trace(go.Scatter(
        x=m['month'], y=m['e'] / m['t'] * 100, mode='lines',
        name=f'{op}',
        line=dict(color=color, width=2.4),
        hovertemplate=f'{op}<br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>'))

fig.add_vline(x=EVENT_DATE,
              line=dict(color=PALETTE['text'], dash='dash', width=1))
fig.add_annotation(x=EVENT_DATE, y=1.02, yref='paper',
                   text='CBD fee starts', showarrow=False,
                   font=dict(size=11), xanchor='left', yanchor='bottom')
fig.update_layout(**base_layout(height=440))
fig.update_yaxes(title='Share of monthly trips (%)')
fig.update_xaxes(title=None)
save_chart(fig, 'cbd/cbd_share_inside_both')
print('Wrote outputs/cbd/cbd_share_inside_both.html')
