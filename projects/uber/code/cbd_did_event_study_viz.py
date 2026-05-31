"""
cbd_did_event_study_viz.py

Event-study chart for the CBD congestion fee DiD.

Plots monthly mean base_passenger_fare for Uber and Lyft separately,
split by treatment status (trip has PU or DO in CBD vs neither).
Vertical line marks the fee event (5 Jan 2025) plus shaded pre/post
windows used in the DiD.

Reads:  outputs/tables/cbd_did_event_study.csv (Uber treated/control monthly)
        also computes Lyft series in-script.
Writes: outputs/cbd/cbd_did_event_study.html
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import save_chart, PALETTE

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
DATA = ROOT / 'data'
OUT_DIR = ROOT / 'outputs' / 'cbd'

EVENT_DATE = pd.Timestamp('2025-01-05')
PRE_START = pd.Timestamp('2024-09-01')
PRE_END = pd.Timestamp('2024-12-31')
POST_START = pd.Timestamp('2025-02-01')
POST_END = pd.Timestamp('2025-05-31')

centroids = pd.read_csv(DATA / 'zone_centroids.csv').drop_duplicates(subset='zone_id', keep='first')
SIXTIETH_LAT = 40.7644
manhattan_zones = centroids[centroids['borough'] == 'Manhattan']
cbd_zones = set(manhattan_zones[manhattan_zones['latitude'] <= SIXTIETH_LAT]['zone_id']) - {194, 202, 103, 104, 105}

print(f'CBD zones: {len(cbd_zones)}')
print('Loading trip sample...')
df = pd.read_parquet(TABLES / 'trip_sample_full.parquet',
    columns=['PULocationID','DOLocationID','pickup_datetime','operator',
             'base_passenger_fare','sampling_weight'])
df = df[df['base_passenger_fare'] > 0].copy()
df['month'] = df['pickup_datetime'].dt.to_period('M').dt.to_timestamp()
df['treat'] = (df['PULocationID'].isin(cbd_zones) | df['DOLocationID'].isin(cbd_zones)).astype(int)

def wmean(g, col):
    w = g['sampling_weight']
    return (g[col] * w).sum() / w.sum()

# Monthly means by operator x treatment
print('Computing monthly means...')
monthly = df.groupby(['operator','month','treat']).apply(
    lambda g: wmean(g, 'base_passenger_fare'),
    include_groups=False,
).reset_index(name='mean_base_fare')

# Restrict to 2024-01 onward for clarity
monthly = monthly[monthly['month'] >= pd.Timestamp('2024-01-01')].copy()

print('Building two-panel figure...')
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Uber', 'Lyft'),
    shared_yaxes=True,
    horizontal_spacing=0.07,
)

OP_POS = {'Uber': 1, 'Lyft': 2}
TREAT_COLOR = {0: PALETTE['steel'], 1: PALETTE['rose']}
TREAT_LABEL = {0: 'Control: no CBD pickup or dropoff',
               1: 'Treated: pickup or dropoff in CBD'}

for op in ['Uber', 'Lyft']:
    col = OP_POS[op]
    for t in [0, 1]:
        sub = monthly[(monthly['operator']==op) & (monthly['treat']==t)].sort_values('month')
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['mean_base_fare'],
            mode='lines+markers',
            name=TREAT_LABEL[t],
            line=dict(color=TREAT_COLOR[t], width=2),
            marker=dict(size=5),
            legendgroup=str(t),
            showlegend=(op == 'Uber'),
            hovertemplate=f"{TREAT_LABEL[t][:7]}<br>%{{x|%b %Y}}<br>$%{{y:.2f}}<extra></extra>",
        ), row=1, col=col)

# Event date line + shading for pre/post windows on both panels
for col in [1, 2]:
    fig.add_vline(x=EVENT_DATE, line=dict(color='#1a1a1a', dash='dash', width=1.4),
                  row=1, col=col)
    fig.add_vrect(x0=PRE_START, x1=PRE_END, fillcolor=PALETTE['slate'], opacity=0.07,
                  layer='below', line_width=0, row=1, col=col)
    fig.add_vrect(x0=POST_START, x1=POST_END, fillcolor=PALETTE['gold'], opacity=0.07,
                  layer='below', line_width=0, row=1, col=col)

fig.update_yaxes(title='Mean base fare per trip (USD)', col=1)
fig.update_xaxes(title=None)
fig.update_layout(
    height=480,
    legend=dict(
        orientation='h',
        yanchor='top', y=-0.16,
        xanchor='center', x=0.5,
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.9)',
    ),
    margin=dict(l=70, r=30, t=60, b=95),
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'cbd_did_event_study.html'
save_chart(fig, OUT)
print(f'wrote {OUT.name}')
