"""
edge_vs_core_viz_switch.py

Single-panel scatter with a per-cluster switcher. One zone = one point.
Top 3 gainers + top 3 losers labelled per cluster, Spearman rho annotated.

Reads:  outputs/tables/edge_vs_core_test.csv
Writes: outputs/sample/edge_vs_core_scatter.html (overwrites the 2x2 version)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import save_chart, PALETTE

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'
CHART_DIR = ROOT / 'outputs' / 'sample'

LBL = ['Manhattan', 'Brooklyn', 'Bronx + Upper Manhattan', 'Queens']
COLORS = [PALETTE['navy'], PALETTE['rose'], PALETTE['gold'], PALETTE['sage']]

# Shorten common long names so labels fit on the chart
NAME_FIX = {
    'Washington Heights South': 'Wash. Heights S',
    'Washington Heights North': 'Wash. Heights N',
    'Flushing Meadows-Corona Park': 'Flushing Meadows',
    'Long Island City/Hunters Point': 'LIC / Hunters Pt',
    'Long Island City/Queens Plaza': 'LIC / Queens Plz',
    'Penn Station/Madison Sq West': 'Penn Station',
    'Downtown Brooklyn/MetroTech': 'Downtown Bklyn',
    'East Flatbush/Remsen Village': 'East Flatbush',
    'Mott Haven/Port Morris': 'Mott Haven',
    'East Concourse/Concourse Village': 'East Concourse',
    'Yorkville West': 'Yorkville W',
    'East Harlem South': 'East Harlem S',
    'East Harlem North': 'East Harlem N',
}

df = pd.read_csv(TABLES / 'edge_vs_core_test.csv')
df['zone_short'] = df['zone'].map(lambda z: NAME_FIX.get(z, z))
# Distance from cluster center: 0 means zone sits at cluster centroid;
# 0.5 means zone sits on the boundary with another cluster (equidistant).
df['distance'] = df['edge_ness']
print(f'Loaded {len(df)} zones')

vols = df['total_pickups'].clip(lower=1)
def size_scale(v):
    return 5 + 22*(np.sqrt(v)-np.sqrt(vols.min()))/(np.sqrt(vols.max())-np.sqrt(vols.min())+1e-9)

fig = go.Figure()

# One trace per cluster, only one visible at a time
n_clusters = 4
traces_per_cluster = []  # remember index for visibility toggling

for c in range(n_clusters):
    g = df[df['cluster']==c].copy().sort_values('total_pickups')
    sizes = size_scale(g['total_pickups'])
    fig.add_trace(go.Scatter(
        x=g['distance'], y=g['delta_pp'],
        mode='markers',
        marker=dict(size=sizes, color=COLORS[c], opacity=0.78,
                    line=dict(width=0.5, color='rgba(0,0,0,0.4)')),
        text=[f"<b>{z}</b><br>{b}<br>distance: {p:.2f}<br>Δ share: {d:+.3f}pp<br>volume: {v:,.0f}"
              for z,b,p,d,v in zip(g['zone'], g['borough'], g['distance'], g['delta_pp'], g['total_pickups'])],
        hovertemplate='%{text}<extra></extra>',
        name=LBL[c],
        visible=(c==0),
        showlegend=False,
    ))
    traces_per_cluster.append(len(fig.data)-1)

# Pre-compute annotation layouts: one list per cluster.
# Labels are placed with bigger offsets and a horizontal stagger by rank
# so the boxes don't overlap each other or cover nearby dots.
def make_annotations_for_cluster(c):
    g = df[df['cluster']==c].copy()
    rho, p = spearmanr(g['distance'], g['delta_pp'])
    sig = '*' if p < 0.05 else ''
    top_gainers = g.nlargest(3, 'delta_pp').sort_values('distance').reset_index(drop=True)
    top_losers  = g.nsmallest(3, 'delta_pp').sort_values('distance').reset_index(drop=True)

    # Fan offsets: leftmost label goes upper-left, middle straight up, rightmost upper-right
    GAINER_OFFSETS = [(-45, -30), (0, -42), (45, -30)]
    LOSER_OFFSETS  = [(-45,  30), (0,  42), (45,  30)]

    anns = []
    for i, rr in top_gainers.iterrows():
        ax, ay = GAINER_OFFSETS[i] if i < len(GAINER_OFFSETS) else (0, -30)
        anns.append(dict(
            x=rr['distance'], y=rr['delta_pp'],
            text=rr['zone_short'],
            showarrow=True, arrowhead=0, arrowsize=0.5, arrowwidth=0.8, arrowcolor='#666',
            ax=ax, ay=ay,
            font=dict(size=11, color='#1a1a1a'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#c0c4cc', borderwidth=0.5, borderpad=3,
        ))
    for i, rr in top_losers.iterrows():
        ax, ay = LOSER_OFFSETS[i] if i < len(LOSER_OFFSETS) else (0, 30)
        anns.append(dict(
            x=rr['distance'], y=rr['delta_pp'],
            text=rr['zone_short'],
            showarrow=True, arrowhead=0, arrowsize=0.5, arrowwidth=0.8, arrowcolor='#666',
            ax=ax, ay=ay,
            font=dict(size=11, color='#1a1a1a'),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#c0c4cc', borderwidth=0.5, borderpad=3,
        ))
    anns.append(dict(
        xref='paper', yref='paper',
        x=0.98, y=0.98, xanchor='right', yanchor='top',
        text=f"Spearman ρ = {rho:+.2f}{sig}  (p = {p:.3f}, n={len(g)})",
        showarrow=False,
        font=dict(size=11, color='#333', family='IBM Plex Sans'),
        bgcolor='rgba(255,255,255,0.92)',
        bordercolor='#c0c4cc', borderwidth=1, borderpad=5,
    ))
    return anns

cluster_anns = [make_annotations_for_cluster(c) for c in range(n_clusters)]

# Initial layout: cluster 0 (Manhattan)
fig.add_hline(y=0, line_color='#aab0bd', line_width=1, line_dash='dot')

buttons = []
for c in range(n_clusters):
    vis = [i == traces_per_cluster[c] for i in range(len(fig.data))]
    buttons.append(dict(
        label=LBL[c],
        method='update',
        args=[
            {'visible': vis},
            {'annotations': cluster_anns[c]},
        ],
    ))

fig.update_layout(
    annotations=cluster_anns[0],
    xaxis=dict(
        title='Distance from cluster center',
        range=[-0.02, 0.85],
    ),
    yaxis=dict(title='Change in share of total Uber pickups, 2019 to 2026 (pp)'),
    height=680,
    margin=dict(l=70, r=30, t=70, b=60),
    updatemenus=[dict(
        type='buttons',
        direction='right',
        x=0.5, xanchor='center',
        y=1.06, yanchor='bottom',
        active=0,
        showactive=True,
        buttons=buttons,
        font=dict(size=11),
        bgcolor='white',
        bordercolor='#9aa3b2',
        pad=dict(r=6, l=6, t=4, b=4),
    )],
)

OUT = CHART_DIR / 'edge_vs_core_scatter.html'
save_chart(fig, OUT)
print(f'wrote {OUT.name}')
