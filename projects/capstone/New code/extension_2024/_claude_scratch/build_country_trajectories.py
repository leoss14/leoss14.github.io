"""Paired ECI / wide-resource-share trajectories for the appendix.

Six countries spanning the resource-curse spectrum within the developing
panel: case study trio plus one representative per cluster archetype.
Each subplot shows ECI (solid, left axis) and wide resource share
(dashed, right axis) from 1995 to 2024.

Output: Graphics/NB3/country_trajectories.html
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
sys.path.insert(0, str(EXT))
from _mice_pool import iter_imputations

panels = [p for _, p in iter_imputations()]
df = (pd.concat(panels)
        .groupby(['Country Code', 'Country Name', 'Year'], as_index=False)
        .mean(numeric_only=True))
# wide_resource_share is already in the imputed panel

COUNTRIES = [
    ('COG', 'Republic of Congo',       'case study'),
    ('AZE', 'Azerbaijan',              'case study'),
    ('CHL', 'Chile',                   'case study'),
    ('SAU', 'Saudi Arabia',            'Hydrocarbon Petrostates'),
    ('BWA', 'Botswana',                'Precious Metals'),
    ('IDN', 'Indonesia',               'Mixed Extractives'),
]

NAVY = '#1F3A5F'
ACCENT = '#C23A3A'
GRID = '#E5E7EB'
FONT = 'IBM Plex Sans, system-ui, sans-serif'

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f'{n} <span style="color:#9ca3af;font-weight:400">· {tag}</span>'
                    for _, n, tag in COUNTRIES],
    specs=[[{'secondary_y': True}]*3, [{'secondary_y': True}]*3],
    vertical_spacing=0.16, horizontal_spacing=0.07,
)

for i, (code, name, _) in enumerate(COUNTRIES):
    r = i // 3 + 1
    c = i % 3 + 1
    sub = df[df['Country Code'] == code].sort_values('Year')
    # ECI (left axis, solid navy)
    eci = sub.dropna(subset=['Economic Complexity Index'])
    fig.add_trace(go.Scatter(
        x=eci['Year'], y=eci['Economic Complexity Index'],
        mode='lines',
        line=dict(color=NAVY, width=2),
        name='ECI',
        legendgroup='ECI',
        showlegend=(i == 0),
        hovertemplate=f'{name} %{{x:.0f}}<br>ECI: %{{y:+.1f}}<extra></extra>',
    ), row=r, col=c, secondary_y=False)
    # Wide resource share (right axis, dashed red)
    rs = sub.dropna(subset=['wide_resource_share'])
    fig.add_trace(go.Scatter(
        x=rs['Year'], y=rs['wide_resource_share'] * 100,
        mode='lines',
        line=dict(color=ACCENT, width=2, dash='dash'),
        name='Wide resource share',
        legendgroup='Wide resource share',
        showlegend=(i == 0),
        hovertemplate=f'{name} %{{x:.0f}}<br>Resource share: %{{y:.0f}}%<extra></extra>',
    ), row=r, col=c, secondary_y=True)

fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10),
                 tickformat='d', hoverformat='d', dtick=8)
fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10),
                 hoverformat='.1f', secondary_y=False)
fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=10),
                 ticksuffix='%', hoverformat='.0f', secondary_y=True)

# only outer axes get titles
fig.update_yaxes(title=dict(text='ECI', font=dict(size=11)),
                 row=1, col=1, secondary_y=False)
fig.update_yaxes(title=dict(text='ECI', font=dict(size=11)),
                 row=2, col=1, secondary_y=False)
fig.update_yaxes(title=dict(text='Resource share', font=dict(size=11)),
                 row=1, col=3, secondary_y=True)
fig.update_yaxes(title=dict(text='Resource share', font=dict(size=11)),
                 row=2, col=3, secondary_y=True)

fig.update_layout(
    height=560,
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(family=FONT, size=11, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=-0.16,
                xanchor='center', x=0.5,
                font=dict(family=FONT, size=11),
                bgcolor='rgba(255,255,255,0.6)',
                bordercolor=GRID, borderwidth=1),
)
# Make subplot titles smaller / consistent
fig.update_annotations(font=dict(family=FONT, size=12, color=NAVY))

out = EXT / 'Graphics' / 'NB3' / 'country_trajectories.html'
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out), include_plotlyjs='cdn', full_html=True)
print(f'Wrote: {out}')
