"""Build viz_1_descriptive.ipynb for the extension_2024/ workspace.

Mirrors the structure of New code/viz_1_descriptive.ipynb (the rents-based
capstone's descriptive-viz notebook): a setup cell, then one Chart cell
per descriptive output. The extension version currently has one chart:
the resource export-share world map.

Outputs the notebook to:
  extension_2024/viz_1_descriptive.ipynb

When that notebook is executed, it writes:
  extension_2024/Graphics/NB1/resource_export_share_map.html
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/viz_1_descriptive.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text.splitlines(keepends=True),
    }


cells = []

# ----------------------------------------------------------------------------
cells.append(md('''# Viz 1 — Descriptive Charts (Extension)

Renders the descriptive-tier charts for the trade-side extension.
The structure mirrors `New code/viz_1_descriptive.ipynb` so the
extension page picks up the same visual language as the original
rents-based page.

| Chart | Page heading | Artifact |
|---|---|---|
| 1 | Resource Export Share | trade_metrics.csv |

The notebook reads from `intermediary/` and writes to `Graphics/NB1/`.
No model fitting, no sample logic. If a chart looks wrong, the fix
is in the prep notebook that produced the artifact (e1b for trade,
e2 for the merged panel), not here.
'''))

# ----------------------------------------------------------------------------
cells.append(code('''import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

EXT = Path('.').resolve()
INTER = EXT / 'intermediary'
OUT = EXT / 'Graphics' / 'NB1'
OUT.mkdir(parents=True, exist_ok=True)

# Style constants — inline rather than imported, to keep extension_2024/
# self-contained (the original viz_1 uses _style.py from New code/).
FONT   = 'Public Sans, sans-serif'
NAVY   = '#1a2744'
GRID   = '#e2e6eb'
BG     = 'white'

YL_OR_RD = [
    [0.000, "rgb(255,255,204)"], [0.125, "rgb(255,237,160)"],
    [0.250, "rgb(254,217,118)"], [0.375, "rgb(254,178,76)"],
    [0.500, "rgb(253,141,60)"],  [0.625, "rgb(252,78,42)"],
    [0.750, "rgb(227,26,28)"],   [0.875, "rgb(189,0,38)"],
    [1.000, "rgb(128,0,38)"],
]

# Defunct / aggregate ISO3 codes to drop from rankings and maps
DEFUNCT = {'ZA1', 'SCG', 'YUG', 'SUN', 'DDR', 'CSK', 'ANT', 'PCI', 'NTZ'}

# Hand-coded country names for hover labels on the top entries
NAME = {
    'IRQ': 'Iraq', 'LBY': 'Libya', 'DZA': 'Algeria', 'COD': 'DRC',
    'AGO': 'Angola', 'KWT': 'Kuwait', 'QAT': 'Qatar', 'SAU': 'Saudi Arabia',
    'NGA': 'Nigeria', 'IRN': 'Iran', 'GAB': 'Gabon', 'AZE': 'Azerbaijan',
    'OMN': 'Oman', 'RUS': 'Russia', 'USA': 'United States', 'CHN': 'China',
    'CAN': 'Canada', 'AUS': 'Australia', 'NOR': 'Norway', 'CHL': 'Chile',
    'PER': 'Peru', 'MEX': 'Mexico', 'ZMB': 'Zambia', 'BWA': 'Botswana',
    'MNG': 'Mongolia', 'GHA': 'Ghana', 'MLI': 'Mali', 'BFA': 'Burkina Faso',
    'GIN': 'Guinea', 'COG': 'Republic of Congo', 'TLS': 'Timor-Leste',
    'PNG': 'Papua New Guinea', 'BOL': 'Bolivia', 'IDN': 'Indonesia',
    'KAZ': 'Kazakhstan', 'ZAF': 'South Africa', 'ARE': 'United Arab Emirates',
    'VEN': 'Venezuela', 'BHR': 'Bahrain', 'TTO': 'Trinidad and Tobago',
    'COL': 'Colombia', 'BRA': 'Brazil', 'IND': 'India', 'TUR': 'Turkey',
    'ECU': 'Ecuador', 'EGY': 'Egypt',
}

print(f'Working dir: {EXT}')
print(f'Output dir:  {OUT}')
'''))

# ----------------------------------------------------------------------------
cells.append(md('''## Chart 1 — Resource Export Share (world map)

World choropleth of the mean resource share in the country's export
basket, averaged across the sample window. Two buttons at the top
switch the view:

- **Gross**: commodity exports / total exports. Bounded at zero.
  Inflated for re-export hubs (refiners, transshipment ports) that
  import commodities and re-export them without domestic extraction.
- **Net**: (commodity exports − commodity imports) / total exports.
  Strips re-export traffic from the numerator. Can go slightly
  negative for services-heavy economies because the denominator
  excludes services revenue while commodity imports are still
  subtracted.

Reads `intermediary/trade_metrics.csv`. Writes
`Graphics/NB1/resource_export_share_map.html`, which is the path the
page references.
'''))

# ----------------------------------------------------------------------------
cells.append(code('''tm = pd.read_csv(INTER / 'trade_metrics.csv')
print(f'Loaded trade_metrics: {tm.shape}')

agg = (tm[~tm['Country Code'].isin(DEFUNCT)]
       .groupby('Country Code', as_index=False)
       .agg(mean_wide      =('wide_resource_share',     'mean'),
            mean_wide_net  =('wide_resource_share_net', 'mean'),
            mean_hydro     =('hydrocarbon_share',       'mean'),
            mean_ores      =('ores_share',              'mean'),
            mean_base      =('base_metals_share',       'mean'),
            mean_prec      =('precious_share',          'mean'),
            mean_hydro_net =('hydrocarbon_share_net',   'mean'),
            mean_ores_net  =('ores_share_net',          'mean'),
            mean_base_net  =('base_metals_share_net',   'mean'),
            mean_prec_net  =('precious_share_net',      'mean'))
       .dropna(subset=['mean_wide']))

agg['country_name'] = agg['Country Code'].map(NAME).fillna(agg['Country Code'])

def hover_gross(r):
    return (f"<b>{r['country_name']}</b><br>"
            f"Gross wide share: {100*r['mean_wide']:.1f}%<br>"
            f"&nbsp;&nbsp;Hydrocarbons: {100*r['mean_hydro']:.1f}%<br>"
            f"&nbsp;&nbsp;Ores: {100*r['mean_ores']:.1f}%<br>"
            f"&nbsp;&nbsp;Base metals: {100*r['mean_base']:.1f}%<br>"
            f"&nbsp;&nbsp;Precious: {100*r['mean_prec']:.1f}%")

def hover_net(r):
    return (f"<b>{r['country_name']}</b><br>"
            f"Net wide share: {100*r['mean_wide_net']:+.1f}%<br>"
            f"&nbsp;&nbsp;Hydrocarbons (net): {100*r['mean_hydro_net']:+.1f}%<br>"
            f"&nbsp;&nbsp;Ores (net): {100*r['mean_ores_net']:+.1f}%<br>"
            f"&nbsp;&nbsp;Base metals (net): {100*r['mean_base_net']:+.1f}%<br>"
            f"&nbsp;&nbsp;Precious (net): {100*r['mean_prec_net']:+.1f}%")

agg['hover_gross'] = agg.apply(hover_gross, axis=1)
agg['hover_net']   = agg.apply(hover_net,   axis=1)

# Symmetric range for the net view, clipped to [-1, +1] so the colorscale
# matches the gross view rather than being dominated by services-heavy
# outliers (e.g. Maldives, Tonga at -20+ because the denominator is goods
# exports while commodity imports are still subtracted).
NET_LIM = 1.0

COLORBAR_KW = dict(
    thickness=14, len=0.55,
    bgcolor='rgba(255,255,255,0.7)',
    bordercolor=GRID, borderwidth=1, outlinewidth=0,
)

trace_gross = go.Choropleth(
    locations=agg['Country Code'],
    z=agg['mean_wide'],
    text=agg['hover_gross'],
    colorscale=YL_OR_RD,
    zmin=0, zmax=1.0,
    colorbar=dict(title=dict(text='Gross share',
                             font=dict(family=FONT, size=11)),
                  tickformat='.0%', **COLORBAR_KW),
    marker=dict(line=dict(color='#c9cfd6', width=0.5)),
    hovertemplate='%{text}<extra></extra>',
    visible=True,
)

trace_net = go.Choropleth(
    locations=agg['Country Code'],
    z=agg['mean_wide_net'].clip(-NET_LIM, NET_LIM),
    text=agg['hover_net'],
    colorscale='RdBu_r',
    zmin=-NET_LIM, zmid=0, zmax=NET_LIM,
    colorbar=dict(title=dict(text='Net share',
                             font=dict(family=FONT, size=11)),
                  tickformat='.0%', **COLORBAR_KW),
    marker=dict(line=dict(color='#c9cfd6', width=0.5)),
    hovertemplate='%{text}<extra></extra>',
    visible=False,
)

fig_map = go.Figure(data=[trace_gross, trace_net])
fig_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        landcolor='#f0f2f5',
        coastlinecolor='#9fa6b0',
        bgcolor='white',
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor=BG,
    height=520,
)

# Build the Plotly HTML, then inject a styled <select> control on the
# right that mirrors the production-intensity map's controls. Native
# HTML select + JS calling Plotly.restyle, rather than Plotly's own
# updatemenus, so the look matches the production map exactly.
PLOTLY_HTML = fig_map.to_html(
    full_html=True, include_plotlyjs='cdn',
    config=dict(displayModeBar=False, responsive=True),
)

CONTROLS = """
<style>body{margin:0;overflow:hidden;}</style>
<div style="position:fixed;top:10px;right:10px;z-index:1000;background:white;
padding:10px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.06);
border:1px solid #dde1e7;font-family:Public Sans, -apple-system, BlinkMacSystemFont, sans-serif;">
  <label style="font-weight:600;color:#1a2744;margin-right:8px;
  font-size:13px;">View:</label>
  <select id="viewSelect" style="padding:6px;border:1px solid #dde1e7;
  border-radius:4px;font-size:13px;font-family:Public Sans, -apple-system, BlinkMacSystemFont, sans-serif;">
    <option value="0">Gross exports</option>
    <option value="1">Net exports</option>
  </select>
</div>

<script>
setTimeout(function(){
  document.getElementById('viewSelect').addEventListener('change', function(){
    const vis = this.value === '0' ? [true, false] : [false, true];
    const p = document.getElementsByClassName('plotly-graph-div')[0];
    if (p) Plotly.restyle(p, {visible: vis});
  });
}, 100);
</script>
"""

final_html = PLOTLY_HTML.replace('</body>', CONTROLS + '\\n</body>')

out_map = OUT / 'resource_export_share_map.html'
with open(out_map, 'w', encoding='utf-8') as fh:
    fh.write(final_html)
print(f'Wrote: {out_map}')
print(f'  {len(agg)} countries plotted')
print(f'  Net range: [{agg["mean_wide_net"].min():+.3f}, {agg["mean_wide_net"].max():+.3f}]')
'''))

# ----------------------------------------------------------------------------
cells.append(md('''## Chart 2 — Median ECI by Cluster (over time)

For each of the four aggregate resource-rich clusters, plots the
median Economic Complexity Index across member countries by year.
Cluster colours match the description boxes on the page and the
cluster map. ECI is pooled across the M = 5 MICE imputations.

Reads `intermediary/Master_v2_clusters.csv` (for cluster assignments)
and the imputed panels (for ECI). Writes
`Graphics/NB1/median_eci_by_cluster.html`, the path the page
references.
'''))

# ----------------------------------------------------------------------------
cells.append(code('''import sys
sys.path.insert(0, str(EXT))
from _mice_pool import iter_imputations

cl = pd.read_csv(INTER / 'Master_v2_clusters.csv')[['Country Code', 'RR_k4_aggregate']].dropna()

# Pool ECI across M imputations by averaging per (country, year)
panels = []
for _imp_id, panel in iter_imputations():
    panels.append(panel[['Country Code', 'Year', 'Economic Complexity Index']])
combined = (pd.concat(panels)
              .groupby(['Country Code', 'Year'], as_index=False)
              .mean())
combined = combined.merge(cl, on='Country Code', how='inner')

CLUSTER_NAMES = {
    0: 'Mixed Extractives',
    1: 'Hydrocarbon Petrostates',
    2: 'Base-Metals Exporters',
    3: 'Precious Metals and Stones',
}
CLUSTER_COLOURS = {
    0: '#2A9D8F',  # teal
    1: '#457B9D',  # blue
    2: '#E63946',  # red
    3: '#E9C46A',  # yellow
}

# median ECI by (cluster, year), then a 3-year rolling mean with
# min_periods=1 (first two years use whatever data is available, so the
# series starts at year 0 rather than at year 2).
med = (combined.groupby(['RR_k4_aggregate', 'Year'])['Economic Complexity Index']
                .median()
                .reset_index()
                .rename(columns={'Economic Complexity Index': 'median_eci'}))
med['median_eci'] = (med.sort_values(['RR_k4_aggregate', 'Year'])
                       .groupby('RR_k4_aggregate')['median_eci']
                       .transform(lambda s: s.rolling(window=3, min_periods=1).mean()))

fig_eci = go.Figure()
# draw clusters in panel-mean ECI order, low to high, so the legend reads
# in the order that matches the y-axis
order = (med.groupby('RR_k4_aggregate')['median_eci']
            .mean().sort_values().index.tolist())
for cid in order:
    sub = med[med['RR_k4_aggregate'] == cid].sort_values('Year')
    # Pre-format hover labels to eliminate any float precision leakage.
    hover_text = [f'{int(yr)}: {val:+.1f}' for yr, val in zip(sub['Year'], sub['median_eci'])]
    fig_eci.add_trace(go.Scatter(
        x=sub['Year'], y=sub['median_eci'],
        mode='lines+markers',
        line=dict(color=CLUSTER_COLOURS[int(cid)], width=2.2),
        marker=dict(size=5),
        name=CLUSTER_NAMES[int(cid)],
        text=hover_text,
        hoverinfo='text+name',
        hovertemplate='%{text}<extra>' + CLUSTER_NAMES[int(cid)] + '</extra>',
    ))

# reference line at ECI = 0 (global mean)
fig_eci.add_hline(y=0, line=dict(color='#9fa6b0', width=1, dash='dash'),
                  annotation_text='Global mean', annotation_position='top right',
                  annotation_font=dict(family=FONT, size=11, color='#6b7280'))

fig_eci.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=20, b=10),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor=BG, plot_bgcolor=BG,
    xaxis=dict(title=dict(text='Year', font=dict(size=12)),
               gridcolor=GRID, zeroline=False,
               tickformat='d', hoverformat='d',
               showspikes=False,
               tickfont=dict(family=FONT, size=11)),
    yaxis=dict(title=dict(text='Median Economic Complexity Index', font=dict(size=12)),
               gridcolor=GRID, zeroline=False,
               tickformat='.1f', hoverformat='.1f',
               showspikes=False,
               tickfont=dict(family=FONT, size=11)),
    hovermode='closest',
    hoverlabel=dict(bgcolor='white', bordercolor='#cbd5e1',
                    font=dict(family=FONT, size=12, color='#1F3A5F')),
    legend=dict(orientation='h', yanchor='bottom', y=-0.28,
                xanchor='center', x=0.5,
                font=dict(family=FONT, size=11),
                bgcolor='rgba(255,255,255,0.6)',
                bordercolor=GRID, borderwidth=1),
)
out_eci = OUT / 'median_eci_by_cluster.html'
fig_eci.write_html(str(out_eci), include_plotlyjs='cdn', full_html=True)
print(f'Wrote: {out_eci}')
print(f'  panel-mean median ECI by cluster (sorted low to high):')
for cid in order:
    nm = CLUSTER_NAMES[int(cid)]
    m  = med[med['RR_k4_aggregate'] == cid]['median_eci'].mean()
    print(f'    {nm:30s} {m:+.3f}')
'''))

# ----------------------------------------------------------------------------
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.4'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Wrote {NB_PATH}')
print(f'Cells: {len(cells)} ({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')
