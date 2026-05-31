"""
Generate descriptive charts for capstone extension page_new.html.

Builds three deliverables in the same style as viz_1_descriptive.ipynb:
  1. resource_export_share_map.html  - choropleth of mean wide resource share
  2. correlation_bar.html             - Pearson correlations with ECI, grouped by category
  3. Top-5 rankings printed to stdout for manual embedding

Outputs written to: extension_2024/Graphics/NB1/
"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
OUT = EXT / 'Graphics' / 'NB1'
OUT.mkdir(parents=True, exist_ok=True)

# Style constants, matching _style.py / viz_1_descriptive.ipynb
FONT   = 'Public Sans, sans-serif'
NAVY   = '#1a2744'
ACCENT = '#c23a3a'
GRID   = '#e2e6eb'
BG     = 'white'

YL_OR_RD = [
    [0.000, "rgb(255,255,204)"], [0.125, "rgb(255,237,160)"],
    [0.250, "rgb(254,217,118)"], [0.375, "rgb(254,178,76)"],
    [0.500, "rgb(253,141,60)"],  [0.625, "rgb(252,78,42)"],
    [0.750, "rgb(227,26,28)"],   [0.875, "rgb(189,0,38)"],
    [1.000, "rgb(128,0,38)"],
]

CAT_COLORS = {
    'Trade exposure (new)':   '#E74C3C',
    'Macro & Structure':      '#8B5CF6',
    'Finance & Investment':   '#E67E22',
    'Human Capital & Infra':  '#1ABC9C',
    'Governance':             '#3498DB',
}

# Defunct / aggregate ISO3 codes to drop from rankings
DEFUNCT = {'ZA1', 'SCG', 'YUG', 'SUN', 'DDR', 'CSK', 'ANT', 'PCI', 'NTZ'}

# Country-name lookup, hand-coded for top entries only
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

# ============================================================
# Load data
# ============================================================
tm = pd.read_csv(EXT / 'intermediary' / 'trade_metrics.csv')
print(f'Loaded trade_metrics: {tm.shape}')

# ============================================================
# 1. CHOROPLETH MAP: MEAN WIDE RESOURCE SHARE
# ============================================================
agg = (tm[~tm['Country Code'].isin(DEFUNCT)]
       .groupby('Country Code', as_index=False)
       .agg(mean_wide=('wide_resource_share', 'mean'),
            mean_hydro=('hydrocarbon_share', 'mean'),
            mean_ores=('ores_share', 'mean'),
            mean_base=('base_metals_share', 'mean'),
            mean_prec=('precious_share', 'mean'),
            mean_total_usd=('total_exports_usd', 'mean'),
            mean_resource_usd=('wide_resource_usd', 'mean'))
       .dropna(subset=['mean_wide']))

agg['country_name'] = agg['Country Code'].map(NAME).fillna(agg['Country Code'])

def hover(r):
    return (f"<b>{r['country_name']}</b><br>"
            f"Wide resource share: {100*r['mean_wide']:.1f}%<br>"
            f"&nbsp;&nbsp;Hydrocarbons: {100*r['mean_hydro']:.1f}%<br>"
            f"&nbsp;&nbsp;Ores: {100*r['mean_ores']:.1f}%<br>"
            f"&nbsp;&nbsp;Base metals: {100*r['mean_base']:.1f}%<br>"
            f"&nbsp;&nbsp;Precious: {100*r['mean_prec']:.1f}%")

agg['hover_txt'] = agg.apply(hover, axis=1)

fig_map = go.Figure(go.Choropleth(
    locations=agg['Country Code'],
    z=agg['mean_wide'],
    text=agg['hover_txt'],
    colorscale=YL_OR_RD,
    zmin=0,
    zmax=1.0,
    colorbar=dict(
        title=dict(text='Share', font=dict(family=FONT, size=11)),
        tickformat='.0%',
        thickness=14,
        len=0.55,
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor=GRID,
        borderwidth=1,
        outlinewidth=0,
    ),
    marker=dict(line=dict(color='#c9cfd6', width=0.5)),
    hovertemplate='%{text}<extra></extra>',
))
fig_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        landcolor='#f0f2f5',
        coastlinecolor='#9fa6b0',
        bgcolor='white',
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor=BG,
)
out_map = OUT / 'resource_export_share_map.html'
fig_map.write_html(str(out_map), include_plotlyjs='cdn', full_html=True)
print(f'Wrote: {out_map}')

# ============================================================
# 2. TOP-5 RANKINGS
# ============================================================
# Substantive resource exporters: mean wide_resource_usd >= $3B
substantive = agg[agg['mean_resource_usd'] >= 3e9].copy()

# Diversity: 1 - mean_herfindahl, computed across the resource sub-codes
# Use the resource_herfindahl already in trade_metrics
hhi = (tm[~tm['Country Code'].isin(DEFUNCT)]
       .groupby('Country Code', as_index=False)['resource_herfindahl'].mean()
       .rename(columns={'resource_herfindahl': 'mean_hhi'}))
substantive = substantive.merge(hhi, on='Country Code')
substantive['diversity'] = 1 - substantive['mean_hhi']

# Filter to: at least $3B mean resource exports AND wide share >= 25% (genuine resource-dependence)
diversified = substantive[substantive['mean_wide'] >= 0.25].sort_values('diversity', ascending=False).head(5)
print('\n=== MOST DIVERSIFIED (>=$3B and >=25% share, ranked by 1 - HHI) ===')
for _, r in diversified.iterrows():
    print(f"  {r['country_name']:30s}  diversity={r['diversity']:.2f}  share={100*r['mean_wide']:.1f}%")

# Highest intensity (share of total exports)
intensity = substantive.sort_values('mean_wide', ascending=False).head(5)
print('\n=== HIGHEST INTENSITY (% of total exports, conditional on >=$3B) ===')
for _, r in intensity.iterrows():
    print(f"  {r['country_name']:30s}  {100*r['mean_wide']:.1f}%")

# Largest absolute (total resource exports in USD)
absolute = substantive.sort_values('mean_resource_usd', ascending=False).head(5)
print('\n=== LARGEST ABSOLUTE (mean wide_resource_usd) ===')
for _, r in absolute.iterrows():
    print(f"  {r['country_name']:30s}  ${r['mean_resource_usd']/1e9:.0f}B")

# ============================================================
# 3. CORRELATIONS WITH ECI, BY CATEGORY
# ============================================================
mw = pd.read_csv(EXT / 'intermediary' / 'master_data_wide.csv')
m = mw.merge(tm[['Country Code', 'Year', 'wide_resource_share', 'hydrocarbon_share',
                  'ores_share', 'base_metals_share', 'precious_share',
                  'coal_share', 'crude_oil_share', 'refined_oil_share', 'gas_share',
                  'resource_herfindahl']],
              on=['Country Code', 'Year'], how='left')

# (display_label, source_col, category)
CANDIDATES = [
    ('Wide resource share',        'wide_resource_share',                                            'Trade exposure (new)'),
    ('Hydrocarbon share',          'hydrocarbon_share',                                              'Trade exposure (new)'),
    ('Ores share',                 'ores_share',                                                     'Trade exposure (new)'),
    ('Base metals share',          'base_metals_share',                                              'Trade exposure (new)'),
    ('Precious share',             'precious_share',                                                 'Trade exposure (new)'),
    ('Coal share',                 'coal_share',                                                     'Trade exposure (new)'),
    ('Crude oil share',            'crude_oil_share',                                                'Trade exposure (new)'),
    ('Gas share',                  'gas_share',                                                      'Trade exposure (new)'),
    ('Resource HHI',               'resource_herfindahl',                                            'Trade exposure (new)'),
    ('Manufacturing share',        'Manufacturing',                                                  'Macro & Structure'),
    ('Industry share',             'Industry',                                                       'Macro & Structure'),
    ('Services share',             'Services',                                                       'Macro & Structure'),
    ('Agriculture share',          'Agriculture',                                                    'Macro & Structure'),
    ('GDP per capita',             'GDP per capita (constant prices, PPP)',                          'Macro & Structure'),
    ('Trade openness',             'Trade (% of GDP)',                                               'Macro & Structure'),
    ('Inflation',                  'Inflation, consumer prices (annual %)',                          'Macro & Structure'),
    ('Domestic credit (% GDP)',    'Domestic credit to private sector (% of GDP)',                   'Finance & Investment'),
    ('Gross fixed capital',        'Gross fixed capital formation, all, Constant prices, Percent of GDP', 'Finance & Investment'),
    ('Government revenue',         'Government revenue',                                             'Finance & Investment'),
    ('Human capital index',        'Human capital index',                                            'Human Capital & Infra'),
    ('Urban population',           'Urban population (% of total population)',                       'Human Capital & Infra'),
    ('Rule of law',                'Rule of law index',                                              'Governance'),
    ('Political stability',        'Political stability — estimate',                                 'Governance'),
    ('Political corruption',       'Political corruption index',                                     'Governance'),
]

eci = m['Economic Complexity Index']
rows = []
for label, col, cat in CANDIDATES:
    if col not in m.columns:
        print(f'  skip (missing): {col}')
        continue
    pair = pd.concat([eci, m[col]], axis=1).dropna()
    if len(pair) < 100:
        continue
    r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    rows.append({'label': label, 'r': r, 'n': len(pair), 'cat': cat})

corr_df = pd.DataFrame(rows).sort_values('r', ascending=True)

print('\n=== CORRELATIONS WITH ECI ===')
for _, row in corr_df.iloc[::-1].iterrows():
    print(f"  {row['label']:28s}  r = {row['r']:+.3f}  (n={row['n']:>5d})  [{row['cat']}]")

# Plot grouped by category
fig_corr = go.Figure()
fig_corr.add_vline(x=0, line=dict(color='#9fa6b0', width=1.5))
for cat, color in CAT_COLORS.items():
    sub = corr_df[corr_df['cat'] == cat]
    if len(sub) == 0:
        continue
    fig_corr.add_trace(go.Bar(
        y=sub['label'],
        x=sub['r'],
        orientation='h',
        name=cat,
        legendgroup=cat,
        marker=dict(color=color, opacity=0.87, line=dict(color='white', width=0.4)),
        hovertemplate='<b>%{y}</b><br>Pearson r = %{x:+.3f}<extra></extra>',
    ))
fig_corr.update_layout(
    barmode='overlay',
    height=620,
    margin=dict(l=10, r=10, t=20, b=60),
    xaxis=dict(title='Pearson correlation with ECI',
               range=[-0.8, 0.8], gridcolor=GRID, zeroline=False,
               tickfont=dict(family=FONT, size=11)),
    yaxis=dict(gridcolor=GRID, showgrid=False,
               tickfont=dict(family=FONT, size=11), autorange='reversed'),
    legend=dict(title=dict(text='Category', font=dict(size=11)),
                font=dict(family=FONT, size=10),
                bgcolor='rgba(250,250,250,0.92)',
                bordercolor=GRID, borderwidth=1,
                orientation='h',
                yanchor='bottom', y=-0.18, xanchor='center', x=0.5),
    font=dict(family=FONT, size=12, color=NAVY),
    plot_bgcolor=BG,
    paper_bgcolor=BG,
)
out_corr = OUT / 'correlation_bar.html'
fig_corr.write_html(str(out_corr), include_plotlyjs='cdn', full_html=True)
print(f'\nWrote: {out_corr}')

print('\n========== DONE ==========')
