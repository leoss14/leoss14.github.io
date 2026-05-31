"""
Builds a clean Plotly forest plot of M1 coefficients (two-way fixed
effects, country-clustered standard errors), pooled across M=5 MICE
imputations.

Outputs Graphics/NB5/m1_forest_plot.html, which the page embeds in place
of the ugly matplotlib PNG.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
INTER = EXT / 'intermediary'
OUT = EXT / 'Graphics' / 'NB5'
OUT.mkdir(parents=True, exist_ok=True)

# --- load & flatten ---------------------------------------------------------
w = pd.read_csv(INTER / 'e5_results_wide.csv', header=[0, 1])
w.columns = ['_'.join(map(str, c)).strip() for c in w.columns]
w = w.rename(columns={'Unnamed: 0_level_0_model': 'variable'})

m1 = (w[['variable',
         'beta_M1_2wayFE_country_cluster',
         'p_M1_2wayFE_country_cluster',
         'se_M1_2wayFE_country_cluster']]
        .dropna(subset=['beta_M1_2wayFE_country_cluster'])
        .rename(columns={
            'beta_M1_2wayFE_country_cluster': 'beta',
            'p_M1_2wayFE_country_cluster':   'p',
            'se_M1_2wayFE_country_cluster':  'se',
        }))

# drop COVID interactions (separate table on the page) and the header row
m1 = m1[~m1['variable'].str.startswith('post2019_')]
m1 = m1[m1['variable'] != 'variable']

m1['ci_low']  = m1['beta'] - 1.96 * m1['se']
m1['ci_high'] = m1['beta'] + 1.96 * m1['se']

# --- friendly variable labels ----------------------------------------------
LABELS = {
    'wide_resource_share':                       'Wide resource share',
    'log_HCI':                                   'log Human Capital Index',
    'log_GFCF':                                  'log Gross Fixed Capital Formation',
    'log_Pop':                                   'log Population',
    'Manufacturing':                             'Manufacturing (% GDP)',
    'Industry':                                  'Industry (% GDP)',
    'Agriculture':                               'Agriculture (% GDP)',
    'Services':                                  'Services (% GDP)',
    'Government revenue':                        'Government revenue',
    'Primary net lending, General government, Percent of GDP': 'Primary net lending',
    'Rule of law index':                         'Rule of law',
    'Political stability — estimate':            'Political stability',
    'Political corruption index':                'Political corruption',
    'Trade (% of GDP)':                          'Trade openness',
    'Inflation, consumer prices (annual %)':     'Inflation',
    'Real interest rate (%)':                    'Real interest rate',
    'Domestic credit to private sector (% of GDP)': 'Domestic credit',
    'Urban population (% of total population)':  'Urban population',
}
m1['label'] = m1['variable'].map(LABELS).fillna(m1['variable'])

# --- order: headline at top, then by absolute magnitude --------------------
headline = m1[m1['variable'] == 'wide_resource_share']
others = (m1[m1['variable'] != 'wide_resource_share']
          .assign(_abs=lambda d: d['beta'].abs())
          .sort_values('_abs', ascending=True))
plot_df = pd.concat([others, headline]).reset_index(drop=True)
# y positions: 0 at bottom, increasing up; headline at top
plot_df['y'] = np.arange(len(plot_df))

# --- colours ---------------------------------------------------------------
NAVY    = '#1F3A5F'
ACCENT  = '#E63946'   # for the headline row
GREY    = '#9CA3AF'
GREEN   = '#2A9D8F'   # significant
GRID    = '#E5E7EB'

def colour(p, is_head):
    if is_head:
        return ACCENT
    if p < 0.10:
        return NAVY
    return GREY

plot_df['colour'] = plot_df.apply(
    lambda r: colour(r['p'], r['variable'] == 'wide_resource_share'),
    axis=1)

# --- build figure ----------------------------------------------------------
fig = go.Figure()

# zero line
fig.add_vline(x=0, line=dict(color='#6B7280', width=1, dash='dash'))

# CI bars + points, drawn per-row so we can colour them individually
for _, r in plot_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[r['ci_low'], r['ci_high']],
        y=[r['y'], r['y']],
        mode='lines',
        line=dict(color=r['colour'], width=2),
        hoverinfo='skip',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[r['beta']], y=[r['y']],
        mode='markers',
        marker=dict(color=r['colour'], size=10,
                    line=dict(color='white', width=1.2)),
        hovertemplate=(
            f'<b>{r["label"]}</b><br>'
            f'β = {r["beta"]:+.3f}<br>'
            f'95%% CI: [{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]<br>'
            f'p = {r["p"]:.3g}<extra></extra>'
        ),
        showlegend=False,
    ))

# axis labels
fig.update_layout(
    height=520,
    margin=dict(l=10, r=20, t=20, b=10),
    font=dict(family='IBM Plex Sans, system-ui, sans-serif',
              size=12, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    xaxis=dict(
        title=dict(text='Coefficient (β) with 95% confidence interval',
                   font=dict(size=12)),
        gridcolor=GRID, zeroline=False,
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=plot_df['y'].tolist(),
        ticktext=plot_df['label'].tolist(),
        gridcolor=GRID, zeroline=False,
        tickfont=dict(size=11),
    ),
)

out_path = OUT / 'm1_forest_plot.html'
fig.write_html(str(out_path), include_plotlyjs='cdn', full_html=True)
print(f'Wrote: {out_path}')
print(f'  {len(plot_df)} variables plotted')
print(f'  Headline: wide_resource_share β = {headline["beta"].iloc[0]:+.3f}, '
      f'p = {headline["p"].iloc[0]:.4g}')
