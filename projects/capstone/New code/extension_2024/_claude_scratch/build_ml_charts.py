"""
Builds Plotly versions of the ML charts in the page.html style.

Chart 1 (SHAP feature importance, page.html chart 08 style): horizontal bar
of top-12 SHAP features for the headline structural spec. Reads
intermediary/e4_shap_structural.csv. Writes
Graphics/NB4/shap_feature_importance.html.

Chart 2 (coefficient agreement, page.html chart 07 style): grouped
horizontal bars of Lasso/Ridge/ElasticNet standardised coefficients.
Requires intermediary/e4_coefs_structural.csv produced by the patched
e4_ml notebook. Writes Graphics/NB4/coef_agreement.html. Skipped (with a
print) if the input CSV is missing.

Chart 3 (predictions vs observed, appendix): scatter of actual vs OOF
predicted ECI for the headline model. Reads
intermediary/e4_predictions.csv. Writes
Graphics/NB4/predictions_vs_observed.html.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
INTER = EXT / 'intermediary'
OUT = EXT / 'Graphics' / 'NB4'
OUT.mkdir(parents=True, exist_ok=True)

FONT = 'IBM Plex Sans, system-ui, sans-serif'
NAVY = '#1F3A5F'
GRID = '#E5E7EB'

# Friendly labels for raw column names
LABELS = {
    'wide_resource_share':        'Wide resource share',
    'wide_resource_share_net':    'Wide resource share (net)',
    'log_HCI':                    'log Human Capital Index',
    'log_GFCF':                   'log Gross Fixed Capital Formation',
    'log_Pop':                    'log Population',
    'log_GDPpc':                  'log GDP per capita',
    'L1_ECI':                     'Lagged ECI (L1)',
    'Manufacturing':              'Manufacturing (% GDP)',
    'Industry':                   'Industry (% GDP)',
    'Agriculture':                'Agriculture (% GDP)',
    'Services':                   'Services (% GDP)',
    'Government revenue':         'Government revenue',
    'Primary net lending, General government, Percent of GDP': 'Primary net lending',
    'Rule of law index':          'Rule of law',
    'Political stability — estimate': 'Political stability',
    'Political corruption index': 'Political corruption',
    'Trade (% of GDP)':           'Trade openness',
    'Inflation, consumer prices (annual %)': 'Inflation',
    'Real interest rate (%)':     'Real interest rate',
    'Domestic credit to private sector (% of GDP)': 'Domestic credit',
    'Urban population (% of total population)': 'Urban population',
    'precious_share':             'Precious metals share',
    'precious_share_net':         'Precious metals share (net)',
    'ores_share':                 'Ores share',
    'ores_share_net':             'Ores share (net)',
    'hydrocarbon_share':          'Hydrocarbons share',
    'hydrocarbon_share_net':      'Hydrocarbons share (net)',
    'base_metals_share':          'Base metals share',
    'base_metals_share_net':      'Base metals share (net)',
    'Resource_HHI_trade':         'Resource concentration (HHI)',
    'post2019_x_hydrocarbon_share': 'Post-2019 × hydrocarbons',
    'post2019_x_ores_share':      'Post-2019 × ores',
    'post2019_x_base_metals_share': 'Post-2019 × base metals',
}

def label(name: str) -> str:
    return LABELS.get(name, name.replace('_', ' '))

# ---------------------------------------------------------------------------
# Chart 1 — SHAP feature importance (page.html chart 08 style)
# ---------------------------------------------------------------------------
shap = pd.read_csv(INTER / 'e4_shap_structural.csv')
top12 = shap.sort_values('shap_mean', ascending=False).head(12).copy()
top12['short'] = top12['feature'].map(label)
top12 = top12.sort_values('shap_mean', ascending=True)  # bars: largest at top

fig08 = go.Figure(go.Bar(
    y=top12['short'], x=top12['shap_mean'], orientation='h',
    marker=dict(color='#c97030', opacity=0.9,
                line=dict(color='white', width=0.5)),
    hovertemplate='%{y}: %{x:.3f}<extra>XGBoost SHAP</extra>',
))
fig08.update_layout(
    height=500,
    margin=dict(l=180, r=40, t=20, b=70),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    xaxis=dict(title=dict(text='Mean absolute SHAP value (XGBoost)',
                          font=dict(size=12)),
               gridcolor=GRID, zeroline=False,
               tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=11)),
    showlegend=False,
)
out_shap = OUT / 'shap_feature_importance.html'
fig08.write_html(str(out_shap), include_plotlyjs='cdn', full_html=True)
print(f'Wrote: {out_shap}')
print(f'  Top 12 features by mean |SHAP|:')
for _, r in top12.iloc[::-1].iterrows():
    print(f'    {r["short"]:35s} {r["shap_mean"]:.3f}')

# ---------------------------------------------------------------------------
# Chart 2 — Coefficient agreement (page.html chart 07 style)
# Requires e4_coefs_structural.csv from the patched e4 notebook.
# ---------------------------------------------------------------------------
coef_csv = INTER / 'e4_ml_coefficients.csv'
if coef_csv.exists():
    coef = pd.read_csv(coef_csv)
    # expected columns: feature, lasso, ridge, en (averaged across M imputations)
    coef['short'] = coef['feature'].map(label)
    coef['abs_avg'] = coef[['lasso', 'ridge', 'en']].abs().mean(axis=1)
    top = coef.sort_values('abs_avg', ascending=True).tail(12)

    MC = {'LASSO': '#c23a3a', 'Ridge': '#3498DB', 'Elastic Net': '#2e7d4a'}
    KEY = {'LASSO': 'lasso', 'Ridge': 'ridge', 'Elastic Net': 'en'}

    fig07 = go.Figure()
    fig07.add_vline(x=0, line=dict(color='#c9cfd6', width=1.5))
    for m, c in MC.items():
        fig07.add_trace(go.Bar(
            y=top['short'], x=top[KEY[m]], name=m, orientation='h',
            marker=dict(color=c, opacity=0.88,
                        line=dict(color='white', width=0.5)),
            hovertemplate=f'%{{y}}: %{{x:+.3f}}<extra>{m}</extra>',
        ))
    fig07.update_layout(
        height=560,
        margin=dict(l=180, r=40, t=20, b=70),
        font=dict(family=FONT, size=12, color=NAVY),
        paper_bgcolor='white', plot_bgcolor='white',
        barmode='group',
        xaxis=dict(title=dict(text='Coefficient (standardised inputs)',
                              font=dict(size=12)),
                   gridcolor=GRID, zeroline=False,
                   tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=11)),
        legend=dict(x=1.01, y=0.99, font=dict(size=11),
                    bgcolor='rgba(250,250,250,0.9)',
                    bordercolor=GRID, borderwidth=1),
    )
    out_coef = OUT / 'coef_agreement.html'
    fig07.write_html(str(out_coef), include_plotlyjs='cdn', full_html=True)
    print(f'Wrote: {out_coef}')
else:
    print(f'Skipped coefficient-agreement chart (e4_coefs_structural.csv not found). '
          f'Run the patched e4_ml notebook first.')

# ---------------------------------------------------------------------------
# Chart 3 — Predictions vs Observed (for appendix)
# ---------------------------------------------------------------------------
pred_csv = INTER / 'e4_predictions.csv'
if pred_csv.exists():
    preds = pd.read_csv(pred_csv)
    # merge in true ECI from the imputed panel
    import sys; sys.path.insert(0, str(EXT))
    from _mice_pool import iter_imputations
    truth = next(iter_imputations())[1][['Country Code', 'Year', 'Economic Complexity Index']].dropna()
    merged = truth.merge(preds, on=['Country Code', 'Year'], how='inner')

    lo = min(merged['Economic Complexity Index'].min(), merged['pred'].min()) - 0.1
    hi = max(merged['Economic Complexity Index'].max(), merged['pred'].max()) + 0.1

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode='lines',
        line=dict(color='#c97030', width=1.5, dash='dash'),
        name='45° line', hoverinfo='skip',
    ))
    fig_pred.add_trace(go.Scatter(
        x=merged['Economic Complexity Index'], y=merged['pred'],
        mode='markers',
        marker=dict(color=NAVY, size=5, opacity=0.4,
                    line=dict(color='white', width=0.3)),
        name='Country-year',
        hovertemplate='Actual: %{x:+.2f}<br>Predicted: %{y:+.2f}<extra></extra>',
    ))
    from sklearn.metrics import r2_score
    r2 = r2_score(merged['Economic Complexity Index'], merged['pred'])
    fig_pred.update_layout(
        height=520,
        margin=dict(l=60, r=40, t=20, b=60),
        font=dict(family=FONT, size=12, color=NAVY),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(title=dict(text='Actual ECI', font=dict(size=12)),
                   gridcolor=GRID, zeroline=False, range=[lo, hi]),
        yaxis=dict(title=dict(text='Predicted ECI (out-of-fold, pooled M = 5)',
                              font=dict(size=12)),
                   gridcolor=GRID, zeroline=False, range=[lo, hi]),
        legend=dict(x=0.02, y=0.98, font=dict(size=11),
                    bgcolor='rgba(250,250,250,0.85)',
                    bordercolor=GRID, borderwidth=1),
    )
    fig_pred.add_annotation(
        xref='paper', yref='paper', x=0.98, y=0.05, xanchor='right',
        text=f'R<sup>2</sup> = {r2:.3f}  &nbsp;&nbsp;  N = {len(merged):,}',
        showarrow=False, font=dict(family=FONT, size=12, color=NAVY),
        bgcolor='rgba(250,250,250,0.9)', bordercolor=GRID, borderwidth=1,
        borderpad=6,
    )
    out_pred = OUT / 'predictions_vs_observed.html'
    fig_pred.write_html(str(out_pred), include_plotlyjs='cdn', full_html=True)
    print(f'Wrote: {out_pred}')
    print(f'  N = {len(merged):,}, R² = {r2:.3f}')
