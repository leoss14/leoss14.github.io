"""Build the supplementary appendix charts to mirror page.html's tail.

Charts produced:
 - pca_scatter.html: PCA scatter coloured by cluster
 - pca_pc1_loadings.html, pca_pc2_loadings.html: feature loadings on PC1, PC2
 - variables_treemap.html: variables grouped by theme
 - data_sources_bubble.html: data sources, bubble sized by feature count
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
sys.path.insert(0, str(EXT))
from _mice_pool import iter_imputations

OUT = EXT / 'Graphics' / 'NB3'
OUT.mkdir(parents=True, exist_ok=True)
FONT = 'IBM Plex Sans, system-ui, sans-serif'
NAVY = '#1F3A5F'
GRID = '#E5E7EB'

CLUSTER_NAMES = {0: 'Mixed Extractives', 1: 'Hydrocarbon Petrostates',
                 2: 'Base-Metals Exporters', 3: 'Precious Metals and Stones'}
CLUSTER_COLOURS = {0: '#2A9D8F', 1: '#457B9D', 2: '#E63946', 3: '#E9C46A'}

# ---------- PCA on cluster features --------------------------------------------------
clusters = pd.read_csv(EXT / 'intermediary' / 'Master_v2_clusters.csv')
panel0 = next(iter_imputations())[1]

AGG_SHARES = ['coal_share', 'crude_oil_share', 'refined_oil_share', 'gas_share',
              'ores_share', 'base_metals_share', 'precious_share']
SHARE_LABELS = {'coal_share': 'Coal', 'crude_oil_share': 'Crude oil',
                'refined_oil_share': 'Refined oil', 'gas_share': 'Gas',
                'ores_share': 'Ores', 'base_metals_share': 'Base metals',
                'precious_share': 'Precious metals'}

# country-mean shares for the resource-rich subsample
mean_shares = panel0.groupby('Country Code')[AGG_SHARES].mean()
rr = clusters.dropna(subset=['RR_k4_aggregate'])[['Country Code', 'Country Name', 'RR_k4_aggregate']]
df = mean_shares.merge(rr, left_index=True, right_on='Country Code', how='inner')
df['cluster_name'] = df['RR_k4_aggregate'].astype(int).map(CLUSTER_NAMES)

X = np.log1p(df[AGG_SHARES].values)  # log1p as in e3
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xs)
scores = pca.transform(Xs)
df['PC1'] = scores[:, 0]
df['PC2'] = scores[:, 1]
loadings = pca.components_  # shape (2, n_features)
exp_var = pca.explained_variance_ratio_

# 1) PCA scatter
fig = go.Figure()
for cid in sorted(df['RR_k4_aggregate'].astype(int).unique()):
    sub = df[df['RR_k4_aggregate'] == cid]
    fig.add_trace(go.Scatter(
        x=sub['PC1'], y=sub['PC2'], mode='markers+text',
        text=sub['Country Code'], textposition='top center',
        textfont=dict(size=9, color='#6b7280'),
        marker=dict(size=11, color=CLUSTER_COLOURS[int(cid)],
                    line=dict(color='white', width=1)),
        name=CLUSTER_NAMES[int(cid)],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
    ))
fig.add_hline(y=0, line=dict(color='#cbd5e1', width=1))
fig.add_vline(x=0, line=dict(color='#cbd5e1', width=1))
fig.update_layout(
    height=520, margin=dict(l=10, r=10, t=20, b=10),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    xaxis=dict(title=dict(text=f'PC1 ({100*exp_var[0]:.0f}% variance)', font=dict(size=12)),
               gridcolor=GRID, zeroline=False, tickfont=dict(size=11),
               tickformat='.1f', hoverformat='.2f'),
    yaxis=dict(title=dict(text=f'PC2 ({100*exp_var[1]:.0f}% variance)', font=dict(size=12)),
               gridcolor=GRID, zeroline=False, tickfont=dict(size=11),
               tickformat='.1f', hoverformat='.2f'),
    legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='center', x=0.5,
                font=dict(size=11), bgcolor='rgba(255,255,255,0.6)',
                bordercolor=GRID, borderwidth=1),
)
fig.write_html(str(OUT / 'pca_scatter.html'), include_plotlyjs='cdn', full_html=True)
print(f'Wrote {OUT / "pca_scatter.html"}')

# 2,3) PC1 and PC2 loadings (horizontal bars)
def loadings_chart(pc_idx, title_suffix, colour, out_path):
    vals = loadings[pc_idx]
    labels = [SHARE_LABELS[s] for s in AGG_SHARES]
    pairs = sorted(zip(labels, vals), key=lambda kv: kv[1])
    labs = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    fig2 = go.Figure(go.Bar(
        x=vs, y=labs, orientation='h',
        marker=dict(color=colour, line=dict(color='white', width=0.5)),
        hovertemplate='%{y}: %{x:+.2f}<extra></extra>',
    ))
    fig2.add_vline(x=0, line=dict(color='#9ca3af', width=1))
    fig2.update_layout(
        height=420, margin=dict(l=150, r=30, t=20, b=60),
        font=dict(family=FONT, size=12, color=NAVY),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(title=dict(text=f'PC{pc_idx+1} loading', font=dict(size=12)),
                   gridcolor=GRID, zeroline=False, tickfont=dict(size=11),
                   tickformat='.1f', hoverformat='.2f'),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=11)),
        showlegend=False,
    )
    fig2.write_html(str(out_path), include_plotlyjs='cdn', full_html=True)
    print(f'Wrote {out_path}')

loadings_chart(0, 'PC1', '#457B9D', OUT / 'pca_pc1_loadings.html')
loadings_chart(1, 'PC2', '#E63946', OUT / 'pca_pc2_loadings.html')

# 4) Variables treemap
VARIABLES = [
    # (label, theme)
    ('Wide resource share',       'Resource exposure'),
    ('Hydrocarbon share',          'Resource exposure'),
    ('Ores share',                  'Resource exposure'),
    ('Base metals share',           'Resource exposure'),
    ('Precious metals share',      'Resource exposure'),
    ('Resource concentration (HHI)', 'Resource exposure'),
    ('post-2019 × hydrocarbons',   'Resource exposure'),
    ('post-2019 × ores',            'Resource exposure'),
    ('post-2019 × base metals',     'Resource exposure'),
    ('log Human Capital Index',     'Investment & labour'),
    ('log Gross Fixed Capital Formation', 'Investment & labour'),
    ('log Population',              'Investment & labour'),
    ('Agriculture (% GDP)',         'Sectoral composition'),
    ('Industry (% GDP)',            'Sectoral composition'),
    ('Manufacturing (% GDP)',       'Sectoral composition'),
    ('Services (% GDP)',            'Sectoral composition'),
    ('Trade openness (% GDP)',      'Macro & financial'),
    ('Domestic credit (% GDP)',     'Macro & financial'),
    ('Inflation (5-yr rolling)',    'Macro & financial'),
    ('Real interest rate (5-yr roll)', 'Macro & financial'),
    ('Urban population (% total)',  'Macro & financial'),
    ('Rule of law',                 'Institutions'),
    ('Political stability',         'Institutions'),
    ('Political corruption',        'Institutions'),
    ('Government revenue (% GDP)',  'Fiscal'),
    ('Primary net lending (% GDP)', 'Fiscal'),
]
theme_palette = {
    'Resource exposure':     '#E63946',
    'Investment & labour':   '#2A9D8F',
    'Sectoral composition':  '#457B9D',
    'Macro & financial':     '#F4A261',
    'Institutions':          '#9d4edd',
    'Fiscal':                '#6c757d',
}
tree_df = pd.DataFrame(VARIABLES, columns=['label', 'theme'])
fig_tm = px.treemap(
    tree_df, path=['theme', 'label'], values=[1]*len(tree_df),
    color='theme', color_discrete_map=theme_palette,
)
fig_tm.update_traces(
    hovertemplate='<b>%{label}</b><br>%{parent}<extra></extra>',
    marker=dict(line=dict(color='white', width=1)),
    textfont=dict(family=FONT, size=12),
)
fig_tm.update_layout(
    height=540, margin=dict(l=10, r=10, t=20, b=10),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    showlegend=False,
)
fig_tm.write_html(str(OUT / 'variables_treemap.html'), include_plotlyjs='cdn', full_html=True)
print(f'Wrote {OUT / "variables_treemap.html"}')

# 5) Data sources bubble
SOURCES = [
    ('UN Comtrade', 'Trade flows (HS sub-codes, 1995-2024)', 10),
    ('Atlas of Economic Complexity', 'ECI, product complexity', 1),
    ('Penn World Tables 11.0', 'Human capital index, GFCF, population, GDP per capita', 4),
    ('IMF WEO (2025 release)', 'Government revenue, fiscal balance', 2),
    ('World Bank WDI', 'Manufacturing, services, agriculture, industry, trade openness, urban pop, credit, inflation', 8),
    ('V-Dem', 'Rule of law, political stability, corruption', 3),
]
src_df = pd.DataFrame(SOURCES, columns=['source', 'detail', 'feature_count'])
# arrange as a bubble layout in a horizontal strip
fig_b = go.Figure()
xs = list(range(len(src_df)))
for i, r in src_df.iterrows():
    fig_b.add_trace(go.Scatter(
        x=[xs[i]], y=[0],
        mode='markers+text',
        marker=dict(size=20 + r['feature_count']*8, color='#457B9D',
                    line=dict(color='white', width=1.5), opacity=0.85),
        text=[r['source']], textposition='bottom center',
        textfont=dict(family=FONT, size=11, color=NAVY),
        hovertemplate=f'<b>{r["source"]}</b><br>{r["detail"]}<br>{r["feature_count"]} feature(s)<extra></extra>',
        showlegend=False,
    ))
fig_b.update_layout(
    height=320, margin=dict(l=30, r=30, t=20, b=40),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    xaxis=dict(visible=False, range=[-0.7, len(src_df)-0.3]),
    yaxis=dict(visible=False, range=[-1.2, 0.8]),
    showlegend=False,
)
fig_b.write_html(str(OUT / 'data_sources_bubble.html'), include_plotlyjs='cdn', full_html=True)
print(f'Wrote {OUT / "data_sources_bubble.html"}')
