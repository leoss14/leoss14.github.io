#!/usr/bin/env python3
"""
generate_charts_supplementary.py

Companion to generate_charts_5.py. Produces the additional charts the portfolio
page references that V5 does not generate:

  22_production_intensity_map.html     world map, production value per capita
  25_resource_portfolio_decomposition.html
                                       stacked bar, top 20 most diversified
  26_pca_scatter_clusters.html         PC1 vs PC2 scatter, coloured by cluster
  27_pca_loadings_pc1.html             PC1 loadings horizontal bar
  28_pca_loadings_pc2.html             PC2 loadings horizontal bar

Reads the same intermediary/ files as V5 (Master.csv, clusters1995.csv).
Writes to: outputs/ next to page.html (auto-discovered).

Run:
  /usr/local/bin/python3.10 generate_charts_supplementary.py

Design system (font, colours, layout) matches V5 exactly so charts feel
unified across the page.
"""

import os, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve intermediary/ and outputs/ regardless of whether this script sits
# next to V5 (in code/v2/) or one level up (in code/).
HERE  = os.path.dirname(os.path.abspath(__file__))

def _find_dir(name, must_contain=None):
    """Look for a folder called `name` in likely locations."""
    candidates = [
        os.path.join(HERE, name),
        os.path.join(HERE, "v2", name),
        os.path.join(HERE, "..", "v2", name),
        os.path.join(HERE, "..", name),
    ]
    for c in candidates:
        if os.path.isdir(c):
            if must_contain is None or os.path.exists(os.path.join(c, must_contain)):
                return os.path.abspath(c)
    return None

V1 = _find_dir("intermediary", must_contain="Master.csv")
if V1 is None:
    raise FileNotFoundError(
        "Could not locate intermediary/Master.csv. Looked in: "
        + str([os.path.join(HERE, "intermediary"),
               os.path.join(HERE, "v2", "intermediary"),
               os.path.join(HERE, "..", "v2", "intermediary"),
               os.path.join(HERE, "..", "intermediary")])
        + ". Place this script next to generate_charts_5.py inside code/v2/, "
          "or set V1 manually at the top of this file."
    )

# Output dir: write directly to projects/capstone/outputs/ where the page reads.
# Walk up from intermediary/ until we find a folder containing page.html, or fall
# back to the parent of code/.
def _find_page_root(start):
    cur = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(cur, "page.html")):
            return cur
        cur = os.path.dirname(cur)
    return None

_page_root = _find_page_root(V1)
if _page_root:
    OUT = os.path.join(_page_root, "outputs")
else:
    # fall back: assume V1 is .../code/v2/intermediary, so capstone is two up from V1
    OUT = os.path.abspath(os.path.join(V1, "..", "..", "..", "outputs"))
os.makedirs(OUT, exist_ok=True)

print(f"  intermediary/: {V1}")
print(f"  outputs/:      {OUT}")

# ── Design system (mirrors V5) ────────────────────────────────────────────────
FONT  = "Public Sans, system-ui, -apple-system, sans-serif"
NAVY  = "#1a2744"
ACC   = "#c23a3a"
SUBTT = "#6b7280"
BG    = "#fafafa"
GRID  = "#e5e7eb"
CFG   = dict(displayModeBar=False, displaylogo=False, responsive=True)

CLUSTER_COLORS = {
    "Petrostates":        "#E63946",
    "Oil Exporters":      "#457B9D",
    "Major Producers":    "#2A9D8F",
    "Mining Exporters":   "#E9C46A",
    "Forestry Intensive": "#8B5CF6",
}

# Resource categories used for diversity / decomposition (matches V5 rent_cols)
RENT_COLS = [
    ("Oil rents (% of GDP)",         "Oil",      "#E63946"),
    ("Natural gas rents (% of GDP)", "Gas",      "#F4A261"),
    ("Mineral rents (% of GDP)",     "Minerals", "#2A9D8F"),
    ("Forestry rents (% of GDP)",    "Forestry", "#8B5CF6"),
]
# Optional: include coal if column exists
COAL_COL = "Coal rents (% of GDP)"

def base_layout(**kw):
    d = dict(
        template="plotly_white",
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family=FONT, size=12, color=NAVY),
        margin=dict(l=60, r=40, t=70, b=50),
        height=560,
    )
    d.update(kw)
    return d

def title(text, sub=None):
    return dict(
        text=text if sub is None else
             f"{text}<br><sup style='font-size:11px;font-weight:normal;color:{SUBTT}'>{sub}</sup>",
        x=0.5, xanchor="center",
        font=dict(size=16, color=NAVY, family=FONT),
    )

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.write_html(path, config=CFG, include_plotlyjs="cdn")
    print(f"  → {name}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading capstone data ===")
master = pd.read_csv(os.path.join(V1, "Master.csv"))
cl1995 = pd.read_csv(os.path.join(V1, "clusters1995.csv"))
print(f"  Master.csv: {len(master):,} rows, {master['Country Code'].nunique()} countries")

# Identify which rent columns actually exist (coal is optional)
present_rents = [c for c in [r[0] for r in RENT_COLS] if c in master.columns]
if COAL_COL in master.columns:
    present_rents.append(COAL_COL)
print(f"  Rent columns found: {present_rents}")

# Per-capita production
master = master.copy()
master["Prod_pc"] = (master["Total_Production_Value"]
                    / master["Population"].replace(0, np.nan))

# ──────────────────────────────────────────────────────────────────────────────
# 22. Production intensity map (production value per capita, latest year)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 22. Production intensity map ===")

LATEST = int(master["Year"].max())
m22 = (master[master["Year"] == LATEST]
       .dropna(subset=["Prod_pc", "Country Code"])
       .copy())

# Add per-GDP intensity to the hover
m22["Intensity_pct"] = (
    100 * m22["Total_Production_Value"]
    / (m22["GDP per capita (constant prices, PPP)"] * m22["Population"])
).replace([np.inf, -np.inf], np.nan)

# log scale handles the long tail (few countries dominate global production)
m22["log_prod_pc"] = np.log10(m22["Prod_pc"].clip(lower=1))

fig22 = go.Figure(go.Choropleth(
    locations=m22["Country Code"],
    z=m22["log_prod_pc"],
    text=m22["Country Name"] if "Country Name" in m22.columns else m22["Country Code"],
    customdata=m22[["Prod_pc", "Total_Production_Value", "Intensity_pct",
                    "GDP per capita (constant prices, PPP)"]].values,
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Production per capita: $%{customdata[0]:,.0f}<br>"
        "Total production value: $%{customdata[1]:,.0f}<br>"
        "Intensity: %{customdata[2]:.1f}% of GDP<br>"
        "GDP per capita: $%{customdata[3]:,.0f}<extra></extra>"
    ),
    colorscale=[[0, "#f1f5f9"], [0.4, "#7da7d3"], [0.7, "#3a6fa5"], [1, NAVY]],
    colorbar=dict(
        title=dict(text="USD per capita<br><sup>(log scale)</sup>",
                   font=dict(size=11)),
        tickvals=[2, 3, 4, 5],
        ticktext=["$100", "$1k", "$10k", "$100k"],
        thickness=14, len=0.7,
    ),
    marker=dict(line=dict(color="white", width=0.5)),
))
fig22.update_layout(
    **base_layout(height=580),
    title=title("Resource Production Intensity",
                f"USD per capita · {LATEST}"),
    geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth",
             bgcolor=BG, lakecolor=BG, landcolor="#f1f5f9"),
)
save(fig22, "22_production_intensity_map.html")

# Note: charts 23 (rents intensity) and 24 (diversity world map) intentionally
# omitted. The page references neither. We still compute Shannon entropy for
# chart 25's "top 20 most diversified" ranking.

# ──────────────────────────────────────────────────────────────────────────────
# Compute Shannon-entropy diversity index (used to rank countries for chart 25)
# ──────────────────────────────────────────────────────────────────────────────
div_cols = list(present_rents)  # all available rent columns
m24 = master[master["Year"] == LATEST].copy()

def shannon(row):
    vals = np.array([row[c] for c in div_cols], dtype=float)
    vals = np.clip(vals, 0, None)
    s = vals.sum()
    if s <= 0:
        return np.nan
    p = vals / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

m24["Diversity"] = m24.apply(shannon, axis=1)
m24 = m24.dropna(subset=["Diversity", "Country Code"])

# ──────────────────────────────────────────────────────────────────────────────
# 25. Resource portfolio decomposition (top 20 most diversified)
#
# Stacked horizontal bars of rent shares for the 20 countries scoring highest
# on diversity. Lets the reader see which resources drive each portfolio.
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 25. Resource portfolio decomposition ===")

top20 = m24.nlargest(20, "Diversity").copy()

# Compute share of each rent type within total
short = {c: c.replace(" rents (% of GDP)", "") for c in div_cols}
totals = top20[div_cols].sum(axis=1).replace(0, np.nan)
shares = top20[div_cols].div(totals, axis=0).fillna(0) * 100

# Order countries by total rents (descending) so largest portfolios go on top
top20 = top20.assign(Total_rents=top20[div_cols].sum(axis=1))
order = top20.sort_values("Total_rents", ascending=True).index
country_order = top20.loc[order, "Country Name" if "Country Name" in top20.columns else "Country Code"]

# Color map
RENT_COLOR = {
    "Oil rents (% of GDP)":         "#E63946",
    "Natural gas rents (% of GDP)": "#F4A261",
    "Mineral rents (% of GDP)":     "#2A9D8F",
    "Forestry rents (% of GDP)":    "#8B5CF6",
    "Coal rents (% of GDP)":        "#6c757d",
}

fig25 = go.Figure()
for col in div_cols:
    fig25.add_trace(go.Bar(
        y=country_order,
        x=shares.loc[order, col],
        name=short[col],
        orientation="h",
        marker=dict(color=RENT_COLOR.get(col, NAVY),
                    line=dict(color="white", width=0.5)),
        hovertemplate=f"<b>%{{y}}</b><br>{short[col]}: %{{x:.1f}}%<extra></extra>",
    ))
fig25.update_layout(
    **base_layout(height=720, margin=dict(l=160, r=40, t=70, b=50)),
    barmode="stack",
    title=title("Resource Portfolio Composition",
                "Top 20 most diversified countries"),
    xaxis=dict(title="% of total rents", gridcolor=GRID, range=[0, 100], ticksuffix="%"),
    yaxis=dict(title="", gridcolor=GRID),
    legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
)
save(fig25, "25_resource_portfolio_decomposition.html")

# ──────────────────────────────────────────────────────────────────────────────
# 26-28. PCA on the clustering variables
#
# Variables: per-capita production, ECI, log GDP per capita, HCI, electricity
# access, and rent shares. Reproduces the 6-variable PCA the methodology box
# describes (PC1 ~ Economic Development, PC2 ~ Hydrocarbon Intensity).
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== 26-28. PCA on cluster variables ===")

pca_year = LATEST
pca_vars = []
for col in [
    "Economic Complexity Index",
    "Human capital index",
    "Oil rents (% of GDP)",
    "Natural gas rents (% of GDP)",
    "Mineral rents (% of GDP)",
    "Forestry rents (% of GDP)",
]:
    if col in master.columns:
        pca_vars.append(col)

pca_short = {
    "Economic Complexity Index":     "ECI",
    "Human capital index":           "Human Capital",
    "Oil rents (% of GDP)":          "Oil Rents",
    "Natural gas rents (% of GDP)":  "Gas Rents",
    "Mineral rents (% of GDP)":      "Mineral Rents",
    "Forestry rents (% of GDP)":     "Forestry Rents",
}

pdf = (master[master["Year"] == pca_year]
       .dropna(subset=pca_vars + ["Country Code"])
       .copy())

# Merge cluster labels
cl_map = cl1995[["Country Code", "ClusterLabels"]].drop_duplicates("Country Code")
pdf = pdf.merge(cl_map, on="Country Code", how="left")

X = pdf[pca_vars].values
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
scores = pca.fit_transform(Xs)
pdf["PC1"] = scores[:, 0]
pdf["PC2"] = scores[:, 1]

evr = pca.explained_variance_ratio_  # share of variance explained
loadings = pd.DataFrame(pca.components_.T,
                        index=[pca_short[v] for v in pca_vars],
                        columns=["PC1", "PC2"])
print(f"  PC1: {evr[0]*100:.1f}%  PC2: {evr[1]*100:.1f}%  total: {evr.sum()*100:.1f}%")

# 26. PCA scatter coloured by cluster
fig26 = go.Figure()
for lbl, color in CLUSTER_COLORS.items():
    sub = pdf[pdf["ClusterLabels"] == lbl]
    if sub.empty:
        continue
    fig26.add_trace(go.Scatter(
        x=sub["PC1"], y=sub["PC2"],
        mode="markers",
        name=lbl,
        marker=dict(size=10, color=color, line=dict(color="white", width=1.2),
                    opacity=0.85),
        text=sub["Country Name"] if "Country Name" in sub.columns else sub["Country Code"],
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
    ))
# Unclustered countries (rare, but possible)
unc = pdf[pdf["ClusterLabels"].isna()]
if not unc.empty:
    fig26.add_trace(go.Scatter(
        x=unc["PC1"], y=unc["PC2"], mode="markers", name="Unclassified",
        marker=dict(size=8, color="#cccccc", line=dict(color="white", width=1)),
        text=unc["Country Name"] if "Country Name" in unc.columns else unc["Country Code"],
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
    ))

fig26.update_layout(
    **base_layout(height=600),
    title=title("PCA Projection of Resource Profiles",
                f"PC1 + PC2 = {evr.sum()*100:.0f}% of variance · {pca_year}"),
    xaxis=dict(title=f"PC1 — Economic Development ({evr[0]*100:.1f}%)",
               gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1),
    yaxis=dict(title=f"PC2 — Hydrocarbon Intensity ({evr[1]*100:.1f}%)",
               gridcolor=GRID, zerolinecolor=GRID, zerolinewidth=1),
    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
)
save(fig26, "26_pca_scatter_clusters.html")

# 27. PC1 loadings
def loadings_bar(comp, name, sub_text):
    sorted_idx = loadings[comp].abs().sort_values(ascending=True).index
    vals = loadings.loc[sorted_idx, comp].values
    colors = [NAVY if v >= 0 else ACC for v in vals]
    fig = go.Figure(go.Bar(
        y=sorted_idx, x=vals,
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=NAVY),
        hovertemplate="<b>%{y}</b><br>Loading: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(height=440, margin=dict(l=140, r=60, t=70, b=50)),
        title=title(f"{name} Loadings", sub_text),
        xaxis=dict(title="Loading", gridcolor=GRID, zeroline=True,
                   zerolinecolor=NAVY, zerolinewidth=1.2,
                   range=[-1, 1]),
        yaxis=dict(title="", gridcolor=GRID),
        showlegend=False,
    )
    return fig

fig27 = loadings_bar(
    "PC1", "PC1: Economic Development",
    f"{evr[0]*100:.0f}% of variance"
)
save(fig27, "27_pca_loadings_pc1.html")

fig28 = loadings_bar(
    "PC2", "PC2: Hydrocarbon Intensity",
    f"{evr[1]*100:.0f}% of variance"
)
save(fig28, "28_pca_loadings_pc2.html")

print("\n=== Done. 5 supplementary charts written to outputs/ ===")
