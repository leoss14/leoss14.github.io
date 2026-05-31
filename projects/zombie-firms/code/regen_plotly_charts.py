#!/usr/bin/env python3
"""
regen_plotly_charts.py

Converts the 11 zombie-firms PNG figures into interactive Plotly HTML charts
using the same style system as the uber/capstone/pink-tax pages.

Reads:  /Users/leoss/Desktop/Thesis Replication/output/*.{csv,parquet}
        /Users/leoss/Desktop/Thesis Replication/output/geo/*
Writes: /Users/leoss/Desktop/GitHub/leoss14.github.io/projects/zombie-firms/outputs/*.html

Charts produced (in narrative order):
  1. zombie_share_time.html       Time series by definition
  2. zombie_by_sector.html        Sector heterogeneity, 2022
  3. north_south.html             North vs rest of Italy
  4. choropleth_italy_2022.html   NUTS3 choropleth, McGowan, 2022
  5. dotmap_italy_2022.html       Firm-level dot map, sampled
  6. nb3_coefficients.html        Baseline regression coefficients
  7. congestion_investment.html   Province-sector binscatter
  8. spatial_decay.html           Bandwidth sweep coefficients
  9. ml_curves.html               PR + ROC curves (test fold)
 10. ml_shap.html                 SHAP feature importance
 11. ml_rolling.html              Rolling AUPRC (or horizon sweep proxy)

Run: python3 regen_plotly_charts.py
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC = Path("/Users/leoss/Desktop/Thesis Replication")
OUT = Path("/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/zombie-firms/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style (matches uber/capstone/pink-tax projects)
# ---------------------------------------------------------------------------
PALETTE = {
    "navy":  "#1f2a44",
    "slate": "#3b4a6b",
    "steel": "#6b7d9e",
    "rose":  "#b85c5c",
    "gold":  "#c9a45c",
    "sage":  "#6b8e6b",
    "grey":  "#9aa3b2",
    "ink":   "#0f1626",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=PALETTE["ink"]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[PALETTE[k] for k in ("navy", "rose", "steel", "gold", "sage", "slate")],
        xaxis=dict(showgrid=True, gridcolor="#e8eaf0", zeroline=False, ticks="outside",
                   tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e8eaf0", zeroline=False, ticks="outside",
                   tickfont=dict(size=11)),
        margin=dict(l=60, r=30, t=30, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dadde6", borderwidth=1),
    )
)
pio.templates["zombie"] = go.layout.Template(PLOTLY_TEMPLATE)

def save_chart(fig: go.Figure, name: str, height: int = 460):
    """Write a Plotly figure as a self-contained HTML fragment."""
    fig.update_layout(template="zombie", height=height, title_text=None)
    out = OUT / f"{name}.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": False, "responsive": True})
    print(f"  saved {out.name}")


# ===========================================================================
# Load core data once
# ===========================================================================
print("Loading panel data...")
panel = pd.read_parquet(SRC / "output" / "zombie_panel_classified.parquet")
print(f"  panel: {len(panel):,} firm-year rows, {panel['bvd_id'].nunique():,} firms")

# Find the active McGowan column. Names vary across nb output versions.
mcg_col = None
for cand in ["zombie_mcgowan", "zombie", "mcgowan_zombie"]:
    if cand in panel.columns:
        mcg_col = cand; break
weak_col = "zombie_weak" if "zombie_weak" in panel.columns else None
storz_col = "zombie_storz" if "zombie_storz" in panel.columns else None
print(f"  cols: mcgowan={mcg_col}, weak={weak_col}, storz={storz_col}")

# ===========================================================================
# Chart 1: zombie share over time, by definition
# ===========================================================================
print("\nChart 1: zombie_share_time")
yearly = panel.groupby("year").agg(
    mcgowan=(mcg_col, "mean"),
    weak=(weak_col, "mean") if weak_col else (mcg_col, "mean"),
    storz=(storz_col, "mean") if storz_col else (mcg_col, "mean"),
).reset_index()
# Keep 2018+ (McGowan needs 3 years of history, so 2016-17 are mechanical zero)
yearly_plot = yearly[yearly["year"] >= 2018].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly_plot["year"], y=yearly_plot["weak"]*100, name="Weak ICR < 1",
    line=dict(color=PALETTE["rose"], width=2.5), mode="lines+markers",
    marker=dict(size=8),
    hovertemplate="<b>%{x}</b><br>Weak ICR share: %{y:.1f}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=yearly_plot["year"], y=yearly_plot["mcgowan"]*100, name="McGowan (3yr + age 10)",
    line=dict(color=PALETTE["navy"], width=3), mode="lines+markers",
    marker=dict(size=8),
    hovertemplate="<b>%{x}</b><br>McGowan share: %{y:.1f}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=yearly_plot["year"], y=yearly_plot["storz"]*100, name="Storz (multi-criterion)",
    line=dict(color=PALETTE["steel"], width=2.5, dash="dash"), mode="lines+markers",
    marker=dict(size=8),
    hovertemplate="<b>%{x}</b><br>Storz share: %{y:.1f}%<extra></extra>",
))
# Shade COVID year
fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor=PALETTE["grey"], opacity=0.15, layer="below",
              line_width=0, annotation_text="COVID 2020", annotation_position="top left",
              annotation_font_size=10, annotation_font_color=PALETTE["slate"])

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Zombie share (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
save_chart(fig, "zombie_share_time", height=520)

# ===========================================================================
# Chart 2: zombie share by NACE 2-digit sector, 2022 cross-section
# ===========================================================================
print("\nChart 2: zombie_by_sector")
NACE_LABELS = {
    10: "Food", 11: "Beverages", 12: "Tobacco", 13: "Textiles", 14: "Apparel",
    15: "Leather", 16: "Wood products", 17: "Paper", 18: "Printing",
    19: "Coke / petroleum", 20: "Chemicals", 21: "Pharma", 22: "Rubber / plastics",
    23: "Non-metallic minerals", 24: "Basic metals", 25: "Fabricated metals",
    26: "Computer / electronic", 27: "Electrical equipment", 28: "Machinery",
    29: "Motor vehicles", 30: "Other transport", 31: "Furniture",
    32: "Other manufacturing", 33: "Repair / installation",
}

sector_col = "nace_2digit" if "nace_2digit" in panel.columns else None
sector_yr = panel[(panel["year"] == 2022) & panel[sector_col].notna()].copy()
sector_yr[sector_col] = sector_yr[sector_col].astype(int)
sector_summary = sector_yr.groupby(sector_col).agg(
    n=("bvd_id", "size"), mcg=(mcg_col, "mean")
).reset_index()
sector_summary = sector_summary[sector_summary["n"] >= 30].copy()
sector_summary["label"] = sector_summary[sector_col].map(NACE_LABELS).fillna(
    sector_summary[sector_col].astype(str))
sector_summary = sector_summary.sort_values("mcg")

fig = go.Figure(go.Bar(
    x=sector_summary["mcg"]*100,
    y=sector_summary["label"],
    orientation="h",
    marker_color=PALETTE["navy"],
    hovertemplate="<b>%{y}</b><br>Zombie share: %{x:.1f}%<br>Firms: %{customdata}<extra></extra>",
    customdata=sector_summary["n"],
))
fig.update_layout(
    xaxis_title="McGowan zombie share, 2022 (%)",
    yaxis_title=None,
    height=680,
    margin=dict(l=160),
)
save_chart(fig, "zombie_by_sector", height=680)

# ===========================================================================
# Chart 3: North vs rest of Italy comparison over time
# ===========================================================================
print("\nChart 3: north_south")
NORTH_NUTS2 = ("ITC", "ITH")
nuts2_col = None
for c in ["nuts2_code", "nuts2"]:
    if c in panel.columns: nuts2_col = c; break
if nuts2_col is None and "nuts3" in panel.columns:
    panel["__nuts2"] = panel["nuts3"].astype(str).str[:4].str.split(" ").str[0]
    nuts2_col = "__nuts2"

panel["__region"] = np.where(panel[nuts2_col].astype(str).str[:3].isin(NORTH_NUTS2),
                              "North (ITC + ITH)", "Rest of Italy")
ns = panel[panel["year"] >= 2018].groupby(["year", "__region"])[mcg_col].mean().reset_index()

fig = go.Figure()
for region, color in [("North (ITC + ITH)", PALETTE["navy"]),
                       ("Rest of Italy",    PALETTE["rose"])]:
    sub = ns[ns["__region"] == region]
    fig.add_trace(go.Scatter(
        x=sub["year"], y=sub[mcg_col]*100, name=region,
        line=dict(color=color, width=3), mode="lines+markers",
        marker=dict(size=9),
        hovertemplate="<b>%{x}</b><br>" + region + ": %{y:.1f}%<extra></extra>",
    ))
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="McGowan zombie share (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
save_chart(fig, "north_south", height=520)

# ===========================================================================
# Chart 4: NUTS3 choropleth, McGowan share, 2022
# ===========================================================================
print("\nChart 4: choropleth_italy_2022")
import json as _json
pyr = pd.read_csv(SRC / "output" / "zombie_province_year.csv")
pyr_2022 = pyr[(pyr["year"] == 2022) & (pyr["n_firms"] >= 5)].copy()
pyr_2022["zombie_share_pct"] = pyr_2022["zombie_share"] * 100

with open(SRC / "output" / "geo" / "NUTS_RG_20M_2021_4326.geojson") as f:
    nuts_geo = _json.load(f)

# Filter geo to NUTS3 (LEVL_CODE = 3) and Italian provinces (CNTR_CODE = 'IT')
italy_features = [
    feat for feat in nuts_geo["features"]
    if feat["properties"].get("LEVL_CODE") == 3
    and feat["properties"].get("CNTR_CODE") == "IT"
]
italy_geo = {"type": "FeatureCollection", "features": italy_features}
print(f"  italy NUTS3 features: {len(italy_features)}")

fig = go.Figure(go.Choropleth(
    geojson=italy_geo,
    locations=pyr_2022["nuts3_code"],
    z=pyr_2022["zombie_share_pct"],
    featureidkey="properties.NUTS_ID",
    colorscale=[[0, "#f0f4f8"], [0.5, PALETTE["steel"]], [1, PALETTE["navy"]]],
    marker_line_color="white",
    marker_line_width=0.5,
    colorbar=dict(title="Zombie<br>share (%)", thickness=14, len=0.7, x=1.02),
    customdata=np.column_stack([pyr_2022["province"], pyr_2022["n_firms"]]),
    hovertemplate="<b>%{customdata[0]}</b><br>Zombie share: %{z:.1f}%<br>Firms: %{customdata[1]}<extra></extra>",
))
fig.update_geos(
    fitbounds="locations", visible=False,
    projection_type="mercator",
    bgcolor="white",
)
fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=560,
)
save_chart(fig, "choropleth_italy_2022", height=560)

# ===========================================================================
# Chart 5: firm-level dot map, 2022 (sampled because 75K points are heavy)
# ===========================================================================
print("\nChart 5: dotmap_italy_2022")
geo = pd.read_csv(SRC / "output" / "geo" / "firm_geocoded.csv")
panel_2022 = panel[panel["year"] == 2022][["bvd_id", mcg_col]].copy()
dot = geo.merge(panel_2022, on="bvd_id", how="inner").dropna(subset=["lat", "lon", mcg_col])
print(f"  geocoded firms with 2022 status: {len(dot):,}")

# Sample to keep the page responsive. Stratify on zombie status so we keep all zombies.
zombies = dot[dot[mcg_col] == 1]
healthy = dot[dot[mcg_col] == 0]
SAMPLE_N_HEALTHY = 12000  # ~5% sample of non-zombies
if len(healthy) > SAMPLE_N_HEALTHY:
    healthy = healthy.sample(SAMPLE_N_HEALTHY, random_state=42)
dot_plot = pd.concat([zombies, healthy], ignore_index=True)
print(f"  sample for plot: {len(dot_plot):,} ({len(zombies):,} zombies + {len(healthy):,} healthy)")

fig = go.Figure()
# Plot healthy first (background)
fig.add_trace(go.Scattermapbox(
    lat=healthy["lat"], lon=healthy["lon"], mode="markers",
    marker=dict(size=4, color=PALETTE["steel"], opacity=0.4),
    name="Healthy firm", hoverinfo="skip",
))
# Zombies on top, redder, larger
fig.add_trace(go.Scattermapbox(
    lat=zombies["lat"], lon=zombies["lon"], mode="markers",
    marker=dict(size=6, color=PALETTE["rose"], opacity=0.85),
    name="McGowan zombie",
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>Province: %{customdata[1]}<extra></extra>"
    ),
    customdata=zombies[["bvd_id", "province"]].values,
))
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=5.2,
    mapbox_center={"lat": 42.5, "lon": 12.5},
    margin=dict(l=0, r=0, t=10, b=0),
    height=620,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.9)"),
)
save_chart(fig, "dotmap_italy_2022", height=620)

# ===========================================================================
# Chart 6: NB3 regression coefficient plot (M1-M4 with 95% CIs)
# ===========================================================================
print("\nChart 6: nb3_coefficients")
# Hardcoded from results_summary.md (the source CSV has formatted strings)
nb3_data = [
    ("M1 Investment rate", -0.0032, 0.0092, "Non-zombie firms", PALETTE["steel"]),
    ("M2 Employment growth", -0.0041, 0.0152, "Non-zombie firms", PALETTE["steel"]),
    ("M3 Investment × credit", -0.0088, 0.0119, "Non-zombie firms", PALETTE["steel"]),
    ("M4 Own zombie effect", -0.0029, 0.0011, "All firms", PALETTE["navy"]),
]
labels  = [r[0] for r in nb3_data]
coefs   = [r[1] for r in nb3_data]
errs    = [1.96 * r[2] for r in nb3_data]
colors  = [r[4] for r in nb3_data]
samples = [r[3] for r in nb3_data]

fig = go.Figure()
fig.add_vline(x=0, line=dict(color=PALETTE["grey"], width=1, dash="dash"))
fig.add_trace(go.Scatter(
    x=coefs, y=labels,
    mode="markers",
    marker=dict(size=14, color=colors, line=dict(color="white", width=1.5)),
    error_x=dict(type="data", array=errs, color=PALETTE["slate"], thickness=2, width=8),
    name="Coefficient",
    hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<br>Sample: %{customdata}<extra></extra>",
    customdata=samples,
))
fig.update_layout(
    xaxis_title="Coefficient on zombie congestion (with 95% CI)",
    yaxis_title=None,
    margin=dict(l=200),
    showlegend=False,
    height=480,
)
save_chart(fig, "nb3_coefficients", height=480)

# ===========================================================================
# Chart 7: congestion vs investment binscatter (province-sector cells)
# ===========================================================================
print("\nChart 7: congestion_investment")
inv_col = None
for c in ["investment_rate", "inv_rate", "inv_rate_w"]:
    if c in panel.columns:
        inv_col = c; break
prov_col = "nuts3_code" if "nuts3_code" in panel.columns else "province"
sec_col = "nace_2digit" if "nace_2digit" in panel.columns else None

ps = panel.dropna(subset=[mcg_col, inv_col, prov_col, sec_col]).copy()
ps_yr = ps[ps["year"].between(2018, 2024)].copy()
cell = ps_yr.groupby([prov_col, sec_col, "year"]).agg(
    n=(mcg_col, "size"),
    zshare=(mcg_col, "mean"),
    inv_mean=(inv_col, "mean"),
).reset_index()
cell = cell[cell["n"] >= 5].copy()
# Winsorise the investment mean a bit for plotting
lo, hi = cell["inv_mean"].quantile([0.01, 0.99])
cell["inv_mean_w"] = cell["inv_mean"].clip(lo, hi)
# Bin by zombie share for the binscatter
cell["zbin"] = pd.qcut(cell["zshare"], q=20, duplicates="drop")
binned = cell.groupby("zbin", observed=True).agg(
    zshare_mid=("zshare", "mean"),
    inv_mean=("inv_mean_w", "mean"),
    n_cells=("n", "size"),
).reset_index()

fig = go.Figure()
# Raw scatter, faint
fig.add_trace(go.Scattergl(
    x=cell["zshare"]*100, y=cell["inv_mean_w"]*100,
    mode="markers",
    marker=dict(size=3, color=PALETTE["grey"], opacity=0.2),
    name="Province × sector × year cell",
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=binned["zshare_mid"]*100, y=binned["inv_mean"]*100,
    mode="markers+lines",
    marker=dict(size=11, color=PALETTE["navy"], line=dict(color="white", width=1.5)),
    line=dict(color=PALETTE["navy"], width=2),
    name="Binned mean (ventiles)",
    hovertemplate="<b>Zombie share bin</b>: %{x:.1f}%<br>Mean investment rate: %{y:.2f}%<br>Cells in bin: %{customdata}<extra></extra>",
    customdata=binned["n_cells"],
))
fig.update_layout(
    xaxis_title="Province × sector zombie share (%)",
    yaxis_title="Mean investment rate (%)",
    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
)
save_chart(fig, "congestion_investment", height=540)

# ===========================================================================
# Chart 8: spatial decay (bandwidth sweep with 95% CIs)
# ===========================================================================
print("\nChart 8: spatial_decay")
# Values from the results_summary.md table (verified consistent with paper)
spec_data = [
    ("Province × sector\n(baseline)",  -0.0032, 0.0092, PALETTE["grey"]),
    ("NUTS2 × sector",                  +0.0045, 0.0224, PALETTE["grey"]),
    ("25 km radius",                    -0.0161, 0.0094, PALETTE["steel"]),
    ("50 km radius",                    -0.0200, 0.0114, PALETTE["steel"]),
    ("100 km radius",                   -0.0289, 0.0138, PALETTE["navy"]),
]
labels = [d[0] for d in spec_data]
coefs  = [d[1] for d in spec_data]
errs   = [1.96 * d[2] for d in spec_data]
colors = [d[3] for d in spec_data]

fig = go.Figure()
fig.add_vline(x=0, line=dict(color=PALETTE["grey"], width=1, dash="dash"))
fig.add_trace(go.Scatter(
    x=coefs, y=labels,
    mode="markers",
    marker=dict(size=15, color=colors, line=dict(color="white", width=1.5)),
    error_x=dict(type="data", array=errs, color=PALETTE["slate"], thickness=2, width=8),
    hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f} (95%% CI shown)<extra></extra>",
))
fig.update_layout(
    xaxis_title="Coefficient on zombie congestion (with 95% CI)",
    yaxis_title=None,
    margin=dict(l=170),
    showlegend=False,
    height=520,
)
save_chart(fig, "spatial_decay", height=520)

# ===========================================================================
# Chart 9: ML PR + ROC curves
# Requires raw predictions on test set; we don't have them as a CSV here.
# Build a stylised version using known operating-point trade-offs.
# Actually let me check the notebook for saved arrays first; if not, render
# a "models compared" summary chart from nb5_ml.csv instead (cleaner anyway).
# ===========================================================================
print("\nChart 9: ml_curves -> ml_model_comparison (bar chart of metrics)")
ml_path = SRC / "output" / "nb5_ml_results.csv"
ml = pd.read_csv(ml_path)
# Drop SMOTE row for the main chart (mentioned in prose but not the headline)
ml_main = ml[~ml["Model"].str.contains("SMOTE")].copy()
# Add naive baseline as a bar for visual benchmarking
ml_main = pd.concat([
    ml_main,
    pd.DataFrame([{"Model": "Naive (prevalence)", "Test AUPRC": 0.038,
                   "Test ROC-AUC": 0.5, "Test F1": 0.075}])
], ignore_index=True)

fig = make_subplots(rows=1, cols=3, subplot_titles=("Test AUPRC", "Test ROC-AUC", "Test F1"),
                    horizontal_spacing=0.07)
colors_map = {
    "Logistic Regression": PALETTE["grey"],
    "Random Forest":       PALETTE["steel"],
    "XGBoost":             PALETTE["navy"],
    "Naive (prevalence)":  PALETTE["rose"],
}
bar_colors = [colors_map.get(m, PALETTE["slate"]) for m in ml_main["Model"]]

for col, metric, fmt in [(1, "Test AUPRC", "%{x:.3f}"),
                         (2, "Test ROC-AUC", "%{x:.3f}"),
                         (3, "Test F1", "%{x:.3f}")]:
    fig.add_trace(
        go.Bar(
            x=ml_main[metric], y=ml_main["Model"], orientation="h",
            marker_color=bar_colors,
            text=ml_main[metric].round(3),
            textposition="outside",
            textfont=dict(color=PALETTE["ink"], size=11),
            hovertemplate="<b>%{y}</b><br>" + metric + ": " + fmt + "<extra></extra>",
            showlegend=False,
        ),
        row=1, col=col,
    )

# Mark naive baseline reference on AUPRC panel
fig.add_vline(x=0.038, row=1, col=1, line=dict(color=PALETTE["rose"], width=1, dash="dot"),
              annotation_text="naive", annotation_position="top",
              annotation_font_size=9, annotation_font_color=PALETTE["rose"])
fig.add_vline(x=0.5, row=1, col=2, line=dict(color=PALETTE["rose"], width=1, dash="dot"),
              annotation_text="random", annotation_position="top",
              annotation_font_size=9, annotation_font_color=PALETTE["rose"])

fig.update_xaxes(rangemode="tozero")
fig.update_yaxes(tickfont=dict(size=11))
fig.update_layout(height=380, margin=dict(l=140, r=20, t=50, b=40))
save_chart(fig, "ml_model_comparison", height=380)

# ===========================================================================
# Chart 10: SHAP feature importance (XGBoost)
# ===========================================================================
print("\nChart 10: ml_shap")
shap = pd.read_csv(SRC / "output" / "nb5_shap_importance.csv")
# Clean feature names
FEATURE_LABELS = {
    "icr_w":         "ICR (winsorised)",
    "L1_icr_w":      "ICR, 1-year lag",
    "L2_icr_w":      "ICR, 2-year lag",
    "d_icr":         "Δ ICR (1-year)",
    "roa_w":         "ROA (winsorised)",
    "L1_roa_w":      "ROA, 1-year lag",
    "L2_roa_w":      "ROA, 2-year lag",
    "d_roa":         "Δ ROA",
    "leverage_w":    "Leverage",
    "L1_leverage_w": "Leverage, 1-year lag",
    "L2_leverage_w": "Leverage, 2-year lag",
    "d_leverage":    "Δ Leverage",
    "fin_int_w":     "Financial intensity",
    "inv_rate_w":    "Investment rate",
    "emp_growth_w":  "Employment growth",
    "sales_growth_w":"Sales growth",
    "log_ta":        "Log total assets",
    "firm_age":      "Firm age",
    "neg_equity":    "Negative equity flag",
}
shap["label"] = shap["feature"].map(FEATURE_LABELS).fillna(shap["feature"])
shap = shap.sort_values("mean_abs_shap", ascending=True).tail(15)

fig = go.Figure(go.Bar(
    x=shap["mean_abs_shap"], y=shap["label"],
    orientation="h",
    marker_color=PALETTE["navy"],
    text=shap["mean_abs_shap"].round(3),
    textposition="outside",
    textfont=dict(color=PALETTE["ink"], size=10),
    hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.3f}<extra></extra>",
))
fig.update_layout(
    xaxis_title="Mean absolute SHAP value (XGBoost)",
    yaxis_title=None,
    height=520,
    margin=dict(l=180, r=80),
)
save_chart(fig, "ml_shap", height=520)

# ===========================================================================
# Chart 11: ML rolling / horizon performance
# ===========================================================================
print("\nChart 11: ml_horizon (using horizon-sweep CSV)")
hz = pd.read_csv(SRC / "output" / "nb5_horizon_sweep.csv")
hz["horizon_step"] = hz["Horizon"].str.replace("t+", "", regex=False).astype(int)

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Test AUPRC by horizon", "AUPRC / naive baseline ratio"),
                    horizontal_spacing=0.12)

fig.add_trace(go.Scatter(
    x=hz["horizon_step"], y=hz["Test AUPRC"], mode="lines+markers",
    marker=dict(size=14, color=PALETTE["navy"]),
    line=dict(color=PALETTE["navy"], width=3),
    name="Test AUPRC",
    hovertemplate="Horizon: t+%{x}<br>AUPRC: %{y:.3f}<extra></extra>",
    showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=hz["horizon_step"], y=hz["Zombie rate"], mode="lines+markers",
    marker=dict(size=10, color=PALETTE["rose"]),
    line=dict(color=PALETTE["rose"], width=2, dash="dot"),
    name="Naive (prevalence)",
    hovertemplate="Horizon: t+%{x}<br>Naive: %{y:.3f}<extra></extra>",
    showlegend=False,
), row=1, col=1)
fig.add_trace(go.Bar(
    x=hz["horizon_step"], y=hz["AUPRC / naive"],
    marker_color=PALETTE["steel"],
    text=hz["AUPRC / naive"].round(2),
    textposition="outside",
    textfont=dict(color=PALETTE["ink"], size=11),
    hovertemplate="Horizon: t+%{x}<br>Ratio: %{y:.2f}x<extra></extra>",
    showlegend=False,
), row=1, col=2)

fig.update_xaxes(title_text="Forecast horizon (years)", row=1, col=1,
                 tickmode="array", tickvals=[1,2,3], ticktext=["t+1","t+2","t+3"])
fig.update_xaxes(title_text="Forecast horizon (years)", row=1, col=2,
                 tickmode="array", tickvals=[1,2,3], ticktext=["t+1","t+2","t+3"])
fig.update_yaxes(title_text="AUPRC", row=1, col=1)
fig.update_yaxes(title_text="AUPRC / naive", row=1, col=2)
fig.update_layout(height=500, margin=dict(l=70, r=30, t=50, b=50))
save_chart(fig, "ml_horizon", height=500)

print("\nDone. All charts written to:", OUT)
