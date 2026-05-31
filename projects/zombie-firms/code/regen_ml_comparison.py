#!/usr/bin/env python3
"""Rebuild ml_model_comparison.html to show with-ICR vs without-ICR side by side."""
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

PALETTE = {"navy":"#1f2a44","slate":"#3b4a6b","steel":"#6b7d9e","rose":"#b85c5c",
           "gold":"#c9a45c","sage":"#6b8e6b","grey":"#9aa3b2","ink":"#0f1626"}
TEMPLATE = dict(layout=dict(
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=PALETTE["ink"]),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=True, gridcolor="#e8eaf0", zeroline=False, ticks="outside"),
    yaxis=dict(showgrid=True, gridcolor="#e8eaf0", zeroline=False, ticks="outside"),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dadde6", borderwidth=1),
))
pio.templates["zombie"] = go.layout.Template(TEMPLATE)

OUT = Path("/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/zombie-firms/outputs")

# Load both specs
sc = pd.read_csv(OUT / "scratch" / "ml_icr_comparison.csv")
sc["spec_short"] = sc["spec"].map({
    "Without ICR (leakage-free)": "Without ICR<br>(leakage-free)",
    "With ICR lags (leaky baseline)": "With ICR<br>(leaky)",
})
models = ["Logistic", "Random Forest", "XGBoost"]
naive = 0.038

fig = make_subplots(rows=1, cols=2,
    subplot_titles=("Test AUPRC", "Test ROC-AUC"),
    horizontal_spacing=0.14)

# Grouped bars: model on x, spec as group
for col, metric in [(1, "test_auprc"), (2, "test_roc")]:
    for spec, color in [("Without ICR (leakage-free)", PALETTE["navy"]),
                         ("With ICR lags (leaky baseline)", PALETTE["rose"])]:
        d = sc[sc["spec"] == spec].set_index("model").reindex(models).reset_index()
        spec_label = "Without ICR (legitimate)" if "Without" in spec else "With ICR (leaky)"
        fig.add_trace(go.Bar(
            x=d["model"], y=d[metric], name=spec_label,
            marker_color=color,
            text=d[metric].round(3),
            textposition="outside",
            textfont=dict(size=11, color=PALETTE["ink"]),
            hovertemplate=f"<b>%{{x}}</b><br>{spec_label}<br>{metric}: %{{y:.3f}}<extra></extra>",
            showlegend=(col == 1),
        ), row=1, col=col)
    # Reference lines
    if metric == "test_auprc":
        fig.add_hline(y=naive, row=1, col=col, line=dict(color=PALETTE["grey"], width=1, dash="dot"),
                      annotation_text=f"naive baseline ({naive:.3f})", annotation_position="right",
                      annotation_font_size=10, annotation_font_color=PALETTE["grey"])
    else:
        fig.add_hline(y=0.5, row=1, col=col, line=dict(color=PALETTE["grey"], width=1, dash="dot"),
                      annotation_text="random (0.5)", annotation_position="right",
                      annotation_font_size=10, annotation_font_color=PALETTE["grey"])

fig.update_xaxes(tickfont=dict(size=11))
fig.update_yaxes(range=[0, 1], row=1, col=2)
fig.update_layout(
    barmode="group", template="zombie",
    height=480,
    margin=dict(l=60, r=30, t=80, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
)
fig.write_html(OUT / "ml_model_comparison.html", include_plotlyjs="cdn", full_html=True,
               config={"displayModeBar": False, "responsive": True})
print(f"Saved {OUT/'ml_model_comparison.html'}")
