"""
build_margin_charts.py

Rebuilds the three Part 2 margin charts from existing CSV outputs.
No recomputation, no DuckDB. Reads:
  outputs/tables/trip_margin_monthly_percentiles.csv
  outputs/tables/trip_margin_by_length_year.csv
  outputs/tables/trip_margin_by_zone_class_monthly.csv

Writes:
  outputs/margin/trip_margin_fan_monthly.html       (median + revenue-weighted ONLY, no fan)
  outputs/margin/trip_margin_by_length_year.html    (4 buckets x 4 reference years)
  outputs/margin/trip_margin_by_zone_class_p50.html (unchanged content but smaller legend)
"""
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import save_chart, PALETTE

HERE = Path(__file__).parent
T = HERE.parent / "outputs" / "tables"
C = HERE.parent / "outputs" / "margin"

LEGEND_COMPACT = dict(
    orientation="h",
    yanchor="bottom", y=1.0, xanchor="right", x=1.0,
    font=dict(size=11),
    bgcolor="rgba(255,255,255,0.85)",
    bordercolor="rgba(0,0,0,0)",
    borderwidth=0,
)

REGIMES = [
    ("2019-02-01", "Min-pay rule"),
    ("2020-03-01", "Pandemic"),
    ("2025-01-05", "CBD fee"),
]


# ============================================================================
# Chart 1: monthly margin, just median + revenue-weighted (no fan, no IQR)
# ============================================================================
def build_monthly():
    df = pd.read_csv(T / "trip_margin_monthly_percentiles.csv", parse_dates=["month"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["month"], y=df["p50"], mode="lines",
        line=dict(color=PALETTE["navy"], width=2.2),
        name="Median trip",
        hovertemplate="%{x|%b %Y}<br>Median: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["month"], y=df["margin_revenue_weighted"], mode="lines",
        line=dict(color=PALETTE["gold"], width=2.2),
        name="Revenue-weighted",
        hovertemplate="%{x|%b %Y}<br>Rev-weighted: %{y:.1%}<extra></extra>",
    ))

    for x, lbl in REGIMES:
        fig.add_vline(x=x, line=dict(color=PALETTE["grey"], width=0.8, dash="dot"))
        fig.add_annotation(x=x, y=0.35, text=lbl, showarrow=False,
                           font=dict(size=9, color=PALETTE["slate"]),
                           xshift=0, yshift=0)

    fig.update_layout(
        xaxis_title="", yaxis_title="Operator margin proxy",
        yaxis=dict(tickformat=".0%", range=[-0.08, 0.40]),
        height=440,
        legend=LEGEND_COMPACT,
        margin=dict(t=70, b=40, l=60, r=20),
    )
    save_chart(fig, C / "trip_margin_fan_monthly.html")
    print("  wrote trip_margin_fan_monthly.html (2 lines, no fan)")


# ============================================================================
# Chart 2: 4 length buckets x 4 reference years
# ============================================================================
def build_by_length():
    df = pd.read_csv(T / "trip_margin_by_length_year.csv")
    # Aggregate the 5 buckets into 4: combine "0_under_1mi" and "1_1to3mi"
    # into one bucket "Under 3 miles"? Actually user said "4 buckets x 4 years"
    # so let's drop the smallest bucket and keep 4 distinct ones.
    # Simpler: combine the 5 existing buckets into 4 of: <2mi, 2-5mi, 5-10mi, 10+mi
    # by mapping the existing buckets:
    bucket_map = {
        "0_under_1mi":  "Under 3 miles",
        "1_1to3mi":     "Under 3 miles",
        "2_3to6mi":     "3 to 6 miles",
        "3_6to12mi":    "6 to 12 miles",
        "4_over_12mi":  "Over 12 miles",
    }
    df["bucket"] = df["length_bucket"].map(bucket_map)
    # Aggregate p50 weighted by n_trips within each (year, bucket)
    g = df.groupby(["year", "bucket"]).apply(
        lambda x: pd.Series({
            "p50": (x["p50"] * x["n_trips"]).sum() / x["n_trips"].sum(),
            "n_trips": x["n_trips"].sum(),
        })
    ).reset_index()

    YEARS = [2019, 2021, 2023, 2025]  # 4 reference years
    BUCKETS = ["Under 3 miles", "3 to 6 miles", "6 to 12 miles", "Over 12 miles"]
    COLORS = [PALETTE["navy"], PALETTE["sage"], PALETTE["gold"], PALETTE["rose"]]

    fig = go.Figure()
    for year, color in zip(YEARS, COLORS):
        sub = g[g["year"] == year].set_index("bucket").reindex(BUCKETS).reset_index()
        fig.add_trace(go.Bar(
            x=sub["bucket"], y=sub["p50"], name=str(year),
            marker_color=color,
            hovertemplate=str(year) + " %{x}<br>Median margin: %{y:.1%}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="",
        yaxis_title="Median per-trip margin",
        yaxis=dict(tickformat=".0%", range=[-0.20, 0.40]),
        height=430,
        legend=LEGEND_COMPACT,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    save_chart(fig, C / "trip_margin_by_length_year.html")
    print("  wrote trip_margin_by_length_year.html (4 buckets x 4 years)")


# ============================================================================
# Chart 3: revenue-weighted operator margin by hybrid CLUSTER, monthly (Uber).
# Loads from the trip sample so we can use the canonical cluster assignment;
# the CSV used for the earlier zone-class version does not have cluster info.
# Output filename kept (trip_margin_by_zone_class_p50.html) for page.html
# compatibility, but the chart is now by cluster.
# ============================================================================
def build_by_cluster():
    sample_path = C.parent / "tables" / "trip_sample_full.parquet"
    canon_path  = C.parent / "tables" / "zone_clusters_canonical.csv"

    print("  loading trip sample...")
    sample = pd.read_parquet(sample_path,
        columns=['PULocationID', 'pickup_datetime', 'operator',
                 'base_passenger_fare', 'driver_pay', 'sampling_weight'])
    sample = sample[(sample['base_passenger_fare'] > 0)
                    & (sample['operator'] == 'Uber')].copy()
    print(f"    {len(sample):,} Uber trips after fare filter")

    canon = pd.read_csv(canon_path)
    z2c = dict(zip(canon['zone_id'], canon['cluster']))
    sample['cluster'] = sample['PULocationID'].map(z2c)
    sample = sample.dropna(subset=['cluster']).copy()
    sample['cluster'] = sample['cluster'].astype(int)
    sample['month'] = sample['pickup_datetime'].dt.to_period('M').dt.to_timestamp()

    g = sample.groupby(['month', 'cluster']).apply(
        lambda x: pd.Series({
            'rev_w': 1 - (x['driver_pay']*x['sampling_weight']).sum() /
                         (x['base_passenger_fare']*x['sampling_weight']).sum(),
        }), include_groups=False).reset_index()
    g = g.sort_values(['cluster', 'month'])
    g['smooth'] = g.groupby('cluster')['rev_w'].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean())

    LABELS = ['Manhattan', 'Brooklyn', 'Bronx + Upper Manhattan', 'Queens']
    COLORS = [PALETTE['navy'], PALETTE['rose'], PALETTE['gold'], PALETTE['sage']]

    fig = go.Figure()
    for c in range(4):
        sub = g[g['cluster'] == c]
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['smooth'], mode='lines',
            line=dict(color=COLORS[c], width=2.2),
            name=LABELS[c],
            hovertemplate=LABELS[c]+'<br>%{x|%b %Y}<br>%{y:.1%}<extra></extra>',
        ))

    fig.add_annotation(x='2020-08-01', y=0.21,
        text='Convergence (Aug 2020)<br>cross-cluster gap narrows',
        showarrow=True, arrowhead=2, arrowsize=0.9, ax=-30, ay=-50,
        font=dict(size=10, color=PALETTE['ink']),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=PALETTE['grey'], borderwidth=1)
    fig.add_annotation(x='2023-03-01', y=0.27,
        text='Divergence (early 2023)<br>Manhattan pulls ahead',
        showarrow=True, arrowhead=2, arrowsize=0.9, ax=20, ay=-40,
        font=dict(size=10, color=PALETTE['ink']),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=PALETTE['grey'], borderwidth=1)

    fig.update_layout(
        xaxis_title='', yaxis_title='Revenue-weighted operator margin',
        yaxis=dict(tickformat='.0%'),
        height=440, legend=LEGEND_COMPACT,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    save_chart(fig, C / 'trip_margin_by_zone_class_p50.html')
    print("  wrote trip_margin_by_zone_class_p50.html (4 hybrid clusters, revenue-weighted)")


if __name__ == "__main__":
    print("Building Part 2 margin charts from existing CSVs...")
    build_monthly()
    build_by_length()
    build_by_cluster()
    print("Done.")
