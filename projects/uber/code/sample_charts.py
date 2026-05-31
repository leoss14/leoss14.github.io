"""
sample_charts.py

Generates all charts for the new Part 4.5 "Operator Comparison" section
of page.html, using the stratified trip-level sample produced by
trip_sample_extraction.py.

Reads:  outputs/tables/trip_sample_full.parquet  (8.34M weighted rows)
Writes: outputs/sample/*.html                    (one Plotly HTML per chart)
        outputs/tables/sample_*.csv              (numbers behind the charts)

All means are sampling_weight-weighted so they reflect the population, not
the sample. The sampling weight is N_h / n_h for each stratum.

Run: python3.10 sample_charts.py

Expected runtime: 5-8 minutes on a laptop.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import save_chart, PALETTE, log

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "outputs"
CHART_DIR = OUT_DIR / "sample"
TABLE_DIR = OUT_DIR / "tables"
CHART_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = TABLE_DIR / "trip_sample_full.parquet"

# Colour assignments specific to operator comparisons. Uber gets the navy
# (institutional, dominant share) and Lyft gets rose (warm contrast).
OP_COLORS = {"Uber": PALETTE["navy"], "Lyft": PALETTE["rose"]}

# Zone-class ordering for charts. Geographic ordering: airport, CBD, buffer
# (60th to 65th street), upper Manhattan, outer boroughs.
ZONE_ORDER = ["airport", "cbd", "buffer", "upper_manhattan", "outer"]
ZONE_LABEL = {
    "airport": "Airports",
    "cbd": "CBD (below 60th)",
    "buffer": "Buffer (60th–65th)",
    "upper_manhattan": "Upper Manhattan",
    "outer": "Outer boroughs",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wmean(g: pd.DataFrame, col: str) -> float:
    """Weighted mean of `col` in dataframe `g`, using sampling_weight."""
    s = g.dropna(subset=[col])
    if len(s) == 0:
        return float("nan")
    return float((s[col] * s["sampling_weight"]).sum() / s["sampling_weight"].sum())


def wagg(group_keys, df, cols_map):
    """Group `df` by `group_keys`, return wmean of each column in cols_map."""
    out = []
    for keys, g in df.groupby(group_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        for new_col, src_col in cols_map.items():
            row[new_col] = wmean(g, src_col)
        row["n"] = len(g)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Load & prep
# ---------------------------------------------------------------------------
def load_sample() -> pd.DataFrame:
    log(f"Loading sample from {SAMPLE_PATH}")
    df = pd.read_parquet(SAMPLE_PATH)
    df["month"] = df["pickup_datetime"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["pickup_datetime"].dt.year
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dow"] = df["pickup_datetime"].dt.dayofweek
    # Negative-margin flag (subsidised trip)
    df["is_neg_margin"] = (df["margin_proxy"] < 0).astype(float)
    # Tip share = tips / (base + tips)
    denom = (df["base_passenger_fare"] + df["tips"]).replace(0, np.nan)
    df["tip_share"] = df["tips"] / denom
    df["zero_tip"] = (df["tips"] == 0).astype(float)
    # Effective hours: trip + dispatch (clipped). Where on_scene is bad, use trip only.
    resp = df["response_sec"].clip(lower=0, upper=1800).fillna(0)
    df["effective_hours"] = (df["trip_time"] + resp) / 3600.0
    df["effective_pay_per_hour"] = df["driver_pay"] / df["effective_hours"].replace(0, np.nan)
    # Trip-length buckets (page-level constant for U-shape chart)
    df["dist_bin"] = pd.cut(
        df["trip_miles"],
        [0, 1, 2, 5, 10, 20, 100],
        labels=["<1 mi", "1–2 mi", "2–5 mi", "5–10 mi", "10–20 mi", "20+ mi"],
    )
    log(f"  loaded {df.shape}, ops {df['operator'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 1. Operator margin trajectory (monthly time series)
# ---------------------------------------------------------------------------
def chart_operator_margin_monthly(df: pd.DataFrame) -> None:
    log("[1] operator margin trajectory")
    t = wagg(["month", "operator"], df, {"margin": "margin_proxy"})
    fig = go.Figure()
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].sort_values("month")
        fig.add_trace(go.Scatter(
            x=d["month"], y=d["margin"] * 100,
            mode="lines", name=op,
            line=dict(color=OP_COLORS[op], width=2.2),
            hovertemplate="%{x|%b %Y}<br>" + op + ": %{y:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="#aab0bd", line_width=1, line_dash="dot")
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Operator margin proxy (%)",
        yaxis_ticksuffix="%",
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98),
        height=420,
    )
    save_chart(fig, CHART_DIR / "op_margin_monthly.html")
    t.to_csv(TABLE_DIR / "sample_op_margin_monthly.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Margin by zone class (full-panel bars, Uber vs Lyft)
# ---------------------------------------------------------------------------
def chart_margin_by_zone(df: pd.DataFrame) -> None:
    log("[2] margin by zone class")
    t = wagg(["zone_class", "operator"], df, {"margin": "margin_proxy"})
    fig = go.Figure()
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].set_index("zone_class").reindex(ZONE_ORDER)
        fig.add_trace(go.Bar(
            x=[ZONE_LABEL[z] for z in ZONE_ORDER],
            y=d["margin"] * 100,
            name=op,
            marker_color=OP_COLORS[op],
            hovertemplate="%{x}<br>" + op + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Pickup zone class",
        yaxis_title="Operator margin (%)",
        yaxis_ticksuffix="%",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_margin_by_zone.html")
    t.to_csv(TABLE_DIR / "sample_margin_by_zone.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Subsidisation share over time, by zone class (Uber only)
# ---------------------------------------------------------------------------
def chart_subsidised_share(df: pd.DataFrame) -> None:
    log("[3] subsidised-share trajectory")
    sub = df[df["operator"] == "Uber"].copy()
    t = wagg(["year", "zone_class"], sub, {"frac_neg": "is_neg_margin"})
    fig = go.Figure()
    palette_order = [PALETTE["rose"], PALETTE["navy"], PALETTE["steel"],
                     PALETTE["sage"], PALETTE["gold"]]
    for i, z in enumerate(ZONE_ORDER):
        d = t[t["zone_class"] == z].sort_values("year")
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["frac_neg"] * 100,
            mode="lines+markers", name=ZONE_LABEL[z],
            line=dict(color=palette_order[i], width=2),
            marker=dict(size=6),
            hovertemplate=ZONE_LABEL[z] + " %{x}<br>%{y:.1f}% subsidised<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Share of Uber trips with negative margin (%)",
        yaxis_ticksuffix="%",
        height=440,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_subsidised_share.html")
    t.to_csv(TABLE_DIR / "sample_subsidised_share.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Pay per hour: trip vs effective (with dispatch time)
# ---------------------------------------------------------------------------
def chart_pay_per_hour(df: pd.DataFrame) -> None:
    log("[4] pay-per-hour trip vs effective")
    # Restrict to 2024-2026 where on_scene_datetime is clean for both operators
    rec = df[(df["year"] >= 2024) & df["response_sec"].notna() & (df["response_sec"] > 0)]
    t = wagg(["zone_class", "operator"], rec, {
        "trip_pph": "pay_per_hour",
        "eff_pph": "effective_pay_per_hour",
    })
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Uber", "Lyft",
    ), shared_yaxes=True, horizontal_spacing=0.05)
    for col_i, op in enumerate(["Uber", "Lyft"], start=1):
        d = t[t["operator"] == op].set_index("zone_class").reindex(ZONE_ORDER)
        # Trip pay-per-hour (during trip)
        fig.add_trace(go.Bar(
            x=[ZONE_LABEL[z] for z in ZONE_ORDER],
            y=d["trip_pph"],
            name="During trip" if col_i == 1 else None,
            showlegend=(col_i == 1),
            marker_color=PALETTE["steel"],
            hovertemplate="%{x}<br>During trip: $%{y:.0f}/hr<extra></extra>",
        ), row=1, col=col_i)
        # Effective pay-per-hour (incl. dispatch)
        fig.add_trace(go.Bar(
            x=[ZONE_LABEL[z] for z in ZONE_ORDER],
            y=d["eff_pph"],
            name="Including dispatch time" if col_i == 1 else None,
            showlegend=(col_i == 1),
            marker_color=OP_COLORS[op] if op == "Uber" else PALETTE["rose"],
            hovertemplate="%{x}<br>Effective: $%{y:.0f}/hr<extra></extra>",
        ), row=1, col=col_i)
    fig.update_layout(
        barmode="group",
        yaxis_title="Driver pay ($ per hour)",
        yaxis_tickprefix="$",
        height=440,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99),
    )
    fig.update_xaxes(tickangle=-20)
    save_chart(fig, CHART_DIR / "op_pay_per_hour.html")
    t.to_csv(TABLE_DIR / "sample_pay_per_hour.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Pay-per-hour trajectory by operator (monthly time series)
# ---------------------------------------------------------------------------
def chart_pay_per_hour_monthly(df: pd.DataFrame) -> None:
    log("[5] pay-per-hour monthly")
    t = wagg(["month", "operator"], df, {"pph": "pay_per_hour"})
    fig = go.Figure()
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].sort_values("month")
        fig.add_trace(go.Scatter(
            x=d["month"], y=d["pph"],
            mode="lines", name=op,
            line=dict(color=OP_COLORS[op], width=2.2),
            hovertemplate="%{x|%b %Y}<br>" + op + ": $%{y:.0f}/hr<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Driver pay during trip ($ per hour)",
        yaxis_tickprefix="$",
        height=420,
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_pay_per_hour_monthly.html")
    t.to_csv(TABLE_DIR / "sample_pay_per_hour_monthly.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Response time inequity (2024-2026, by zone, with quartiles)
# ---------------------------------------------------------------------------
def chart_response_time(df: pd.DataFrame) -> None:
    log("[6] response time")
    rec = df[(df["year"] >= 2024) & df["response_sec"].notna()].copy()
    rec["response_sec"] = rec["response_sec"].clip(0, 3600)
    # Compute weighted percentiles via the cheap "repeat by integer weight" trick
    # is too memory-heavy; use unweighted percentiles within each cell, which is
    # close to weighted within strata since the weight only varies across strata.
    rows = []
    for op in ["Uber", "Lyft"]:
        for z in ZONE_ORDER:
            s = rec[(rec["operator"] == op) & (rec["zone_class"] == z)]["response_sec"]
            if len(s) == 0:
                continue
            rows.append({
                "operator": op, "zone_class": z, "n": len(s),
                "p10": s.quantile(0.10), "p25": s.quantile(0.25),
                "p50": s.quantile(0.50), "p75": s.quantile(0.75),
                "p90": s.quantile(0.90), "mean": s.mean(),
            })
    t = pd.DataFrame(rows)
    fig = go.Figure()
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].set_index("zone_class").reindex(ZONE_ORDER)
        fig.add_trace(go.Bar(
            x=[ZONE_LABEL[z] for z in ZONE_ORDER],
            y=d["p50"],
            name=op + " median",
            marker_color=OP_COLORS[op],
            error_y=dict(
                type="data",
                symmetric=False,
                array=d["p90"] - d["p50"],
                arrayminus=d["p50"] - d["p10"],
                color="#9aa3b2", thickness=1.2, width=4,
            ),
            hovertemplate="%{x}<br>" + op + " median: %{y:.0f}s<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Pickup zone class",
        yaxis_title="Response time, request to on-scene (seconds)",
        height=440,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    fig.update_xaxes(tickangle=-15)
    save_chart(fig, CHART_DIR / "op_response_time.html")
    t.to_csv(TABLE_DIR / "sample_response_time.csv", index=False)


# ---------------------------------------------------------------------------
# 7. Uber Pool: requested vs matched, monthly
# ---------------------------------------------------------------------------
def chart_pool_trajectory(df: pd.DataFrame) -> None:
    log("[7] Uber Pool trajectory")
    sub = df[df["operator"] == "Uber"].copy()
    t = wagg(["month"], sub, {
        "pool_req": "is_shared_requested",
        "pool_match": "is_shared_matched",
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["month"], y=t["pool_req"] * 100,
        mode="lines", name="Requested",
        line=dict(color=PALETTE["steel"], width=2),
        hovertemplate="%{x|%b %Y}<br>Requested: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t["month"], y=t["pool_match"] * 100,
        mode="lines", name="Matched (actually shared)",
        line=dict(color=PALETTE["navy"], width=2.2),
        hovertemplate="%{x|%b %Y}<br>Matched: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Share of Uber trips (%)",
        yaxis_ticksuffix="%",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_pool_trajectory.html")
    t.to_csv(TABLE_DIR / "sample_pool_trajectory.csv", index=False)


# ---------------------------------------------------------------------------
# 8. Pool revival by zone (2024+)
# ---------------------------------------------------------------------------
def chart_pool_by_zone(df: pd.DataFrame) -> None:
    log("[8] Uber Pool by zone, recent")
    rec = df[(df["operator"] == "Uber") & (df["year"] >= 2024)].copy()
    t = wagg(["zone_class"], rec, {
        "pool_req": "is_shared_requested",
        "pool_match": "is_shared_matched",
    })
    t = t.set_index("zone_class").reindex(ZONE_ORDER).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[ZONE_LABEL[z] for z in ZONE_ORDER],
        y=t["pool_req"] * 100,
        name="Requested", marker_color=PALETTE["steel"],
        hovertemplate="%{x}<br>Requested: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[ZONE_LABEL[z] for z in ZONE_ORDER],
        y=t["pool_match"] * 100,
        name="Matched", marker_color=PALETTE["navy"],
        hovertemplate="%{x}<br>Matched: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Pickup zone class",
        yaxis_title="Share of Uber trips (%)",
        yaxis_ticksuffix="%",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    fig.update_xaxes(tickangle=-15)
    save_chart(fig, CHART_DIR / "op_pool_by_zone.html")
    t.to_csv(TABLE_DIR / "sample_pool_by_zone.csv", index=False)


# ---------------------------------------------------------------------------
# 9. WAV access ramp
# ---------------------------------------------------------------------------
def chart_wav(df: pd.DataFrame) -> None:
    log("[9] WAV ramp")
    t = wagg(["year", "operator"], df, {
        "wav_req": "is_wav_requested",
        "wav_match": "is_wav_matched",
    })
    fig = go.Figure()
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].sort_values("year")
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["wav_match"] * 100,
            mode="lines+markers", name=op,
            line=dict(color=OP_COLORS[op], width=2.2),
            marker=dict(size=7),
            hovertemplate="%{x}<br>" + op + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Share of trips in wheelchair-accessible vehicle (%)",
        yaxis_ticksuffix="%",
        height=400,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_wav_ramp.html")
    t.to_csv(TABLE_DIR / "sample_wav.csv", index=False)


# ---------------------------------------------------------------------------
# 10. Tip geography
# ---------------------------------------------------------------------------
def chart_tip_geography(df: pd.DataFrame) -> None:
    log("[10] tip geography")
    t = wagg(["zone_class", "operator"], df, {
        "tip_amt": "tips",
        "tip_share": "tip_share",
        "zero_rate": "zero_tip",
    })
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Mean tip ($/trip)",
        "Share of trips with no tip (%)",
    ), horizontal_spacing=0.12)
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].set_index("zone_class").reindex(ZONE_ORDER)
        fig.add_trace(go.Bar(
            x=[ZONE_LABEL[z] for z in ZONE_ORDER],
            y=d["tip_amt"], name=op,
            marker_color=OP_COLORS[op],
            showlegend=True,
            hovertemplate="%{x}<br>" + op + ": $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[ZONE_LABEL[z] for z in ZONE_ORDER],
            y=d["zero_rate"] * 100, name=op,
            marker_color=OP_COLORS[op],
            showlegend=False,
            hovertemplate="%{x}<br>" + op + ": %{y:.1f}%<extra></extra>",
        ), row=1, col=2)
    fig.update_layout(
        barmode="group", height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.45),
    )
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=2)
    fig.update_xaxes(tickangle=-20)
    save_chart(fig, CHART_DIR / "op_tip_geography.html")
    t.to_csv(TABLE_DIR / "sample_tip_geography.csv", index=False)


# ---------------------------------------------------------------------------
# 11. Margin by trip length (U-shape)
# ---------------------------------------------------------------------------
def chart_margin_by_length(df: pd.DataFrame) -> None:
    log("[11] margin by trip length")
    rec = df[df["year"] >= 2024].copy()
    t = wagg(["dist_bin", "operator"], rec, {"margin": "margin_proxy"})
    fig = go.Figure()
    order = ["<1 mi", "1–2 mi", "2–5 mi", "5–10 mi", "10–20 mi", "20+ mi"]
    for op in ["Uber", "Lyft"]:
        d = t[t["operator"] == op].set_index("dist_bin").reindex(order)
        fig.add_trace(go.Bar(
            x=order, y=d["margin"] * 100, name=op,
            marker_color=OP_COLORS[op],
            hovertemplate="%{x}<br>" + op + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Trip distance",
        yaxis_title="Operator margin (%)",
        yaxis_ticksuffix="%",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_margin_by_length.html")
    t.to_csv(TABLE_DIR / "sample_margin_by_length.csv", index=False)


# ---------------------------------------------------------------------------
# 12. Hour-of-day patterns
# ---------------------------------------------------------------------------
def chart_hour_of_day(df: pd.DataFrame) -> None:
    log("[12] hour-of-day patterns")
    rec = df[df["year"] >= 2024].copy()
    t = wagg(["hour"], rec, {
        "margin": "margin_proxy",
        "fpm": "fare_per_mile",
        "pph": "pay_per_hour",
    })
    t = t.sort_values("hour")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=t["hour"], y=t["fpm"], mode="lines+markers",
        name="Fare per mile ($)", line=dict(color=PALETTE["rose"], width=2),
        marker=dict(size=5),
        hovertemplate="hour %{x}<br>$%{y:.2f}/mi<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t["hour"], y=t["pph"], mode="lines+markers",
        name="Driver pay per hour ($)", line=dict(color=PALETTE["navy"], width=2),
        marker=dict(size=5),
        hovertemplate="hour %{x}<br>$%{y:.0f}/hr<extra></extra>",
    ), secondary_y=True)
    fig.update_xaxes(title_text="Hour of day", tickmode="linear", tick0=0, dtick=2)
    fig.update_yaxes(title_text="Rider fare per mile ($)", tickprefix="$",
                     secondary_y=False)
    fig.update_yaxes(title_text="Driver pay per hour ($)", tickprefix="$",
                     secondary_y=True)
    fig.update_layout(height=420,
                      legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98))
    save_chart(fig, CHART_DIR / "op_hour_of_day.html")
    t.to_csv(TABLE_DIR / "sample_hour_of_day.csv", index=False)


# ---------------------------------------------------------------------------
# 13. Speed degradation (annual median)
# ---------------------------------------------------------------------------
def chart_speed_degradation(df: pd.DataFrame) -> None:
    log("[13] speed degradation")
    rows = []
    for y, g in df.groupby("year"):
        rows.append({
            "year": y,
            "p10": g["speed_mph"].quantile(0.1),
            "p50": g["speed_mph"].quantile(0.5),
            "p90": g["speed_mph"].quantile(0.9),
        })
    t = pd.DataFrame(rows).sort_values("year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t["year"], y=t["p90"], line=dict(width=0), showlegend=False,
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t["year"], y=t["p10"], fill="tonexty", line=dict(width=0),
        fillcolor="rgba(31,42,68,0.12)", name="P10 to P90 range",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t["year"], y=t["p50"], mode="lines+markers", name="Median trip speed",
        line=dict(color=PALETTE["navy"], width=2.2), marker=dict(size=7),
        hovertemplate="%{x}<br>Median: %{y:.1f} mph<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Trip speed (mph)",
        height=400,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    save_chart(fig, CHART_DIR / "op_speed_degradation.html")
    t.to_csv(TABLE_DIR / "sample_speed.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    df = load_sample()

    chart_operator_margin_monthly(df)
    chart_margin_by_zone(df)
    chart_subsidised_share(df)
    chart_pay_per_hour(df)
    chart_pay_per_hour_monthly(df)
    chart_response_time(df)
    chart_pool_trajectory(df)
    chart_pool_by_zone(df)
    chart_wav(df)
    chart_tip_geography(df)
    chart_margin_by_length(df)
    chart_hour_of_day(df)
    chart_speed_degradation(df)

    log(f"All sample charts written in {time.time() - t0:.1f}s")
    log(f"  -> {CHART_DIR}")


if __name__ == "__main__":
    main()
