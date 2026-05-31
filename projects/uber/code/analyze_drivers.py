#!/usr/bin/env python3.10
"""
analyze_drivers.py
==================

Question set E. How has Uber driver pay evolved over the panel, and how
geographically uneven is it? Three regulatory events anchor the analysis:
the February 2019 NYC minimum-pay rule, the August 2022 TLC pay raise, and
the March 2024 raise.

What this script computes
-------------------------
1. Time series of driver pay per trip, per mile, and per hour, citywide.
   Vertical markers at the three regulatory events.
2. Operator margin proxy: 1 - (driver_pay / base_fare), monthly.
3. Borough-level driver pay per hour over time, four panels.
4. Zone-level driver pay heterogeneity, measured by the Gini coefficient of
   monthly driver-pay-per-trip across zones. A rising Gini means dispatch
   routes higher-paying trips into a tightening subset of zones.
5. Tip share by borough, monthly: share of total driver compensation that
   comes from tips rather than guaranteed base pay.

Outputs (written to outputs/)
-----------------------------
  drv_pay_trajectory.html        Three-panel pay trajectory citywide
  drv_margin.html                Operator margin share over time
  drv_pay_by_borough.html        Pay per hour by borough, four-panel
  drv_pay_gini.html              Cross-zone Gini of pay per trip over time
  drv_tip_share_by_borough.html  Tips as a share of total comp by borough
  drv_summary.csv                Headline numbers at each regulatory event

Driver pay availability: the sum_driver_pay column exists only from February
2019 onward (the start of the FHVHV schema). The pre-Feb-2019 months are
shown as gaps where required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _panel_loader import (  # noqa: E402
    classify_zones, load_monthly_zone, load_monthly_city, save_chart,
    base_layout, PALETTE, OUT_DIR, coverage_report,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Regulatory events ────────────────────────────────────────────────────
EVENTS = [
    (pd.Timestamp('2019-02-01'), 'Min-pay rule'),
    (pd.Timestamp('2022-12-19'), 'TLC raise (Dec 2022)'),
    (pd.Timestamp('2024-03-04'), 'TLC raise (Mar 2024)'),
]


# ─── Gini helper ──────────────────────────────────────────────────────────

def gini(values: np.ndarray) -> float:
    """Standard Gini coefficient. Values must be non-negative."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= 0)]
    if len(v) < 2 or v.sum() == 0:
        return np.nan
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * v)) / (n * cum[-1]) - (n + 1) / n)


# ─── Chart 1: three-panel pay trajectory (nominal + real) ─────────────────

CPI_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/'
                'central-bank/data/CPIAUCSL.csv')
REAL_BASE_DATE = pd.Timestamp('2019-02-01')  # constant Feb-2019 dollars


def _load_deflator() -> pd.Series:
    """Return monthly deflator series indexed by month-start timestamps.
    Deflator d_m = CPI_BASE / CPI_m so that nominal * d_m = real (Feb-2019 $).
    Missing months are linearly interpolated; the tail is forward-filled."""
    cpi = pd.read_csv(CPI_PATH, parse_dates=['observation_date'])
    cpi = cpi.rename(columns={'observation_date': 'month',
                              'CPIAUCSL': 'cpi'})
    cpi['month'] = cpi['month'].dt.to_period('M').dt.to_timestamp()
    cpi = cpi.sort_values('month').set_index('month')['cpi']
    # Extend index to cover the driver-pay panel
    idx = pd.date_range(cpi.index.min(), pd.Timestamp('2026-04-01'),
                        freq='MS')
    cpi = cpi.reindex(idx).interpolate(method='linear').ffill()
    base = float(cpi.loc[REAL_BASE_DATE])
    return base / cpi


def chart_pay_trajectory(city: pd.DataFrame) -> None:
    if 'mean_driver_pay' not in city.columns:
        print("  (no driver pay columns; skipping)")
        return

    deflator = _load_deflator()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('Driver pay per trip',
                        'Driver pay per mile',
                        'Driver pay per hour'),
        vertical_spacing=0.08)

    series_specs = [
        ('mean_driver_pay', '$/trip', 1),
        ('driver_pay_per_mile', '$/mile', 2),
        ('driver_pay_per_hour', '$/hour', 3),
    ]
    for col, label, row in series_specs:
        if col not in city.columns:
            continue
        sub = city.dropna(subset=[col]).sort_values('month').copy()
        # Map deflator onto each month
        sub['defl'] = sub['month'].map(deflator).astype(float)
        sub['real'] = sub[col] * sub['defl']

        # Nominal (lighter, thinner, reference)
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub[col], mode='lines',
            line=dict(color=PALETTE['steel'], width=1.4, dash='dot'),
            name='Nominal' if row == 1 else None,
            legendgroup='nominal', showlegend=(row == 1),
            hovertemplate=f'%{{x|%b %Y}}<br>Nominal: %{{y:.2f}} {label}<extra></extra>',
        ), row=row, col=1)
        # Real (primary, in constant Feb-2019 $)
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['real'], mode='lines',
            line=dict(color=PALETTE['navy'], width=2.2),
            name='Real (Feb 2019 $)' if row == 1 else None,
            legendgroup='real', showlegend=(row == 1),
            hovertemplate=f'%{{x|%b %Y}}<br>Real: %{{y:.2f}} {label}<extra></extra>',
        ), row=row, col=1)
        fig.update_yaxes(title=label, gridcolor=PALETTE['grid'],
                         row=row, col=1)

    for ev_date, ev_label in EVENTS:
        for row in (1, 2, 3):
            fig.add_vline(x=ev_date, line=dict(color=PALETTE['slate'],
                                               dash='dot', width=1),
                          row=row, col=1)
        fig.add_annotation(x=ev_date, y=1.02, yref='paper',
                           text=ev_label, showarrow=False,
                           font=dict(size=10, color=PALETTE['slate']),
                           xanchor='left', yanchor='bottom')

    lay = base_layout(height=760, width=900, showlegend=True)
    lay['margin'] = dict(l=70, r=30, t=70, b=80)
    fig.update_layout(**lay)
    for ann in fig.layout.annotations[:3]:  # subplot titles
        ann.font = dict(size=13, color=PALETTE['text'],
                        family='IBM Plex Sans, -apple-system, sans-serif')
    save_chart(fig, 'drivers/drv_pay_trajectory')


# ─── Chart 2: operator margin share ──────────────────────────────────────

def chart_margin(city: pd.DataFrame) -> None:
    if 'operator_margin_share' not in city.columns:
        print("  (no margin columns; skipping)")
        return
    sub = city.dropna(subset=['operator_margin_share']).sort_values('month')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub['month'], y=sub['operator_margin_share'] * 100, mode='lines',
        line=dict(color=PALETTE['gold'], width=2),
        fill='tozeroy', fillcolor='rgba(192, 144, 64, 0.10)',
        showlegend=False,
        hovertemplate='%{x|%b %Y}<br>Margin proxy: %{y:.1f}%<extra></extra>',
    ))
    for ev_date, ev_label in EVENTS:
        fig.add_vline(x=ev_date, line=dict(color=PALETTE['slate'],
                                           dash='dot', width=1))
    fig.update_layout(**base_layout(height=380, showlegend=False))
    fig.update_yaxes(title='1 - driver pay / rider base fare (%)')
    fig.update_xaxes(title=None)
    save_chart(fig, 'drivers/drv_margin')


# ─── Chart 3: pay per hour by borough ────────────────────────────────────

def chart_by_borough(uber_z: pd.DataFrame, zones: pd.DataFrame) -> None:
    if 'sum_driver_pay' not in uber_z.columns or 'sum_time_s' not in uber_z.columns:
        print("  (no pay/time columns; skipping borough chart)")
        return
    z = zones[['zone_id', 'borough']]
    c = uber_z.merge(z, on='zone_id', how='left')
    c.loc[c['borough'].eq('EWR'), 'borough'] = 'Queens'
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']
    by = (c[c['borough'].isin(boroughs)]
           .groupby(['month', 'borough'], as_index=False)
           .agg(pay=('sum_driver_pay', 'sum'),
                seconds=('sum_time_s', 'sum')))
    by['pay_per_hour'] = by['pay'] / (by['seconds'] / 3600.0)

    fig = make_subplots(rows=2, cols=2, subplot_titles=boroughs,
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.14, horizontal_spacing=0.07)
    pos = {'Manhattan': (1, 1), 'Brooklyn': (1, 2),
           'Queens': (2, 1), 'Bronx': (2, 2)}
    for b in boroughs:
        sub = by[by['borough'].eq(b)].sort_values('month')
        r, col = pos[b]
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['pay_per_hour'], mode='lines',
            line=dict(color=PALETTE['navy'], width=2), showlegend=False,
            hovertemplate=f'{b}<br>%{{x|%b %Y}}<br>$%{{y:.2f}}/hr<extra></extra>',
        ), row=r, col=col)
        for ev_date, _ in EVENTS:
            fig.add_vline(x=ev_date, line=dict(color=PALETTE['slate'],
                                               dash='dot', width=1),
                          row=r, col=col)
    lay = base_layout(height=560, width=950, showlegend=False)
    lay['margin'] = dict(l=70, r=30, t=60, b=50)
    fig.update_layout(**lay)
    for r in (1, 2):
        for col in (1, 2):
            fig.update_yaxes(title='$/hour', gridcolor=PALETTE['grid'],
                             row=r, col=col)
            fig.update_xaxes(gridcolor=PALETTE['grid'], row=r, col=col)
    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color=PALETTE['text'],
                        family='IBM Plex Sans, -apple-system, sans-serif')
    save_chart(fig, 'drv_pay_by_borough')


# ─── Chart 4: cross-zone Gini of driver pay per trip ─────────────────────

def chart_pay_gini(uber_z: pd.DataFrame) -> None:
    if 'mean_driver_pay' not in uber_z.columns:
        print("  (no pay column; skipping Gini chart)")
        return
    rows = []
    for month, g in uber_z.groupby('month'):
        # Filter to zones with enough trips to make a stable estimate.
        sub = g[g['trips'] >= 50].dropna(subset=['mean_driver_pay'])
        if len(sub) < 20:
            continue
        rows.append({'month': month,
                     'gini': gini(sub['mean_driver_pay'].values),
                     'n_zones': len(sub)})
    gdf = pd.DataFrame(rows).sort_values('month')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gdf['month'], y=gdf['gini'], mode='lines',
        line=dict(color=PALETTE['rose'], width=2), showlegend=False,
        hovertemplate='%{x|%b %Y}<br>Gini: %{y:.3f}<extra></extra>',
    ))
    for ev_date, _ in EVENTS:
        fig.add_vline(x=ev_date, line=dict(color=PALETTE['slate'],
                                           dash='dot', width=1))
    fig.update_layout(**base_layout(height=400, showlegend=False))
    fig.update_yaxes(title='Gini of mean driver pay per trip, across zones')
    fig.update_xaxes(title=None)
    save_chart(fig, 'drv_pay_gini')


# ─── Chart 5: tip share by borough ───────────────────────────────────────

def chart_tip_share(uber_z: pd.DataFrame, zones: pd.DataFrame) -> None:
    if 'sum_tips' not in uber_z.columns:
        print("  (no tips column; skipping tip chart)")
        return
    z = zones[['zone_id', 'borough']]
    c = uber_z.merge(z, on='zone_id', how='left')
    c.loc[c['borough'].eq('EWR'), 'borough'] = 'Queens'
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']
    by = (c[c['borough'].isin(boroughs)]
           .groupby(['month', 'borough'], as_index=False)
           .agg(pay=('sum_driver_pay', 'sum'),
                tips=('sum_tips', 'sum')))
    by['tip_share'] = by['tips'] / (by['pay'] + by['tips'])
    by = by.replace([np.inf, -np.inf], np.nan).dropna()

    fig = go.Figure()
    colors = {'Manhattan': PALETTE['navy'], 'Brooklyn': PALETTE['rose'],
              'Queens': PALETTE['gold'], 'Bronx': PALETTE['sage']}
    for b in boroughs:
        sub = by[by['borough'].eq(b)].sort_values('month')
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['tip_share'] * 100, mode='lines',
            name=b, line=dict(color=colors[b], width=2),
            hovertemplate=f'{b}<br>%{{x|%b %Y}}<br>Tips: %{{y:.2f}}%<extra></extra>',
        ))
    fig.update_layout(**base_layout(height=420))
    fig.update_yaxes(title='Tips as % of (driver pay + tips)')
    fig.update_xaxes(title=None)
    save_chart(fig, 'drv_tip_share_by_borough')


# ─── Summary CSV around regulatory events ────────────────────────────────

def event_summary(city: pd.DataFrame) -> pd.DataFrame:
    """Six-month average pay per trip / per hour just before and just after
    each regulatory event. Pure descriptive, no inference."""
    rows = []
    for ev_date, ev_label in EVENTS:
        pre = city[(city['month'] >= ev_date - pd.DateOffset(months=6))
                   & (city['month'] < ev_date)]
        post = city[(city['month'] >= ev_date)
                    & (city['month'] < ev_date + pd.DateOffset(months=6))]
        if pre.empty or post.empty:
            continue
        r = {'event': ev_label, 'date': ev_date.strftime('%Y-%m')}
        for col, fmt in [('mean_driver_pay', '${:.2f}'),
                         ('driver_pay_per_hour', '${:.2f}'),
                         ('operator_margin_share', '{:.1%}')]:
            if col in city.columns:
                a = pre[col].mean()
                b = post[col].mean()
                r[f'{col}_pre'] = fmt.format(a) if not np.isnan(a) else 'NA'
                r[f'{col}_post'] = fmt.format(b) if not np.isnan(b) else 'NA'
                r[f'{col}_change'] = (f'{(b - a) / a:+.1%}'
                                      if a and not np.isnan(a) and not np.isnan(b)
                                      else 'NA')
        rows.append(r)
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("analyze_drivers.py")
    print("=" * 72)
    coverage_report('uber')

    zones = classify_zones()
    try:
        uber_z = load_monthly_zone('uber')
        city = load_monthly_city('uber')
    except FileNotFoundError as e:
        print(f"Cannot proceed: {e}")
        return

    print(f"Panel range: {uber_z['month'].min():%Y-%m} to "
          f"{uber_z['month'].max():%Y-%m}")
    print(f"  {uber_z['zone_id'].nunique()} unique zones, "
          f"{uber_z['month'].nunique()} months")

    print("Rendering charts...")
    chart_pay_trajectory(city)
    chart_margin(city)
    # Other drv_* chart functions exist in this file but are not used in page.html
    # (drv_pay_by_borough, drv_pay_gini, drv_tip_share). Not called here.

    # Event summary CSV.
    summary = event_summary(city)
    summary_path = OUT_DIR / 'drv_summary.csv'
    if not summary.empty:
        summary.to_csv(summary_path, index=False)

    # ── End-of-run summary ─────────────────────────────────────────────
    print()
    print("-" * 72)
    print("Summary")
    print("-" * 72)
    if 'mean_driver_pay' in city.columns:
        first = city.dropna(subset=['mean_driver_pay']).iloc[0]
        last = city.dropna(subset=['mean_driver_pay']).iloc[-1]
        print(f"Driver pay per trip:")
        print(f"  {first['month']:%Y-%m}: ${first['mean_driver_pay']:.2f}")
        print(f"  {last['month']:%Y-%m}: ${last['mean_driver_pay']:.2f}")
        print(f"  change: {(last['mean_driver_pay'] / first['mean_driver_pay'] - 1):+.1%}")
    if 'driver_pay_per_hour' in city.columns:
        first = city.dropna(subset=['driver_pay_per_hour']).iloc[0]
        last = city.dropna(subset=['driver_pay_per_hour']).iloc[-1]
        print(f"Driver pay per hour (gross of expenses):")
        print(f"  {first['month']:%Y-%m}: ${first['driver_pay_per_hour']:.2f}")
        print(f"  {last['month']:%Y-%m}: ${last['driver_pay_per_hour']:.2f}")
        print(f"  change: {(last['driver_pay_per_hour'] / first['driver_pay_per_hour'] - 1):+.1%}")
    if 'operator_margin_share' in city.columns:
        last = city.dropna(subset=['operator_margin_share']).iloc[-1]
        print(f"Operator margin share (latest month, "
              f"{last['month']:%Y-%m}): {last['operator_margin_share']:.1%}")

    print()
    print("Outputs:")
    for f in ['drivers/drv_pay_trajectory.html', 'drivers/drv_margin.html',
              'drv_summary.csv']:
        p = OUT_DIR / f
        if p.exists():
            print(f"  {f}  ({p.stat().st_size / 1024:.0f} KB)")
    print("=" * 72)


if __name__ == '__main__':
    main()
