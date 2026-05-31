#!/usr/bin/env python3.10
"""
analyze_cbd.py
==============

Question set A. Did the MTA central business district congestion fee (effective
5 January 2025) change FHV (Uber and Lyft) demand and pricing inside Manhattan
south of 60th Street?

The fee is a flat per-trip charge on FHV trips that enter the CBD. The data
schema added a `cbd_congestion_fee` column from January 2025 onward, but the
behavioural shifts we care about are visible in the volume and pricing series
even without that column, by comparing the months either side of the
implementation date.

What this script computes
-------------------------
1. Volume of trips inside CBD vs outside CBD vs in a 5-block buffer (60th to
   65th Street) on a monthly time series, both operators.
2. Difference-in-differences of pre-period to post-period change, Uber vs Lyft
   and inside-CBD vs outside-CBD. Lyft is a partial control: it pays the same
   fee, so the DiD identifies anything operator-specific in the response
   (algorithmic re-routing, surge behaviour) rather than the fee itself.
3. Pass-through: change in mean rider fare per trip and mean driver pay per
   trip inside CBD before vs after, with an outer-borough placebo.
4. Cross-border buffer behaviour: did the share of pickups in the 60th-65th
   buffer rise after the fee, consistent with riders walking out of the CBD
   to avoid the charge?

Outputs (written to outputs/)
-----------------------------
  cbd_volume_by_zone_class.html      Stacked area of trip volume by region
  cbd_share_inside.html              CBD share of total trips over time
  cbd_buffer_share.html              Buffer zone share over time
  cbd_passthrough.html               Fare and driver pay per trip, CBD vs outer
  cbd_did_summary.csv                DiD coefficients in plain text

Requires the aggregator to have finished or to have at least 6 months on each
side of January 2025. The script prints coverage at the top and exits early
if the panel doesn't span the event.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _panel_loader import (  # noqa: E402
    classify_zones, load_monthly_zone, save_chart, base_layout,
    PALETTE, OPERATOR_COLOR, OUT_DIR, coverage_report,
)

import plotly.graph_objects as go


# ─── Event configuration ──────────────────────────────────────────────────
EVENT_DATE = pd.Timestamp('2025-01-01')
PRE_WINDOW_MONTHS = 12
POST_WINDOW_MONTHS = 12
ZONE_CLASS_ORDER = ['cbd', 'buffer', 'upper_manhattan',
                    'outer_borough', 'airport']
CLASS_LABEL = {
    'cbd':              'CBD (south of 60th)',
    'buffer':           'Buffer (60th-65th)',
    'upper_manhattan':  'Upper Manhattan',
    'outer_borough':    'Outer boroughs',
    'airport':          'Airports',
}
CLASS_COLOR = {
    # Outer borough is the largest stack by volume and must recede visually,
    # so it gets a near-neutral light grey. CBD is the focus (saturated rose).
    # Buffer is a warm secondary (gold). Upper Manhattan and airport are
    # muted neutrals at distinct lightness levels.
    'cbd':              '#b04668',   # rose, the focus
    'buffer':           '#d4a44a',   # warm gold, adjacent policy zone
    'upper_manhattan':  '#7a8a9a',   # medium cool grey
    'outer_borough':    '#d8dee5',   # very light cool grey (recedes)
    'airport':          '#9a8870',   # muted warm grey
}


# ─── Helpers ──────────────────────────────────────────────────────────────

def attach_zone_class(panel: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(
        zones[['zone_id', 'borough', 'zone_class', 'is_airport']],
        on='zone_id', how='left')


def monthly_by_class(panel: pd.DataFrame) -> pd.DataFrame:
    """Sum trips per (month, zone_class)."""
    return (panel.groupby(['month', 'zone_class'], as_index=False)
                 .agg(trips=('trips', 'sum'),
                      sum_base_fare=('sum_base_fare', 'sum')
                          if 'sum_base_fare' in panel.columns else
                          ('trips', 'first'),  # placeholder
                      sum_driver_pay=('sum_driver_pay', 'sum')
                          if 'sum_driver_pay' in panel.columns else
                          ('trips', 'first')))


def did_inside_vs_outside(panel: pd.DataFrame, value_col: str = 'trips'):
    """Plain two-by-two DiD on log trips.

    Returns dict with pre/post means inside CBD, pre/post means outside,
    and the DiD point estimate. No standard errors here; this is descriptive.
    """
    pre = panel[(panel['month'] < EVENT_DATE) &
                (panel['month'] >= EVENT_DATE - pd.DateOffset(months=PRE_WINDOW_MONTHS))]
    post = panel[(panel['month'] >= EVENT_DATE) &
                 (panel['month'] < EVENT_DATE + pd.DateOffset(months=POST_WINDOW_MONTHS))]

    def m(df, inside: bool):
        sub = df[df['zone_class'].eq('cbd')] if inside \
            else df[df['zone_class'].isin(['outer_borough', 'upper_manhattan'])]
        return float(sub[value_col].sum())

    in_pre, in_post = m(pre, True), m(post, True)
    out_pre, out_post = m(pre, False), m(post, False)

    def logd(a, b): return np.log(b / a) if a > 0 and b > 0 else np.nan
    did = logd(in_pre, in_post) - logd(out_pre, out_post)
    return {
        'in_pre': in_pre, 'in_post': in_post,
        'out_pre': out_pre, 'out_post': out_post,
        'log_change_inside': logd(in_pre, in_post),
        'log_change_outside': logd(out_pre, out_post),
        'did': did,
    }


# ─── Chart 1: volume by zone class, stacked area ──────────────────────────

def chart_volume_by_class(uber_class: pd.DataFrame,
                          lyft_class: pd.DataFrame | None) -> None:
    fig = go.Figure()
    for cls in ZONE_CLASS_ORDER:
        sub = uber_class[uber_class['zone_class'].eq(cls)].sort_values('month')
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['trips'] / 1e6, mode='lines',
            name=CLASS_LABEL[cls], stackgroup='one',
            line=dict(width=0.6, color=CLASS_COLOR[cls]),
            fillcolor=CLASS_COLOR[cls],
            hovertemplate=f"{CLASS_LABEL[cls]}<br>%{{x|%b %Y}}<br>%{{y:.2f}}M trips<extra></extra>",
        ))
    fig.add_vline(x=EVENT_DATE, line=dict(color=PALETTE['text'], dash='dash', width=1))
    fig.add_annotation(x=EVENT_DATE, y=1.02, yref='paper',
                       text='CBD fee starts', showarrow=False,
                       font=dict(size=11, color=PALETTE['text']),
                       xanchor='left', yanchor='bottom')
    fig.update_layout(**base_layout(height=480))
    fig.update_yaxes(title='Monthly Uber trips (millions)')
    fig.update_xaxes(title=None)
    save_chart(fig, 'cbd/cbd_volume_by_zone_class')


# ─── Chart 2: CBD share of total trips ────────────────────────────────────

def chart_cbd_share(uber_class: pd.DataFrame,
                    lyft_class: pd.DataFrame | None) -> None:
    def _share(df, target='cbd'):
        wide = df.pivot(index='month', columns='zone_class', values='trips').fillna(0)
        return (wide[target] / wide.sum(axis=1)).rename('share').reset_index()

    fig = go.Figure()
    u = _share(uber_class)
    fig.add_trace(go.Scatter(
        x=u['month'], y=u['share'] * 100, mode='lines',
        name='Uber', line=dict(color=OPERATOR_COLOR['uber'], width=2.2),
        hovertemplate='Uber<br>%{x|%b %Y}<br>%{y:.1f}%<extra></extra>'))
    if lyft_class is not None:
        l = _share(lyft_class)
        fig.add_trace(go.Scatter(
            x=l['month'], y=l['share'] * 100, mode='lines',
            name='Lyft', line=dict(color=OPERATOR_COLOR['lyft'], width=2.2),
            hovertemplate='Lyft<br>%{x|%b %Y}<br>%{y:.1f}%<extra></extra>'))

    fig.add_vline(x=EVENT_DATE, line=dict(color=PALETTE['text'], dash='dash', width=1))
    fig.add_annotation(x=EVENT_DATE, y=1.02, yref='paper',
                       text='CBD fee starts', showarrow=False,
                       font=dict(size=11), xanchor='left', yanchor='bottom')
    fig.update_layout(**base_layout(height=420))
    fig.update_yaxes(title='Share of monthly trips picked up in CBD (%)')
    fig.update_xaxes(title=None)
    save_chart(fig, 'cbd/cbd_share_inside')


# ─── Chart 3: buffer zone share ───────────────────────────────────────────

def chart_buffer_share(uber_class: pd.DataFrame) -> None:
    wide = uber_class.pivot(index='month', columns='zone_class',
                            values='trips').fillna(0)
    if 'buffer' not in wide.columns:
        print("  (no buffer zones identified; skipping buffer chart)")
        return
    manhattan_total = (wide.get('cbd', 0) + wide.get('buffer', 0)
                       + wide.get('upper_manhattan', 0))
    share = (wide['buffer'] / manhattan_total).rename('buffer_share').reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=share['month'], y=share['buffer_share'] * 100, mode='lines',
        line=dict(color=PALETTE['gold'], width=2.2),
        hovertemplate='%{x|%b %Y}<br>%{y:.2f}%<extra></extra>',
        name='Buffer share',
    ))
    fig.add_vline(x=EVENT_DATE, line=dict(color=PALETTE['text'], dash='dash', width=1))
    fig.add_annotation(x=EVENT_DATE, y=1.02, yref='paper',
                       text='CBD fee starts', showarrow=False,
                       font=dict(size=11), xanchor='left', yanchor='bottom')
    fig.update_layout(**base_layout(height=380, showlegend=False))
    fig.update_yaxes(title='Share of Manhattan pickups (%)')
    fig.update_xaxes(title=None)
    save_chart(fig, 'cbd/cbd_buffer_share')


# ─── Chart 4: pass-through (fare and driver pay per trip) ────────────────

def chart_passthrough(uber_zone: pd.DataFrame) -> None:
    if 'sum_base_fare' not in uber_zone.columns:
        print("  (no fare columns; skipping pass-through)")
        return
    by = (uber_zone.groupby(['month', 'zone_class'])
                   .agg(trips=('trips', 'sum'),
                        sum_fare=('sum_base_fare', 'sum'),
                        sum_pay=('sum_driver_pay', 'sum'))
                   .reset_index())
    by['fare_per_trip'] = by['sum_fare'] / by['trips']
    by['pay_per_trip'] = by['sum_pay'] / by['trips']

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True,
                        subplot_titles=('Rider fare per trip',
                                        'Driver pay per trip'),
                        horizontal_spacing=0.09)

    for cls, color in [('cbd', PALETTE['rose']),
                       ('outer_borough', PALETTE['navy'])]:
        sub = by[by['zone_class'].eq(cls)].sort_values('month')
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['fare_per_trip'], mode='lines',
            name=CLASS_LABEL[cls], line=dict(color=color, width=2),
            legendgroup=cls, showlegend=True,
            hovertemplate=f"{CLASS_LABEL[cls]}<br>%{{x|%b %Y}}<br>$%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sub['month'], y=sub['pay_per_trip'], mode='lines',
            name=CLASS_LABEL[cls], line=dict(color=color, width=2),
            legendgroup=cls, showlegend=False,
            hovertemplate=f"{CLASS_LABEL[cls]}<br>%{{x|%b %Y}}<br>$%{{y:.2f}}<extra></extra>",
        ), row=1, col=2)

    for col in (1, 2):
        fig.add_vline(x=EVENT_DATE, line=dict(color=PALETTE['text'],
                                              dash='dash', width=1),
                      row=1, col=col)

    lay = base_layout(height=420, width=950)
    lay['margin'] = dict(l=60, r=30, t=60, b=95)
    fig.update_layout(**lay)
    for col in (1, 2):
        fig.update_yaxes(title='USD per trip', gridcolor=PALETTE['grid'],
                         row=1, col=col)
        fig.update_xaxes(gridcolor=PALETTE['grid'], row=1, col=col)
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=PALETTE['text'],
                        family=FONT_FAMILY)
    save_chart(fig, 'cbd/cbd_passthrough')


FONT_FAMILY = 'IBM Plex Sans, -apple-system, sans-serif'


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("analyze_cbd.py")
    print("=" * 72)
    coverage_report('uber')
    coverage_report('lyft')

    zones = classify_zones()
    print(f"Zone classification: "
          + ", ".join(f"{c}={int((zones['zone_class'] == c).sum())}"
                      for c in ZONE_CLASS_ORDER))

    # Load per-zone monthly aggregates and tag each zone with its class.
    uber_z = load_monthly_zone('uber')
    uber_z = attach_zone_class(uber_z, zones)

    try:
        lyft_z = load_monthly_zone('lyft')
        lyft_z = attach_zone_class(lyft_z, zones)
    except Exception as e:
        print(f"Lyft not available ({e}); proceeding Uber-only")
        lyft_z = None

    # Aggregate by class for the volume/share charts.
    uber_class = monthly_by_class(uber_z)
    lyft_class = monthly_by_class(lyft_z) if lyft_z is not None else None

    # Check whether the event date is within the panel.
    min_m, max_m = uber_z['month'].min(), uber_z['month'].max()
    print(f"Panel range: {min_m:%Y-%m} to {max_m:%Y-%m}")
    if max_m < EVENT_DATE:
        print("Panel does not yet include January 2025. Stopping early.")
        return

    print("Rendering charts...")
    chart_volume_by_class(uber_class, lyft_class)
    chart_cbd_share(uber_class, lyft_class)
    chart_buffer_share(uber_class)
    chart_passthrough(uber_z)

    # DiD summary table.
    did_trip = did_inside_vs_outside(uber_class)
    rows = [{
        'measure': 'log(trips) inside CBD vs outside, Uber',
        **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in did_trip.items()},
    }]
    if lyft_class is not None:
        did_lyft = did_inside_vs_outside(lyft_class)
        rows.append({
            'measure': 'log(trips) inside CBD vs outside, Lyft',
            **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in did_lyft.items()},
        })
    did_df = pd.DataFrame(rows)
    did_path = OUT_DIR / 'cbd_did_summary.csv'
    did_df.to_csv(did_path, index=False)

    # ── End-of-run summary ─────────────────────────────────────────────
    print()
    print("-" * 72)
    print("Summary")
    print("-" * 72)
    print(f"Pre-window:  {EVENT_DATE - pd.DateOffset(months=PRE_WINDOW_MONTHS):%Y-%m} "
          f"to {EVENT_DATE - pd.DateOffset(months=1):%Y-%m}")
    print(f"Post-window: {EVENT_DATE:%Y-%m} "
          f"to {EVENT_DATE + pd.DateOffset(months=POST_WINDOW_MONTHS - 1):%Y-%m}")
    print()
    print("Uber DiD (log change, inside CBD - outside):")
    print(f"  inside CBD :  {did_trip['log_change_inside']:+.4f}  "
          f"({did_trip['in_pre']/1e6:.2f}M -> {did_trip['in_post']/1e6:.2f}M trips)")
    print(f"  outside CBD:  {did_trip['log_change_outside']:+.4f}  "
          f"({did_trip['out_pre']/1e6:.2f}M -> {did_trip['out_post']/1e6:.2f}M trips)")
    print(f"  DiD        :  {did_trip['did']:+.4f}  "
          f"({np.exp(did_trip['did']) - 1:+.1%} differential)")
    if lyft_class is not None:
        print()
        print("Lyft DiD (same comparison):")
        print(f"  inside CBD :  {did_lyft['log_change_inside']:+.4f}")
        print(f"  outside CBD:  {did_lyft['log_change_outside']:+.4f}")
        print(f"  DiD        :  {did_lyft['did']:+.4f}")
    print()
    print("Outputs:")
    for f in ['cbd/cbd_volume_by_zone_class.html', 'cbd/cbd_share_inside.html',
              'cbd/cbd_buffer_share.html', 'cbd/cbd_passthrough.html',
              'cbd_did_summary.csv']:
        p = OUT_DIR / f
        if p.exists():
            print(f"  {f}  ({p.stat().st_size / 1024:.0f} KB)")
    print("=" * 72)


if __name__ == '__main__':
    main()
