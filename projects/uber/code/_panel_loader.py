"""
_panel_loader.py
================

Shared utilities for the new (May 2026) analysis scripts that consume the
monthly aggregates produced by aggregate_monthly.py. Three downstream files
import from here:

  analyze_cbd.py        Congestion pricing (Jan 2025 CBD fee)
  analyze_market.py     Uber vs Lyft market structure
  analyze_drivers.py    Driver economics

What's in here
--------------
- load_monthly_panel(operator)         per (zone, hour, dow, month)
- load_monthly_zone(operator)          per (zone, month)
- load_monthly_city(operator)          per month, citywide
- classify_zones()                     CBD / buffer / upper_manhattan / outer
- PLOTLY_LAYOUT                        shared chart styling
- save_chart(fig, name)                writes HTML to outputs/

Robustness
----------
If the aggregator is still mid-run, all loaders will just return whatever
months are on disk and print a one-line coverage summary. No hard fail.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
AGG_ROOT = PROJECT_DIR / 'data' / 'aggregates'
OUT_DIR = PROJECT_DIR / 'outputs'
ZONE_CSV = PROJECT_DIR / 'data' / 'zone_centroids.csv'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CBD geography ────────────────────────────────────────────────────────
# The MTA central business district congestion fee (effective 5 Jan 2025)
# applies to FHV trips that enter Manhattan south of 60th Street. We classify
# zones using their centroid latitude. The cutoff at 40.7635 corresponds to
# Central Park South (60th Street). A 5-block buffer (60th to 65th) is used
# to detect cross-border substitution.
CBD_LAT_CUTOFF = 40.7635          # 60th Street
BUFFER_LAT_TOP = 40.7720          # ~65th Street

# ─── Plotly styling (project standard) ────────────────────────────────────
PALETTE = {
    'navy':       '#1f2c4d',
    'slate':      '#4a5568',
    'steel':      '#8b9bb0',
    'rose':       '#b04668',
    'gold':       '#c09040',
    'sage':       '#5a8a72',
    'cloud':      '#e5eaf0',
    'paper':      '#ffffff',
    'panel_bg':   '#f7f9fc',
    'text':       '#1a202c',
    'grid':       '#e5eaf0',
}

OPERATOR_COLOR = {'uber': PALETTE['navy'], 'lyft': PALETTE['rose']}

FONT = dict(family='IBM Plex Sans, -apple-system, sans-serif',
            size=13, color=PALETTE['text'])

def base_layout(height=460, width=900, showlegend=True):
    """Standard chart layout. Titles are suppressed; the surrounding HTML
    caption box on page.html handles those.

    Legend is bottom-centered (horizontal) so it does not clash with
    top-of-area event annotations like 'CBD fee starts'."""
    return dict(
        font=FONT,
        title=None,
        paper_bgcolor=PALETTE['paper'],
        plot_bgcolor=PALETTE['paper'],
        margin=dict(l=60, r=30, t=40, b=85),
        height=height, width=width,
        showlegend=showlegend,
        legend=dict(orientation='h', yanchor='top', y=-0.14,
                    xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12)),
        xaxis=dict(gridcolor=PALETTE['grid'], zeroline=False,
                   linecolor=PALETTE['slate'], ticks='outside'),
        yaxis=dict(gridcolor=PALETTE['grid'], zeroline=False,
                   linecolor=PALETTE['slate'], ticks='outside'),
        hoverlabel=dict(bgcolor='white', bordercolor=PALETTE['slate'],
                        font=dict(family=FONT['family'], size=12)),
    )

def save_chart(fig, name: str):
    """Write fig to outputs/<name>.html. Returns the path."""
    if not name.endswith('.html'):
        name = name + '.html'
    out = OUT_DIR / name
    fig.write_html(str(out), include_plotlyjs='cdn', full_html=True,
                   config={'displayModeBar': False})
    return out


# ─── Zone classification ──────────────────────────────────────────────────

def classify_zones() -> pd.DataFrame:
    """One row per zone with borough, lat, lon, and CBD/buffer status.

    Returned columns:
      zone_id, zone_name, borough, latitude, longitude,
      zone_class  one of {cbd, buffer, upper_manhattan, outer_borough, airport}
      is_airport  bool (JFK, LGA, EWR)
    """
    z = pd.read_csv(ZONE_CSV)
    z['is_airport'] = (
        z['borough'].eq('EWR')
        | z['zone_name'].str.contains('JFK', case=False, na=False)
        | z['zone_name'].str.contains('LaGuardia', case=False, na=False)
        | z['zone_name'].str.contains('Newark', case=False, na=False)
    )

    def _classify(r):
        if r['is_airport']:
            return 'airport'
        if r['borough'] != 'Manhattan':
            return 'outer_borough'
        if r['latitude'] < CBD_LAT_CUTOFF:
            return 'cbd'
        if r['latitude'] < BUFFER_LAT_TOP:
            return 'buffer'
        return 'upper_manhattan'

    z['zone_class'] = z.apply(_classify, axis=1)
    return z


# ─── Data loading ─────────────────────────────────────────────────────────

_MAIN_RE = re.compile(r'^(\d{4})-(\d{2})_main\.parquet$')

def _list_months(operator: str) -> list[str]:
    op_dir = AGG_ROOT / operator
    if not op_dir.exists():
        return []
    months = []
    for f in op_dir.iterdir():
        m = _MAIN_RE.match(f.name)
        if m:
            months.append(f"{m.group(1)}-{m.group(2)}")
    return sorted(months)

def coverage_report(operator: str) -> None:
    months = _list_months(operator)
    if not months:
        print(f"[{operator}] no aggregates found")
        return
    print(f"[{operator}] {len(months)} months  "
          f"({months[0]} to {months[-1]})")


def load_monthly_panel(operator: str,
                       columns: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every monthly main aggregate for `operator` into one frame.

    Columns are the union of what's available across months (e.g. sum_cbd_fee
    exists only from 2025 onward, sum_response_s only from FHVHV).

    For most analyses you want load_monthly_zone() instead since it drops
    hour/dow and is much smaller.
    """
    months = _list_months(operator)
    if not months:
        raise FileNotFoundError(f"No aggregates for {operator}")
    frames = []
    for tag in months:
        p = AGG_ROOT / operator / f"{tag}_main.parquet"
        df = pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)
        df['month'] = pd.Period(tag, freq='M').to_timestamp()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out['operator'] = operator
    return out


def load_monthly_zone(operator: str) -> pd.DataFrame:
    """Per (zone_id, month) panel, summing across hours and days-of-week.

    Adds derived columns:
      mean_response_s      (sum_response_s / n_with_response, if present)
      mean_driver_pay      (sum_driver_pay / trips, if present)
      mean_miles           (sum_miles / trips, if present)
      mean_base_fare       (sum_base_fare / trips, if present)
    """
    panel = load_monthly_panel(operator)
    # Sum every numeric column that begins with sum_, n_, or is `trips`.
    sum_cols = [c for c in panel.columns
                if c in ('trips',) or c.startswith('sum_') or c.startswith('n_')]
    g = (panel.groupby(['operator', 'month', 'zone_id'])[sum_cols]
              .sum().reset_index())
    # Derived rates.
    if 'sum_response_s' in g.columns and 'n_with_response' in g.columns:
        g['mean_response_s'] = np.where(
            g['n_with_response'] > 0,
            g['sum_response_s'] / g['n_with_response'], np.nan)
    if 'sum_driver_pay' in g.columns:
        g['mean_driver_pay'] = np.where(g['trips'] > 0,
                                        g['sum_driver_pay'] / g['trips'], np.nan)
    if 'sum_miles' in g.columns:
        g['mean_miles'] = np.where(g['trips'] > 0,
                                   g['sum_miles'] / g['trips'], np.nan)
    if 'sum_base_fare' in g.columns:
        g['mean_base_fare'] = np.where(g['trips'] > 0,
                                       g['sum_base_fare'] / g['trips'], np.nan)
    return g


def load_monthly_city(operator: str) -> pd.DataFrame:
    """Per month, citywide totals plus a few derived means."""
    z = load_monthly_zone(operator)
    sum_cols = [c for c in z.columns
                if c == 'trips' or c.startswith('sum_') or c.startswith('n_')]
    g = (z.groupby(['operator', 'month'])[sum_cols].sum().reset_index())
    if 'sum_response_s' in g.columns and 'n_with_response' in g.columns:
        g['mean_response_s'] = np.where(
            g['n_with_response'] > 0,
            g['sum_response_s'] / g['n_with_response'], np.nan)
    if 'sum_driver_pay' in g.columns:
        g['mean_driver_pay'] = np.where(g['trips'] > 0,
                                        g['sum_driver_pay'] / g['trips'], np.nan)
        if 'sum_miles' in g.columns:
            g['driver_pay_per_mile'] = np.where(
                g['sum_miles'] > 0,
                g['sum_driver_pay'] / g['sum_miles'], np.nan)
        if 'sum_time_s' in g.columns:
            g['driver_pay_per_hour'] = np.where(
                g['sum_time_s'] > 0,
                g['sum_driver_pay'] / (g['sum_time_s'] / 3600.0), np.nan)
    if 'sum_base_fare' in g.columns:
        g['mean_base_fare'] = np.where(g['trips'] > 0,
                                       g['sum_base_fare'] / g['trips'], np.nan)
        if 'sum_driver_pay' in g.columns:
            # Operator margin per trip = base fare minus driver pay, normalised
            # by base fare. Note this is gross of operator costs; treat as a
            # rough proxy not an accounting truth.
            g['operator_margin_share'] = np.where(
                g['sum_base_fare'] > 0,
                1.0 - (g['sum_driver_pay'] / g['sum_base_fare']), np.nan)
    return g
