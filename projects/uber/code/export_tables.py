#!/usr/bin/env python3.10
"""
export_tables.py
================

Dump every chart's underlying data as a tidy CSV into outputs/tables/.
Lets Claude (or anyone) interpret results without opening interactive HTML.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _panel_loader import (  # noqa: E402
    classify_zones, load_monthly_zone, load_monthly_city, OUT_DIR,
    coverage_report,
)

TABLES_DIR = OUT_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def gini(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= 0)]
    if len(v) < 2 or v.sum() == 0:
        return np.nan
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * v)) / (n * cum[-1])
                 - (n + 1) / n)


def main():
    print("Exporting tidy tables...")
    zones = classify_zones()
    uber_z = load_monthly_zone('uber')
    lyft_z = load_monthly_zone('lyft')
    uber_z = uber_z.merge(zones[['zone_id', 'borough', 'zone_class']],
                          on='zone_id', how='left')
    lyft_z = lyft_z.merge(zones[['zone_id', 'borough', 'zone_class']],
                          on='zone_id', how='left')
    # EWR fold into Queens for borough totals.
    for df in (uber_z, lyft_z):
        df.loc[df['borough'].eq('EWR'), 'borough'] = 'Queens'

    # ── 1. Monthly trips by zone class, both operators ─────────────────
    def by_class(zdf, op):
        cols = ['trips']
        for c in ('sum_base_fare', 'sum_driver_pay', 'sum_tips',
                  'sum_cbd_fee', 'sum_miles', 'sum_time_s'):
            if c in zdf.columns:
                cols.append(c)
        g = (zdf.groupby(['month', 'zone_class'])[cols].sum().reset_index())
        g.insert(0, 'operator', op)
        return g
    by_cls = pd.concat([by_class(uber_z, 'uber'), by_class(lyft_z, 'lyft')],
                      ignore_index=True)
    by_cls.to_csv(TABLES_DIR / 'monthly_by_class.csv', index=False)

    # ── 2. CBD and buffer shares over time ─────────────────────────────
    def shares(zdf, op):
        g = zdf.groupby(['month', 'zone_class'])['trips'].sum().unstack(fill_value=0)
        man = g.get('cbd', 0) + g.get('buffer', 0) + g.get('upper_manhattan', 0)
        total = g.sum(axis=1)
        out = pd.DataFrame({
            'month': g.index,
            'operator': op,
            'cbd_share_of_city': g.get('cbd', 0) / total,
            'buffer_share_of_manhattan': g.get('buffer', 0) / man,
            'airport_share_of_city': g.get('airport', 0) / total,
            'outer_borough_share': g.get('outer_borough', 0) / total,
        })
        return out
    pd.concat([shares(uber_z, 'uber'), shares(lyft_z, 'lyft')],
              ignore_index=True).to_csv(TABLES_DIR / 'shares_over_time.csv',
                                        index=False)

    # ── 3. Pass-through (fare and pay per trip) by zone class, Uber ────
    pt = (uber_z.groupby(['month', 'zone_class'])
                .agg(trips=('trips', 'sum'),
                     sum_fare=('sum_base_fare', 'sum'),
                     sum_pay=('sum_driver_pay', 'sum'),
                     sum_cbd=('sum_cbd_fee', 'sum')
                         if 'sum_cbd_fee' in uber_z.columns
                         else ('trips', 'first'))
                .reset_index())
    pt['fare_per_trip'] = pt['sum_fare'] / pt['trips']
    pt['pay_per_trip'] = pt['sum_pay'] / pt['trips']
    if 'sum_cbd_fee' in uber_z.columns:
        pt['cbd_fee_per_trip'] = pt['sum_cbd'] / pt['trips']
    pt.to_csv(TABLES_DIR / 'passthrough_by_class.csv', index=False)

    # ── 4. Market share over time, citywide and by borough ─────────────
    keep = ['month', 'zone_id', 'borough', 'trips']
    u = uber_z[keep].rename(columns={'trips': 'u'})
    l = lyft_z[keep].rename(columns={'trips': 'l'})
    combo = u.merge(l, on=['month', 'zone_id', 'borough'], how='outer').fillna(0)
    city = (combo.groupby('month').agg(u=('u', 'sum'), l=('l', 'sum'))
                  .assign(uber_share=lambda d: d['u'] / (d['u'] + d['l']))
                  .reset_index())
    city.to_csv(TABLES_DIR / 'market_share_citywide.csv', index=False)

    by_b = (combo.groupby(['month', 'borough'])
                  .agg(u=('u', 'sum'), l=('l', 'sum')).reset_index())
    by_b['uber_share'] = by_b['u'] / (by_b['u'] + by_b['l'])
    by_b.to_csv(TABLES_DIR / 'market_share_borough.csv', index=False)

    # ── 5. Zone-level persistence: top and bottom of Uber dominance ────
    combo['total'] = combo['u'] + combo['l']
    combo['uber_share'] = combo['u'] / combo['total']
    per = (combo[combo['total'] > 100]
           .assign(um=lambda d: (d['uber_share'] > 0.5).astype(int))
           .groupby('zone_id').agg(uber_maj_pct=('um', 'mean'),
                                   months=('um', 'size'),
                                   mean_total=('total', 'mean'))
           .reset_index()
           .merge(zones[['zone_id', 'zone_name', 'borough', 'zone_class']],
                  on='zone_id', how='left')
           .sort_values('uber_maj_pct'))
    per.to_csv(TABLES_DIR / 'zone_persistence.csv', index=False)

    # ── 6. Driver economics: monthly citywide ──────────────────────────
    city_drv = load_monthly_city('uber')
    keep_cols = [c for c in ['month', 'trips', 'mean_driver_pay',
                             'driver_pay_per_mile', 'driver_pay_per_hour',
                             'operator_margin_share', 'mean_base_fare']
                 if c in city_drv.columns]
    city_drv[keep_cols].to_csv(TABLES_DIR / 'driver_economics_citywide.csv',
                                index=False)

    # ── 7. Driver pay per hour by borough ──────────────────────────────
    if 'sum_driver_pay' in uber_z.columns and 'sum_time_s' in uber_z.columns:
        by = (uber_z.groupby(['month', 'borough'])
                     .agg(pay=('sum_driver_pay', 'sum'),
                          seconds=('sum_time_s', 'sum'),
                          tips=('sum_tips', 'sum'),
                          fare=('sum_base_fare', 'sum'),
                          trips=('trips', 'sum')).reset_index())
        by['pay_per_hour'] = by['pay'] / (by['seconds'] / 3600.0)
        by['pay_per_trip'] = by['pay'] / by['trips']
        by['tip_share'] = by['tips'] / (by['pay'] + by['tips'])
        by['margin_share'] = 1 - by['pay'] / by['fare']
        by.to_csv(TABLES_DIR / 'driver_economics_borough.csv', index=False)

    # ── 8. Cross-zone Gini of pay per trip over time ───────────────────
    if 'mean_driver_pay' in uber_z.columns:
        rows = []
        for month, g in uber_z.groupby('month'):
            sub = g[g['trips'] >= 50].dropna(subset=['mean_driver_pay'])
            if len(sub) < 20:
                continue
            rows.append({'month': month, 'gini_pay_per_trip': gini(sub['mean_driver_pay'].values),
                         'n_zones': len(sub)})
        pd.DataFrame(rows).to_csv(TABLES_DIR / 'gini_pay_per_trip.csv',
                                  index=False)

    # ── 9. Spatial concentration trajectory: Gini of trips per zone ────
    rows = []
    for op, zdf in [('uber', uber_z), ('lyft', lyft_z)]:
        for month, g in zdf.groupby('month'):
            v = g['trips'].values
            if v.sum() <= 0 or len(v) < 20:
                continue
            v_sorted = np.sort(v)
            top10 = v_sorted[-10:].sum() / v.sum()
            top3 = v_sorted[-3:].sum() / v.sum()
            airport = g[g['zone_class'].eq('airport')]['trips'].sum() / g['trips'].sum()
            rows.append({
                'month': month, 'operator': op,
                'gini_trips_per_zone': gini(v),
                'top10_share': top10,
                'top3_share': top3,
                'airport_share': airport,
                'n_active_zones': int((v > 0).sum()),
            })
    pd.DataFrame(rows).to_csv(TABLES_DIR / 'concentration_monthly.csv',
                              index=False)

    print(f"\nTables written to {TABLES_DIR}:")
    for f in sorted(TABLES_DIR.glob('*.csv')):
        n = sum(1 for _ in open(f))
        print(f"  {f.name:42s}  {n - 1:>6,} rows  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
