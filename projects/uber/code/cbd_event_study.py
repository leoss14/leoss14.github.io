"""
cbd_event_study.py

Canonical event-study specification on the CBD congestion fee (5 Jan 2025).

For computational tractability we aggregate trip-level data to
(pickup_zone × calendar_month × treated_status) cells, weighted by
trip count. With pickup zone and calendar month fixed effects, the
coefficient estimates on event-time × treated dummies are numerically
identical to the trip-level FE regression (subject to a benign rescaling
of weights); standard errors clustered at pickup-zone level remain valid.

Treated unit definition: a trip is 'treated' if PU or DO zone is in the
36-zone CBD set. We aggregate to (PU_zone, month, did_either_end_touch_CBD)
cells so the same pickup zone produces a treated cell (trips that also
touched CBD on dropoff) and a control cell (trips with neither end in CBD)
in each month.

Model (per operator):
    log_fare_imt = α + Σ_τ β_τ · 1[event_time = τ] · treated_i
                   + γ_PU + δ_m + ε_imt
    τ ∈ [-12, +12] \ {-1, 0}; ref = -1 (Dec 2024); event month dropped.

Tests:
  • Pre-trends F-test: H0: β_{-12..-2} = 0
  • Static DiD (single post indicator) for direct comparison
  • Post-period mean ATT
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from _panel_loader import classify_zones

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
SAMPLE = ROOT / 'outputs' / 'tables' / 'trip_sample_full.parquet'
TABLES = ROOT / 'outputs' / 'tables'

WINDOW_PRE = 12
WINDOW_POST = 12
LEAD_REF = -1

# ─── 1. Load and prepare trip data ──────────────────────────────────────
print('Loading trip sample ...')
cols = ['operator', 'pickup_datetime', 'PULocationID', 'DOLocationID',
        'base_passenger_fare', 'sampling_weight']
df = pd.read_parquet(SAMPLE, columns=cols)
print(f'  loaded {len(df):,} trips')

zones = classify_zones()
cbd_zones = set(zones.loc[zones['zone_class'] == 'cbd', 'zone_id'].astype(int))
print(f'  CBD zones: {len(cbd_zones)}')

df['month'] = df['pickup_datetime'].dt.to_period('M').dt.to_timestamp()
df['event_time'] = ((df['month'].dt.year - 2025) * 12
                    + (df['month'].dt.month - 1))
df['treated'] = (df['PULocationID'].isin(cbd_zones)
                 | df['DOLocationID'].isin(cbd_zones)).astype(int)

# Filter window, drop event month (Jan 2025), positive fares
df = df[(df['event_time'] >= -WINDOW_PRE)
        & (df['event_time'] <= WINDOW_POST)
        & (df['event_time'] != 0)
        & (df['base_passenger_fare'] > 0)
        & df['base_passenger_fare'].notna()].copy()
df['log_fare'] = np.log(df['base_passenger_fare'])
print(f'  after filters: {len(df):,} trips')

# ─── 2. Aggregate to (PU_zone, month, treated) cells per operator ────────
def aggregate(sub: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean of log_fare per cell; cell weight is sum of sampling weights."""
    g = sub.groupby(['PULocationID', 'month', 'treated'], observed=True)
    cells = pd.DataFrame({
        'log_fare': (g.apply(lambda x: np.average(x['log_fare'],
                                                  weights=x['sampling_weight']))),
        'cell_weight': g['sampling_weight'].sum(),
        'n_trips': g.size(),
    }).reset_index()
    return cells

def tname(tau: int) -> str:
    """Safe column name for event-time tau (patsy can't parse - or + in names)."""
    return f'D_m{abs(tau)}' if tau < 0 else f'D_p{tau}'


# ─── 3. Event-time dummies and regression per operator ───────────────────
results = {}
event_times = [t for t in range(-WINDOW_PRE, WINDOW_POST + 1)
               if t != 0 and t != LEAD_REF]

for op in ['Uber', 'Lyft']:
    sub = df[df['operator'].str.lower() == op.lower()].copy()
    print(f'\n[{op}] aggregating {len(sub):,} trips ...')
    cells = aggregate(sub)
    cells['event_time'] = ((cells['month'].dt.year - 2025) * 12
                           + (cells['month'].dt.month - 1))
    print(f'  cells: {len(cells):,}'
          f'  (zones: {cells["PULocationID"].nunique()}, '
          f'months: {cells["month"].nunique()})')

    # Drop cells with very few trips (noisy)
    cells = cells[cells['n_trips'] >= 30].copy()
    print(f'  after n>=30 filter: {len(cells):,} cells')

    # Build dummies with safe column names
    for tau in event_times:
        cells[tname(tau)] = ((cells['event_time'] == tau)
                              & (cells['treated'] == 1)).astype(int)

    cells['PU_zone'] = cells['PULocationID'].astype('category')
    cells['cal_month'] = cells['month'].dt.strftime('%Y-%m').astype('category')

    coef_cols = [tname(t) for t in event_times]
    formula = ('log_fare ~ ' + ' + '.join(coef_cols)
               + ' + C(PU_zone) + C(cal_month)')

    print('  fitting WLS with zone + month FE ...')
    model = smf.wls(formula, data=cells, weights=cells['cell_weight'])
    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': cells['PULocationID'].astype(int)})

    # Extract event-time coefs
    coefs = []
    for tau in range(-WINDOW_PRE, WINDOW_POST + 1):
        if tau == LEAD_REF:
            coefs.append({'event_time': tau, 'coef': 0.0,
                          'se': 0.0, 'ci_low': 0.0, 'ci_high': 0.0,
                          'operator': op, 'reference': True})
        elif tau == 0:
            continue
        else:
            name = tname(tau)
            b = res.params[name]
            se = res.bse[name]
            coefs.append({
                'event_time': tau, 'coef': b, 'se': se,
                'ci_low': b - 1.96 * se, 'ci_high': b + 1.96 * se,
                'operator': op, 'reference': False,
            })

    # Pre-trends F-test (joint test on pre-event coefs, excluding -1 reference)
    pre_names = [tname(t) for t in range(-WINDOW_PRE, 0) if t != LEAD_REF]
    f_test = res.f_test(' = '.join(pre_names) + ' = 0')

    # Post-period ATT
    post_names = [tname(t) for t in range(1, WINDOW_POST + 1)]
    att = np.mean([res.params[n] for n in post_names])

    # Static DiD (one post indicator)
    cells['post'] = (cells['event_time'] > 0).astype(int)
    cells['did'] = cells['post'] * cells['treated']
    m2 = smf.wls('log_fare ~ did + C(PU_zone) + C(cal_month)',
                 data=cells, weights=cells['cell_weight']).fit(
        cov_type='cluster',
        cov_kwds={'groups': cells['PULocationID'].astype(int)})

    print(f'\n  [{op}] Results:')
    print(f'    Pre-trends F-test:  F = {float(f_test.fvalue):8.2f},'
          f' p = {float(f_test.pvalue):.4f}')
    print(f'    Post-period ATT:    {att:+.4f} log ({att*100:+.2f}%)')
    print(f'    Static DiD coef:    {m2.params["did"]:+.4f} log'
          f' ({m2.params["did"]*100:+.2f}%)')
    print(f'    Static DiD SE:      {m2.bse["did"]:.4f}')
    print(f'    Static DiD p-value: {m2.pvalues["did"]:.4f}')

    results[op] = {
        'coefs': pd.DataFrame(coefs),
        'f_test': (float(f_test.fvalue), float(f_test.pvalue)),
        'att': att,
        'static_did': (m2.params['did'], m2.bse['did'], m2.pvalues['did']),
    }

# ─── 4. Save outputs ────────────────────────────────────────────────────
all_coefs = pd.concat([results['Uber']['coefs'], results['Lyft']['coefs']],
                     ignore_index=True)
all_coefs.to_csv(TABLES / 'cbd_event_study_coefs.csv', index=False)

tests_rows = []
for op in ['Uber', 'Lyft']:
    f, p = results[op]['f_test']
    did, se, pval = results[op]['static_did']
    tests_rows.append({
        'operator': op,
        'pre_trends_F': f,
        'pre_trends_p': p,
        'post_ATT_log': results[op]['att'],
        'post_ATT_pct': results[op]['att'] * 100,
        'static_DiD_log': did,
        'static_DiD_pct': did * 100,
        'static_DiD_SE': se,
        'static_DiD_p': pval,
    })
pd.DataFrame(tests_rows).to_csv(TABLES / 'cbd_event_study_tests.csv', index=False)

print('\n=======================================================')
print('Wrote:')
print('  outputs/tables/cbd_event_study_coefs.csv')
print('  outputs/tables/cbd_event_study_tests.csv')
print('=======================================================')
