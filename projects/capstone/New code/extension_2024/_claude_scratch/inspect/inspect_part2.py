"""Inspect trade_metrics.csv schema and PWT 11.0 year coverage."""
import pandas as pd, os

EXT = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024'
PWT = '/Users/leoss/Desktop/GitHub/capstone-client-submission/main_analysis/rawdata/pwt110.xlsx'

print('=== trade_metrics.csv (stale; schema only) ===')
T = pd.read_csv(f'{EXT}/intermediary/trade_metrics.csv')
print(f'rows={len(T)} cols={len(T.columns)} '
      f'countries={T["Country Code"].nunique()} years={T["Year"].min()}-{T["Year"].max()}')
print('--- columns ---')
for i, c in enumerate(T.columns):
    print(f'  [{i:2d}] {c}')

print()
print('=== PWT 11.0 ===')

if os.path.exists(PWT):
    P = pd.read_excel(PWT, sheet_name='Data')
    print(f'rows={len(P)} cols={len(P.columns)} '
          f'countries={P["countrycode"].nunique()} years={P["year"].min()}-{P["year"].max()}')
    last = P.groupby('year').size().tail(10).to_dict()
    print(f'rows per year (last 10): {last}')
    print('--- columns sample ---')
    for i, c in enumerate(P.columns[:25]):
        print(f'  [{i:2d}] {c}')
    print(f'  (... {len(P.columns) - 25} more cols not shown)')
else:
    print(f'PWT NOT FOUND at {PWT}')
