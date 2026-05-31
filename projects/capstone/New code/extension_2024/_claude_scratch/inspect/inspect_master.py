"""Read-only inspection of Master.csv, trade_metrics.csv (stale), and PWT 11.0."""
import pandas as pd
import os

ROOT = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code'
EXT  = f'{ROOT}/extension_2024'
PWT  = '/Users/leoss/Desktop/GitHub/capstone-client-submission/main_analysis/rawdata/pwt110.xlsx'

print('=' * 70)
print('Master.csv')
print('=' * 70)
M = pd.read_csv(f'{ROOT}/intermediary/Master.csv')
print(f'rows: {len(M)}  cols: {len(M.columns)}')

country_col = next((c for c in M.columns if c.lower() in
    ('country code','country_code','countrycode','iso','iso3','code')), None)
year_col = next((c for c in M.columns if c.lower() in ('year','yr')), None)
print(f'country col: {country_col!r}  year col: {year_col!r}')
if country_col:
    print(f'  countries: {M[country_col].nunique()}')
if year_col:
    print(f'  year range: {M[year_col].min()} - {M[year_col].max()}')
    print(f'  rows by year (last 6): {M.groupby(year_col).size().tail(6).to_dict()}')

print()
print('--- all columns ---')
for i, c in enumerate(M.columns):
    nn = M[c].notna().sum()
    print(f'  [{i:2d}] {c!r:50s}  dtype={str(M[c].dtype):8s}  notna={nn}/{len(M)}')
