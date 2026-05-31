"""ECI file year coverage + locate the PWT cap in e1_data_pull.ipynb."""
import pandas as pd, json

ECI_PATH = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/intermediary/rawdata/growth_proj_eci_rankings-1.csv'

print('=== growth_proj_eci_rankings-1.csv ===')
E = pd.read_csv(ECI_PATH)
print(f'rows={len(E)} cols={len(E.columns)} '
      f'countries={E["country_iso3_code"].nunique()} '
      f'years={E["year"].min()}-{E["year"].max()}')
print('non-null counts:')
print(E[['eci_sitc','eci_hs92','eci_hs12','growth_proj']].notna().sum().to_string())
print('eci_hs92 year range (non-null):')
sub = E[E['eci_hs92'].notna()]
print(f'  {sub["year"].min()}-{sub["year"].max()}  '
      f'countries={sub["country_iso3_code"].nunique()}')
print('rows per year (last 8) for eci_hs92:')
print(sub.groupby('year').size().tail(8).to_string())
