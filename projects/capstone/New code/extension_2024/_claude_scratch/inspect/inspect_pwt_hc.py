"""Check non-null PWT vars by year for 2018-2023 — does HCI actually stop at 2019?"""
import pandas as pd
PWT = '/Users/leoss/Desktop/GitHub/capstone-client-submission/main_analysis/rawdata/pwt110.xlsx'
P = pd.read_excel(PWT, sheet_name='Data')
cols = ['hc','cn','ctfp','cwtfp','csh_c','csh_i','csh_g','delta']
print(f'PWT 11.0 file: {len(P)} rows, year range {P["year"].min()}-{P["year"].max()}')
print(f'countries: {P["countrycode"].nunique()}')
print()
print('Non-null counts by year (last 6 years) for each PWT variable used:')
last6 = P[P['year'] >= 2018]
print(last6.groupby('year')[cols].count().to_string())
print()
print('Compare to total countries per year:')
print(last6.groupby('year').size().to_string())
