"""
Surgical merge: add the net trade columns into Master_v2_observed.csv.

Same pattern as merge_net_into_parquet.py but targeting the observed-only
CSV that M5 reads. Net columns are observed data (computed from imports +
exports), not imputed.
"""
from pathlib import Path
import pandas as pd
import shutil

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
OBS    = EXT / 'intermediary' / 'Master_v2_observed.csv'
TRADE  = EXT / 'intermediary' / 'trade_metrics.csv'
BACKUP = OBS.with_suffix('.csv.bak_before_net_merge')

if not BACKUP.exists():
    shutil.copy2(OBS, BACKUP)
    print(f'Backup: {BACKUP}')
else:
    print(f'Backup exists: {BACKUP}')

obs   = pd.read_csv(OBS)
trade = pd.read_csv(TRADE).rename(columns={'reporterISO': 'Country Code', 'period': 'Year'})
trade['Year'] = trade['Year'].astype(int)

print(f'Observed: {obs.shape}')
print(f'Trade:    {trade.shape}')

# All net + extra observed columns from trade_metrics
net_cols = [c for c in trade.columns if c.endswith('_net')]
extra_cols = ['total_imports_usd', 'resource_herfindahl_net']
extra_present = [c for c in extra_cols if c in trade.columns and c not in net_cols]

# Drop any colliding existing columns in obs
collisions = [c for c in net_cols + extra_present if c in obs.columns]
if collisions:
    print(f'Dropping {len(collisions)} pre-existing collisions: {collisions[:5]}{"..." if len(collisions)>5 else ""}')
    obs = obs.drop(columns=collisions)

merge_cols = ['Country Code', 'Year'] + net_cols + extra_present
trade_subset = trade[merge_cols].copy()

merged = obs.merge(trade_subset, on=['Country Code', 'Year'], how='left')
assert len(merged) == len(obs), f'Row count changed: {len(obs)} -> {len(merged)}'

# Quick coverage check
print(f'\nCoverage of new columns in merged observed CSV:')
for c in ['wide_resource_share_net', 'coal_share_net', 'crude_oil_share_net',
          'refined_oil_share_net', 'gas_share_net', 'ores_share_net',
          'base_metals_share_net']:
    if c in merged.columns:
        frac = merged[c].notna().mean()
        print(f'  {c:35s}  {frac*100:5.1f}%')

merged.to_csv(OBS, index=False)
print(f'\nWrote: {OBS}')
print(f'New shape: {merged.shape}  (added {merged.shape[1] - obs.shape[1] + len(collisions)} columns)')
print('\nM5 should now run under MODE=net.')
