"""
Patch e4_ml.ipynb with the MODE switch (gross vs net).

Mirrors the e3/e5 patches:
  - Cell 2: MODE constant + resolve_col + _suffix helpers
  - Cell 4: route resource-share entries in FEATURE_COLS through resolve_col;
            also build the post2019_x_*_share_net COVID interactions inline
            (same as e5) so the columns exist when MODE='net'
  - Cells that save artifacts: filenames wrapped in _suffix()
    - Cell 10 (forecast results)        -> e4_results[_net].csv
    - Cell 12 (forecast pooled)         -> e4_results_pooled[_net].csv
    - Cell 14 (SHAP forecast)           -> e4_shap[_net].csv
    - Cell 18 (feature_importance PNG)  -> feature_importance[_net].png
                                           predictions_scatter[_net].png
    - Cell 21 (structural everything)   -> e4_results_structural[_net].csv
                                           e4_results_structural_pooled[_net].csv
                                           e4_shap_structural[_net].csv
                                           forecast_vs_structural_r2[_net].png

Graphics folder also routed through MODE subfolder: Graphics/NB4/{gross,net}/
"""
import json
import shutil
import re
from pathlib import Path

NB = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e4_ml.ipynb')
BAK = NB.with_suffix('.ipynb.bak_before_mode_switch')

if not BAK.exists():
    shutil.copy2(NB, BAK)
    print(f'Backup: {BAK}')

with open(NB) as f:
    nb = json.load(f)


def set_cell(idx, src):
    nb['cells'][idx]['source'] = src.splitlines(keepends=True)
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None


# ============================================================
# Cell 2: Setup + MODE switch + helpers
# ============================================================
CELL_2 = '''\
import sys, time, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
import _config as cfg
from _mice_pool import iter_imputations, n_imputations

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGB = True
    print('XGBoost available')
except ImportError:
    HAS_XGB = False
    print('XGBoost not available (pip install xgboost --break-system-packages)')

try:
    import shap
    HAS_SHAP = True
    print('SHAP available')
except ImportError:
    HAS_SHAP = False
    print('SHAP not available (pip install shap --break-system-packages)')


# ----------------------------------------------------------------
# MODE switch: 'gross' uses *_share columns, 'net' uses *_share_net.
# Override via env var or by editing the line below.
# ----------------------------------------------------------------
MODE = os.environ.get('CLUSTER_MODE', 'gross')
assert MODE in ('gross', 'net'), f"Unknown MODE={MODE!r}; must be 'gross' or 'net'"
print(f'MODE = {MODE}')


def resolve_col(name):
    """Map a gross share column name to its net counterpart when MODE='net'."""
    if MODE == 'gross':
        return name
    return name + '_net'


def _suffix(name):
    """Append _net before the file extension when MODE='net'."""
    if MODE == 'gross':
        return name
    p = Path(name)
    return f'{p.stem}_net{p.suffix}'


EXT  = Path('.').resolve()
INTER = EXT / 'intermediary'
# Route Graphics through MODE subfolder so gross/net coexist
GRAPHICS = EXT / 'Graphics' / 'NB4' / MODE
GRAPHICS.mkdir(parents=True, exist_ok=True)

print(f'Working dir: {EXT}')
print(f'Output dir:  {GRAPHICS}')
print(f'M imputations available: {n_imputations()}')
'''
set_cell(2, CELL_2)
print('Cell 2 patched (MODE + helpers + GRAPHICS routing)')


# ============================================================
# Cell 4: feature engineering + FEATURE_COLS via resolve_col + net COVID interactions
# ============================================================
CELL_4 = '''\
# COVID sub-codes used for interactions
COVID_INTERACTION_CODES = ['coal', 'crude_oil', 'refined_oil', 'gas', 'ores', 'base_metals']


def engineer_features(panel):
    """Apply feature engineering identical to each of M panels.

    Under MODE='net' this also constructs post2019_x_*_share_net interaction
    columns (matching the e5 logic) since they aren't pre-computed in the
    parquet for net.
    """
    df = panel.copy()
    df = df.sort_values(['Country Code', 'Year'])

    # Rolling 5-year macro controls
    if 'Inflation, consumer prices (annual %)' in df.columns:
        df['Inflation_roll5'] = (
            df.groupby('Country Code')['Inflation, consumer prices (annual %)']
              .transform(lambda x: x.rolling(5, min_periods=3).mean())
        )
    if 'Real interest rate (%)' in df.columns:
        df['RealRate_roll5'] = (
            df.groupby('Country Code')['Real interest rate (%)']
              .transform(lambda x: x.rolling(5, min_periods=3).mean())
        )

    # Resource concentration via HHI on trade chapter shares.
    # Under MODE='net' use the _net share columns; clip negatives at 0 since
    # HHI on a basket with negative components is ill-defined.
    hs_share_cols = [c for c in df.columns
                     if c.startswith('hs')
                     and c.endswith('_share_net' if MODE == 'net' else '_share')
                     and (c.endswith('_share_net') == (MODE == 'net'))]
    # Defensive: in net mode we want columns ending in _share_net, not _share
    if MODE == 'net':
        hs_share_cols = [c for c in df.columns if c.startswith('hs') and c.endswith('_share_net')]
    else:
        hs_share_cols = [c for c in df.columns
                         if c.startswith('hs') and c.endswith('_share')
                         and not c.endswith('_share_net')]

    if hs_share_cols:
        clipped = df[hs_share_cols].clip(lower=0)
        share_sum = clipped.sum(axis=1).replace(0, np.nan)
        normalised = clipped.div(share_sum, axis=0)
        df['Resource_HHI_trade'] = (normalised ** 2).sum(axis=1)

    # Log transforms (mode-invariant)
    for raw, log in [
        ('Human capital index', 'log_HCI'),
        ('Gross fixed capital formation, all, Constant prices, Percent of GDP', 'log_GFCF'),
        ('GDP per capita (constant prices, PPP)', 'log_GDPpc'),
        ('Population', 'log_Pop'),
    ]:
        if raw in df.columns:
            df[log] = np.log(df[raw].clip(lower=0) + 1)

    # Lagged ECI
    df['L1_ECI'] = df.groupby('Country Code')['Economic Complexity Index'].shift(1)
    df['ECI_delta'] = df['Economic Complexity Index'] - df['L1_ECI']

    # Build post2019_x_*_share_net interactions when MODE='net'
    # (Gross variants are pre-computed in the parquet by e2.)
    if MODE == 'net':
        for code in COVID_INTERACTION_CODES:
            sub_col = f'{code}_share_net'
            int_col = f'post2019_x_{code}_share'
            if sub_col in df.columns and 'post_2019' in df.columns:
                df[int_col] = df['post_2019'] * df[sub_col]

    return df


# Feature list. Resource-share entries route through resolve_col.
# COVID interaction names stay literal (their source columns pick up the
# _net suffix in engineer_features above).
FEATURE_COLS = [
    # Trade-side resource exposure (headline)
    resolve_col('wide_resource_share'),
    resolve_col('hydrocarbon_share'),
    resolve_col('ores_share'),
    resolve_col('base_metals_share'),
    resolve_col('precious_share'),
    'Resource_HHI_trade',
    # COVID interactions
    'post2019_x_coal_share',
    'post2019_x_crude_oil_share',
    'post2019_x_refined_oil_share',
    'post2019_x_gas_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
    # Macro
    'log_GDPpc',
    'log_Pop',
    'log_HCI',
    'log_GFCF',
    'Trade (% of GDP)',
    'Domestic credit to private sector (% of GDP)',
    # Sectoral
    'Agriculture',
    'Industry',
    'Manufacturing',
    'Services',
    'Urban population (% of total population)',
    # Governance
    'Political stability — estimate',
    'Rule of law index',
    'Political corruption index',
    # Monetary
    'Inflation_roll5',
    'RealRate_roll5',
    # Lagged ECI (autoregressive component)
    'L1_ECI',
]

# Verify on first imputation
imp0 = engineer_features(next(iter_imputations())[1])
present = [c for c in FEATURE_COLS if c in imp0.columns]
missing = [c for c in FEATURE_COLS if c not in imp0.columns]
print(f'Features available ({MODE}): {len(present)} / {len(FEATURE_COLS)}')
if missing:
    print(f'  MISSING: {missing}')
'''
set_cell(4, CELL_4)
print('Cell 4 patched (engineer_features + FEATURE_COLS via resolve_col)')


# ============================================================
# Save-path patching: wrap CSV / PNG filenames with _suffix()
# ============================================================
# Filenames that the notebook writes that need MODE-aware naming
SAVE_FILES = [
    'e4_results.csv',
    'e4_results_pooled.csv',
    'e4_shap.csv',
    'e4_predictions.csv',
    'e4_results_structural.csv',
    'e4_results_structural_pooled.csv',
    'e4_shap_structural.csv',
]
SAVE_PNG = [
    'feature_importance.png',
    'predictions_scatter.png',
    'forecast_vs_structural_r2.png',
]

patched_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    new = src

    for fname in SAVE_FILES + SAVE_PNG:
        # Replace 'fname' with _suffix('fname') when used as a literal string
        # in to_csv() or savefig() or similar paths.
        # Match unescaped literal occurrences.
        if f"'{fname}'" in new and f"_suffix('{fname}')" not in new:
            new = new.replace(f"'{fname}'", f"_suffix('{fname}')")
        if f'"{fname}"' in new and f'_suffix("{fname}")' not in new:
            new = new.replace(f'"{fname}"', f'_suffix("{fname}")')

    if new != src:
        set_cell(i, new)
        patched_cells.append(i)

print(f'Save-path-patched cells: {patched_cells}')


with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'\\nSaved: {NB}')
print()
print('========================================================')
print('To run gross: open, restart kernel, Run All (default MODE=gross).')
print('To run net:   set CLUSTER_MODE=net OR edit cell 2.')
print('Outputs:')
print('  intermediary/e4_*.csv (gross)')
print('  intermediary/e4_*_net.csv (net)')
print('  Graphics/NB4/gross/*.png')
print('  Graphics/NB4/net/*.png')
print('========================================================')
