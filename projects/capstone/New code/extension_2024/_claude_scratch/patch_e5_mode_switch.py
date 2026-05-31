"""
Patch e5_regressions.ipynb with the MODE switch (gross vs net).

Mirrors the e3 patch:
  - Cell 2: MODE constant + resolve_col + _suffix helpers
  - Cell 4: route CENTER_VARS through resolve_col; build *_x_*_share_net
           COVID interactions inline in transform_panel for MODE='net'
  - Cell 6: route resource-share entries in CONTROLS_FULL,
           CONTROLS_PARSIMONIOUS, INTERACTIONS_PARSIMONIOUS through resolve_col
  - Cell 20: output filenames wrapped in _suffix()
  - Cell 22: KEY_VARS routed through resolve_col; PNG path wrapped in _suffix()

Default MODE = 'gross' so behaviour is unchanged unless CLUSTER_MODE=net is
set in the environment or cell 2 is edited.
"""
import json
import shutil
from pathlib import Path

NB = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e5_regressions.ipynb')
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
# Cell 2: Setup + MODE switch + resolve_col + _suffix helpers
# ============================================================
CELL_2 = '''\
import os, sys, time, warnings, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from types import SimpleNamespace

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, '.')
import _config as cfg
from _mice_pool import pool_scalars, iter_imputations, load_imputations, PooledScalar

import statsmodels.api as sm
from linearmodels.panel import PanelOLS

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
    # Net column naming: <stem>_net
    return name + '_net'


def _suffix(name):
    """Append _net before the file extension when MODE='net'."""
    if MODE == 'gross':
        return name
    p = Path(name)
    return f'{p.stem}_net{p.suffix}'


EXT  = Path('.').resolve()
INTER = EXT / 'intermediary'

print(f'Working dir: {EXT}')
print(f'Year range:  {cfg.YEAR_MIN}-{cfg.YEAR_MAX}')
'''
set_cell(2, CELL_2)
print('Cell 2 patched (MODE + helpers)')


# ============================================================
# Cell 4: Transforms. Route CENTER_VARS through resolve_col, build net COVID
# interactions inline.
# ============================================================
CELL_4 = '''\
# Common transforms applied identically to each of the M panels and to the
# observed-only panel. Mean-centering uses the grand sample mean within each
# panel (so the interaction coefficients are interpretable at the mean of the
# interacted variable).

LOG_VARS = [
    ('Human capital index', 'log_HCI'),
    ('Gross fixed capital formation, all, Constant prices, Percent of GDP', 'log_GFCF'),
    ('GDP per capita (constant prices, PPP)', 'log_GDPpc'),
    ('Population', 'log_Pop'),
]

# Variables to mean-center for interaction terms. The wide_resource_share entry
# routes through resolve_col so MODE='net' picks up wide_resource_share_net.
CENTER_VARS = ['log_HCI', 'log_GFCF', resolve_col('wide_resource_share')]

# Sub-codes used in COVID interactions. Routed through resolve_col so MODE='net'
# uses the net sub-code share columns.
COVID_INTERACTION_CODES = ['coal', 'crude_oil', 'refined_oil', 'gas', 'ores', 'base_metals']


def transform_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply log and centering transforms. Adds new columns, does not modify inputs.

    Under MODE='net' this also constructs the post2019_x_*_share_net interaction
    columns, since they're not pre-computed in the parquet (the parquet only
    has the gross variants from e2).
    """
    df = panel.copy()

    # Log transforms (log(1 + x) to handle zeros)
    for src, tgt in LOG_VARS:
        if src in df.columns:
            df[tgt] = np.log(df[src].clip(lower=0) + 1)
        else:
            print(f'  WARNING: source variable {src!r} not in panel; skipping {tgt}')

    # Mean centering for interaction terms
    for v in CENTER_VARS:
        if v in df.columns:
            df[f'{v}_c'] = df[v] - df[v].mean()

    # v2-style interactions (HCI x resource intensity, GFCF x resource intensity).
    # The wide_share variable name picks up the _net suffix under MODE='net'.
    wide_share = resolve_col('wide_resource_share')
    if 'log_HCI_c' in df.columns and f'{wide_share}_c' in df.columns:
        df['log_HCI_x_wide_share']  = df['log_HCI_c']  * df[f'{wide_share}_c']
    if 'log_GFCF_c' in df.columns and f'{wide_share}_c' in df.columns:
        df['log_GFCF_x_wide_share'] = df['log_GFCF_c'] * df[f'{wide_share}_c']

    # COVID interactions: pre-computed in e2 for the gross sub-codes; here we
    # construct the net variants on the fly when MODE='net'. The interaction
    # name itself does NOT get a _net suffix (so M1, M3, M5 spec lists can use
    # the same key in either mode and resolve_col only swaps the source column).
    if MODE == 'net':
        for code in COVID_INTERACTION_CODES:
            sub_col = f'{code}_share_net'
            int_col = f'post2019_x_{code}_share'
            if sub_col in df.columns and 'post_2019' in df.columns:
                df[int_col] = df['post_2019'] * df[sub_col]

    return df


# Sanity check on the first imputation
imp1 = next(iter_imputations())[1]
imp1 = transform_panel(imp1)
needed = ['log_HCI', 'log_GFCF', 'log_GDPpc', 'log_Pop',
          'log_HCI_x_wide_share', 'log_GFCF_x_wide_share',
          resolve_col('wide_resource_share'),
          resolve_col('hydrocarbon_share'),
          resolve_col('ores_share'),
          resolve_col('base_metals_share'),
          resolve_col('coal_share'),
          resolve_col('crude_oil_share'),
          resolve_col('refined_oil_share'),
          resolve_col('gas_share'),
          'post2019_x_coal_share',
          'post2019_x_crude_oil_share',
          'post2019_x_refined_oil_share',
          'post2019_x_gas_share',
          'post2019_x_ores_share',
          'post2019_x_base_metals_share']
missing = [c for c in needed if c not in imp1.columns]
print(f'After transform_panel ({MODE}): {len(imp1.columns)} cols')
if missing:
    print(f'  MISSING: {missing}')
else:
    print(f'  All needed columns present.')
'''
set_cell(4, CELL_4)
print('Cell 4 patched (CENTER_VARS + interaction construction)')


# ============================================================
# Cell 6: Specs. Route resource-share entries through resolve_col.
# ============================================================
CELL_6 = '''\
# Full set of controls used in Model 1 (the headline) and Model 2 (robust SE).
# Resource-share entries route through resolve_col so MODE='net' uses the
# net columns. COVID interaction names stay literal (their source columns
# pick up the _net suffix in transform_panel above).

CONTROLS_FULL = [
    # Trade and macro
    'log_GDPpc',
    'log_Pop',
    'Trade (% of GDP)',
    # Investment and human capital
    'log_HCI',
    'log_GFCF',
    'Domestic credit to private sector (% of GDP)',
    # Sectoral
    'Agriculture',
    'Industry',
    'Manufacturing',
    'Services',
    'Urban population (% of total population)',
    # Governance / institutions
    'Political stability — estimate',
    'Rule of law index',
    'Political corruption index',
    # Monetary
    'Inflation, consumer prices (annual %)',
    'Real interest rate (%)',
    # Fiscal
    'Government revenue',
    'Primary net lending, General government, Percent of GDP',
    # Resource exposure (trade-side; net or gross per MODE)
    resolve_col('wide_resource_share'),
    # COVID interactions (HS27 disaggregated)
    'post2019_x_coal_share',
    'post2019_x_crude_oil_share',
    'post2019_x_refined_oil_share',
    'post2019_x_gas_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
]

# Parsimonious set + interactions (Model 3)
CONTROLS_PARSIMONIOUS = [
    'log_HCI',
    'log_GFCF',
    'Political stability — estimate',
    'Rule of law index',
    'log_GDPpc',
    'Trade (% of GDP)',
    resolve_col('wide_resource_share'),
]

INTERACTIONS_PARSIMONIOUS = [
    'log_HCI_x_wide_share',
    'log_GFCF_x_wide_share',
    'post2019_x_coal_share',
    'post2019_x_crude_oil_share',
    'post2019_x_refined_oil_share',
    'post2019_x_gas_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
]

print(f'Full controls ({MODE}):        {len(CONTROLS_FULL)}')
print(f'Parsimonious + ints ({MODE}):  {len(CONTROLS_PARSIMONIOUS)} + {len(INTERACTIONS_PARSIMONIOUS)}')
'''
set_cell(6, CELL_6)
print('Cell 6 patched (CONTROLS via resolve_col)')


# ============================================================
# Cell 20: Save outputs with _suffix()
# ============================================================
CELL_20 = '''\
# Combine all 5 models side by side
all_results = pd.concat([m1, m2, m3, m4, m5], ignore_index=True)

# Wide pivot: variable in rows, (model x measure) in columns
def make_wide(df):
    pivot = df.pivot_table(index='variable', columns='model',
                            values=['beta', 'se', 'p'], aggfunc='first')
    return pivot

wide = make_wide(all_results)
print('Combined results (all 5 models):')
print(wide.to_string())

# Save (MODE-aware filenames via _suffix)
out_csv = INTER / _suffix('e5_results_long.csv')
all_results.to_csv(out_csv, index=False)
print(f'\\nSaved long results: {out_csv}')

out_wide = INTER / _suffix('e5_results_wide.csv')
wide.to_csv(out_wide)
print(f'Saved wide results:  {out_wide}')
'''
set_cell(20, CELL_20)
print('Cell 20 patched (save paths via _suffix)')


# ============================================================
# Cell 22: Coefficient plot. KEY_VARS routed through resolve_col; PNG via _suffix.
# ============================================================
CELL_22 = '''\
import matplotlib.pyplot as plt

# KEY_VARS routes the wide_resource_share entry through resolve_col so MODE='net'
# picks up wide_resource_share_net for the y-axis. Other entries (log_HCI, log_GFCF,
# governance, interactions) are MODE-invariant.
KEY_VARS = [
    resolve_col('wide_resource_share'),
    'log_HCI',
    'log_GFCF',
    'log_HCI_x_wide_share',
    'log_GFCF_x_wide_share',
    'post2019_x_coal_share',
    'post2019_x_crude_oil_share',
    'post2019_x_refined_oil_share',
    'post2019_x_gas_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
    'Rule of law index',
    'Political stability — estimate',
    'Trade (% of GDP)',
    'log_GDPpc',
]

# Take results from Models 1 and 3 (the FE specs of interest)
plot_data = all_results[
    (all_results['model'].isin(['M1_2wayFE_country_cluster', 'M3_parsim_interact']))
    & (all_results['variable'].isin(KEY_VARS))
].copy()

fig, ax = plt.subplots(figsize=(10, 7))
models = plot_data['model'].unique()
y_positions = np.arange(len(KEY_VARS))
offsets = np.linspace(-0.15, 0.15, len(models))

for i, model in enumerate(models):
    sub = plot_data[plot_data['model'] == model].set_index('variable')
    sub = sub.reindex(KEY_VARS)
    ax.errorbar(
        sub['beta'].values,
        y_positions + offsets[i],
        xerr=1.96 * sub['se'].values,
        fmt='o',
        capsize=3,
        label=model,
    )

ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(KEY_VARS)
ax.invert_yaxis()
ax.legend(loc='best', fontsize=9)
ax.set_title(f'Coefficient estimates ({MODE}): pooled across M imputations\\nHorizontal bars are 95% CI (Rubin\\'s rules)')
plt.tight_layout()

out_png = EXT / 'Graphics' / _suffix('e5_coefficients.png')
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')
plt.show()
'''
set_cell(22, CELL_22)
print('Cell 22 patched (KEY_VARS via resolve_col + PNG via _suffix)')


with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'\\nSaved: {NB}')
print()
print('========================================================')
print('To run gross: open, restart kernel, Run All (default MODE=gross).')
print('To run net:   set CLUSTER_MODE=net OR edit cell 2.')
print('Outputs:')
print('  intermediary/e5_results_long.csv      (gross)')
print('  intermediary/e5_results_wide.csv      (gross)')
print('  intermediary/e5_results_long_net.csv  (net)')
print('  intermediary/e5_results_wide_net.csv  (net)')
print('  Graphics/e5_coefficients.png          (gross)')
print('  Graphics/e5_coefficients_net.png      (net)')
print('========================================================')
