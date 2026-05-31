"""Build e5_regressions.ipynb programmatically.

Estimation strategy:
  Headline      : Two-way FE (country + year), country-clustered SE
  Robust SE     : Same FE, two-way clustered SE (Cameron-Gelbach-Miller)
  v2 comparison : Pooled OLS, country-clustered SE
  Observed-only : Two-way FE on observed-only sample (no imputation)

Pooled across M=10 imputations via Rubin's rules from _mice_pool.

Dependent variable: Economic Complexity Index
Resource-dependence proxy: wide_resource_share (trade-derived)
v2-style interactions: log_HCI x wide_resource_share, log_GFCF x wide_resource_share
COVID interactions: post_2019 x {hydrocarbon, ores, base_metals}_share
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e5_regressions.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text.splitlines(keepends=True),
    }


cells = []

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('''# e5 — Headline Regressions

Dependent variable: **Economic Complexity Index** (ECI), level.

Estimation strategy:

| # | Spec | FE | Standard Errors | Sample |
|---|---|---|---|---|
| 1 | Headline | Country + Year | Country-clustered | Imputed (pooled across M=10) |
| 2 | Robust SE | Country + Year | Two-way clustered (CGM) | Imputed (pooled across M=10) |
| 3 | Parsimonious + interactions | Country + Year | Country-clustered | Imputed (pooled across M=10) |
| 4 | v2 comparison | None | Country-clustered | Imputed (pooled across M=10) |
| 5 | Robustness | Country + Year | Country-clustered | Observed-only |

Within-country, within-year identification (Models 1-3, 5) addresses the
canonical resource-curse critique that cross-sectional correlations conflate
geography, history, and institutions with resource exposure. Year FE absorbs
the COVID shock as a common time effect; the COVID interactions
(`post_2019 × *_share`) then identify the differential resource impact.

Standard errors pooled across M=10 imputations via Rubin's rules (within-variance
+ between-variance correction). Fraction of Missing Information (FMI) reported
for each coefficient.

Notes:
- Forestry-rents interactions from v2 are dropped (no Master_v2 equivalent).
- The "Total_Production_Value" variable is queued for re-introduction once
  the e0 PRODVAL_PATH collision is fixed (see todo).
'''))

cells.append(md('## 1. Setup'))

cells.append(code('''import os, sys, time, warnings, itertools
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

EXT  = Path('.').resolve()
INTER = EXT / 'intermediary'

print(f'Working dir: {EXT}')
print(f'Year range:  {cfg.YEAR_MIN}-{cfg.YEAR_MAX}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 2. Variable definitions and transforms'))

cells.append(code('''# Common transforms applied identically to each of the M panels and to the
# observed-only panel. Mean-centering uses the grand sample mean within each
# panel (so the interaction coefficients are interpretable at the mean of the
# interacted variable).

# Variables we expect to find in Master_v2 (verified at load time).
# If something is missing, the cell prints a warning and drops it from the spec.
LOG_VARS = [
    ('Human capital index', 'log_HCI'),
    ('Gross fixed capital formation, all, Constant prices, Percent of GDP', 'log_GFCF'),
    ('GDP per capita (constant prices, PPP)', 'log_GDPpc'),
    ('Population', 'log_Pop'),
]

# Variables to mean-center for interaction terms
CENTER_VARS = ['log_HCI', 'log_GFCF', 'wide_resource_share']


def transform_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply log and centering transforms. Adds new columns, does not modify inputs."""
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

    # v2-style interactions (HCI × resource intensity, GFCF × resource intensity)
    if 'log_HCI_c' in df.columns and 'wide_resource_share_c' in df.columns:
        df['log_HCI_x_wide_share']  = df['log_HCI_c']  * df['wide_resource_share_c']
    if 'log_GFCF_c' in df.columns and 'wide_resource_share_c' in df.columns:
        df['log_GFCF_x_wide_share'] = df['log_GFCF_c'] * df['wide_resource_share_c']

    # COVID interactions are already pre-computed in e2; passing through
    return df


# Sanity check on the first imputation
imp1 = next(iter_imputations())[1]
imp1 = transform_panel(imp1)
needed = ['log_HCI', 'log_GFCF', 'log_GDPpc', 'log_Pop',
          'log_HCI_x_wide_share', 'log_GFCF_x_wide_share',
          'wide_resource_share', 'hydrocarbon_share', 'ores_share', 'base_metals_share',
          'post2019_x_hydrocarbon_share', 'post2019_x_ores_share', 'post2019_x_base_metals_share',
          'Economic Complexity Index']
missing = [c for c in needed if c not in imp1.columns]
print(f'Sanity check: {len(needed)} expected columns, {len(missing)} missing')
if missing:
    print(f'  Missing: {missing}')
else:
    print('  All present.')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 3. Specification registry'))

cells.append(code('''# Full set of controls used in Model 1 (the headline) and Model 2 (robust SE)
# Note: Landlocked, dummies, post_2019 are entity- or year-invariant and absorbed
# by FE, so they're excluded from these specs.

CONTROLS_FULL = [
    # Trade and macro (log_GDPpc dropped: avoid ECI/GDPpc mechanical entanglement;
    # parallel with structural-v2 ML spec which also drops it)
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
    # Resource exposure (trade-side)
    'wide_resource_share',
    # COVID interactions
    'post2019_x_hydrocarbon_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
]

# Parsimonious set + interactions (Model 3)
CONTROLS_PARSIMONIOUS = [
    'log_HCI',
    'log_GFCF',
    'Political stability — estimate',
    'Rule of law index',
    # log_GDPpc dropped (parallel with CONTROLS_FULL change)
    'Trade (% of GDP)',
    'wide_resource_share',
]

INTERACTIONS_PARSIMONIOUS = [
    'log_HCI_x_wide_share',
    'log_GFCF_x_wide_share',
    'post2019_x_hydrocarbon_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
]

print(f'Full controls:        {len(CONTROLS_FULL)}')
print(f'Parsimonious + ints:  {len(CONTROLS_PARSIMONIOUS)} + {len(INTERACTIONS_PARSIMONIOUS)}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 4. Regression helpers'))

cells.append(code('''def fit_panel_fe(panel, depvar, indep_vars,
                 entity_effects=True, time_effects=True,
                 cluster_entity=True, cluster_time=False):
    """Fit two-way FE panel regression with clustered SEs via linearmodels.

    Drops rows with any NaN in depvar or indep_vars. Returns a SimpleNamespace
    with params, bse, tvalues, pvalues, nobs, rsquared, ngroups.
    """
    cols = ['Country Code', 'Year', depvar] + indep_vars
    sub = panel[cols].dropna().copy()
    sub = sub.set_index(['Country Code', 'Year'])

    y = sub[depvar]
    X = sub[indep_vars]

    cov_kwds = {}
    cov_type = 'clustered'
    if cluster_entity:
        cov_kwds['cluster_entity'] = True
    if cluster_time:
        cov_kwds['cluster_time'] = True

    model = PanelOLS(y, X,
                     entity_effects=entity_effects,
                     time_effects=time_effects,
                     drop_absorbed=True,
                     check_rank=False)
    res = model.fit(cov_type=cov_type, **cov_kwds)

    return SimpleNamespace(
        params=res.params,
        bse=res.std_errors,
        tvalues=res.tstats,
        pvalues=res.pvalues,
        nobs=int(res.nobs),
        rsquared=float(res.rsquared),
        ngroups=sub.index.get_level_values('Country Code').nunique(),
    )


def fit_pooled_ols(panel, depvar, indep_vars, cluster='Country Code'):
    """Pooled OLS with country-clustered SEs (v2-style for comparison)."""
    cols = [cluster, depvar] + indep_vars
    sub = panel[cols].dropna().copy()

    y = sub[depvar]
    X = sm.add_constant(sub[indep_vars])
    res = sm.OLS(y, X).fit(cov_type='cluster',
                           cov_kwds={'groups': sub[cluster]})
    return SimpleNamespace(
        params=res.params,
        bse=res.bse,
        tvalues=res.tvalues,
        pvalues=res.pvalues,
        nobs=int(res.nobs),
        rsquared=float(res.rsquared),
        ngroups=sub[cluster].nunique(),
    )


def run_pooled_mice(model_fn, model_label, depvar, indep_vars,
                     panels=None, **kwargs):
    """Run a regression on each of M panels, pool via Rubin's rules.

    Returns DataFrame with columns: variable, beta, se, t, p, df, fmi, m_used.
    """
    results = []
    for imp_id, panel in iter_imputations() if panels is None else enumerate(panels):
        panel = transform_panel(panel)
        try:
            res = model_fn(panel, depvar, indep_vars, **kwargs)
            results.append(res)
        except Exception as e:
            print(f'  WARNING: imputation {imp_id} failed: {e}')
            continue

    if not results:
        return pd.DataFrame()

    # Collect betas/ses per variable
    all_vars = sorted(set().union(*[set(r.params.index) for r in results]))
    rows = []
    for v in all_vars:
        betas = [float(r.params[v]) for r in results if v in r.params.index]
        ses   = [float(r.bse[v])**2 for r in results if v in r.bse.index]
        if len(betas) < 2:
            continue
        pooled = pool_scalars(betas, ses)
        rows.append({
            'model': model_label,
            'variable': v,
            'beta':  pooled.point,
            'se':    pooled.se,
            't':     pooled.t_stat(),
            'p':     pooled.p_value(),
            'df':    pooled.df,
            'fmi':   pooled.fmi,
            'm_used': len(betas),
            'n_obs': int(np.mean([r.nobs for r in results])),
            'n_ctry': int(np.mean([r.ngroups for r in results])),
            'r2_avg': float(np.mean([r.rsquared for r in results])),
        })
    return pd.DataFrame(rows)


def format_results_table(df_results, var_order=None):
    """Format a regression result table for display."""
    if var_order:
        df_results = df_results.set_index('variable').reindex(var_order).reset_index()

    def stars(p):
        if pd.isna(p): return ''
        if p < 0.01: return '***'
        if p < 0.05: return '**'
        if p < 0.10: return '*'
        return ''

    df_results = df_results.copy()
    df_results['sig'] = df_results['p'].apply(stars)
    return df_results


print('Helpers defined.')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 5. Model 1 — Headline (Two-way FE, Country-clustered SE)'))

cells.append(code('''DEPVAR = 'Economic Complexity Index'

print('=' * 70)
print('MODEL 1 — Two-way FE, Country-clustered SE')
print('=' * 70)
t0 = time.time()

m1 = run_pooled_mice(
    fit_panel_fe,
    model_label='M1_2wayFE_country_cluster',
    depvar=DEPVAR,
    indep_vars=CONTROLS_FULL,
    entity_effects=True, time_effects=True,
    cluster_entity=True, cluster_time=False,
)

print(f'Done in {time.time() - t0:.1f}s')
print()
print(format_results_table(m1).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 6. Model 2 — Robust SE (Two-way clustered, CGM)'))

cells.append(code('''print('=' * 70)
print('MODEL 2 — Two-way FE, Two-way clustered SE (Cameron-Gelbach-Miller)')
print('=' * 70)
t0 = time.time()

m2 = run_pooled_mice(
    fit_panel_fe,
    model_label='M2_2wayFE_twoway_cluster',
    depvar=DEPVAR,
    indep_vars=CONTROLS_FULL,
    entity_effects=True, time_effects=True,
    cluster_entity=True, cluster_time=True,
)

print(f'Done in {time.time() - t0:.1f}s')
print()
print(format_results_table(m2).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 7. Model 3 — Parsimonious + Interactions'))

cells.append(code('''print('=' * 70)
print('MODEL 3 — Parsimonious + Interactions (FE, country-clustered)')
print('=' * 70)
t0 = time.time()

m3_vars = CONTROLS_PARSIMONIOUS + INTERACTIONS_PARSIMONIOUS

m3 = run_pooled_mice(
    fit_panel_fe,
    model_label='M3_parsim_interact',
    depvar=DEPVAR,
    indep_vars=m3_vars,
    entity_effects=True, time_effects=True,
    cluster_entity=True, cluster_time=False,
)

print(f'Done in {time.time() - t0:.1f}s')
print()
print(format_results_table(m3).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 8. Model 4 — Pooled OLS (v2-style comparison)'))

cells.append(code('''print('=' * 70)
print('MODEL 4 — Pooled OLS (no FE), Country-clustered SE [v2 baseline]')
print('=' * 70)
t0 = time.time()

m4 = run_pooled_mice(
    fit_pooled_ols,
    model_label='M4_pooled_OLS',
    depvar=DEPVAR,
    indep_vars=CONTROLS_FULL,
)

print(f'Done in {time.time() - t0:.1f}s')
print()
print(format_results_table(m4).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 9. Model 5 — Observed-only (Robustness, single regression)'))

cells.append(code('''print('=' * 70)
print('MODEL 5 — Observed-only sample (no imputation), Two-way FE')
print('=' * 70)

observed = pd.read_csv(INTER / 'Master_v2_observed.csv')
observed = transform_panel(observed)
print(f'Observed-only panel: {len(observed):,} rows before NaN drop')

t0 = time.time()
m5_raw = fit_panel_fe(
    observed, DEPVAR, CONTROLS_FULL,
    entity_effects=True, time_effects=True,
    cluster_entity=True, cluster_time=False,
)

# Format as a single-imputation results table (no Rubin pooling needed)
m5 = pd.DataFrame({
    'model':   'M5_observed_only',
    'variable': m5_raw.params.index,
    'beta':    m5_raw.params.values,
    'se':      m5_raw.bse.values,
    't':       m5_raw.tvalues.values,
    'p':       m5_raw.pvalues.values,
    'df':      np.nan,
    'fmi':     np.nan,   # No imputation -> no FMI
    'm_used':  1,
    'n_obs':   int(m5_raw.nobs),
    'n_ctry':  int(m5_raw.ngroups),
    'r2_avg':  float(m5_raw.rsquared),
})

print(f'Done in {time.time() - t0:.1f}s')
print()
print(format_results_table(m5).to_string(index=False))
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 10. Combined results table'))

cells.append(code('''# Combine all 5 models side by side
all_results = pd.concat([m1, m2, m3, m4, m5], ignore_index=True)

# Wide pivot: variable in rows, (model × measure) in columns
def make_wide(df):
    pivot = df.pivot_table(index='variable', columns='model',
                            values=['beta', 'se', 'p'], aggfunc='first')
    return pivot

wide = make_wide(all_results)
print('Combined results (all 5 models):')
print(wide.to_string())

# Save
out_csv = INTER / 'e5_results_long.csv'
all_results.to_csv(out_csv, index=False)
print(f'\\nSaved long results: {out_csv}')

out_wide = INTER / 'e5_results_wide.csv'
wide.to_csv(out_wide)
print(f'Saved wide results:  {out_wide}')
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 11. Coefficient plot (key variables)'))

cells.append(code('''import matplotlib.pyplot as plt

KEY_VARS = [
    'wide_resource_share',
    'log_HCI',
    'log_GFCF',
    'log_HCI_x_wide_share',
    'log_GFCF_x_wide_share',
    'post2019_x_hydrocarbon_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
    'Rule of law index',
    'Political stability — estimate',
    'Trade (% of GDP)',
    # log_GDPpc removed from KEY_VARS now that it is no longer in CONTROLS_FULL
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
ax.set_title('Coefficient estimates: pooled across M=10 imputations\\nHorizontal bars are 95% CI (Rubin\\'s rules)')
plt.tight_layout()

out_png = EXT / 'Graphics' / 'e5_coefficients.png'
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')
plt.show()
'''))

# ─────────────────────────────────────────────────────────────────────────────
cells.append(md('## 12. Summary'))

cells.append(code('''print('=' * 70)
print('e5 — Regression results summary')
print('=' * 70)
for model_label, df in [
    ('Model 1 (Headline, 2wayFE, country-cluster)', m1),
    ('Model 2 (Robust SE, 2wayFE, two-way cluster)', m2),
    ('Model 3 (Parsimonious + interactions)', m3),
    ('Model 4 (Pooled OLS, v2 baseline)', m4),
    ('Model 5 (Observed-only)', m5),
]:
    if len(df) == 0:
        print(f'{model_label}: empty')
        continue
    n_obs = int(df['n_obs'].iloc[0])
    n_ctry = int(df['n_ctry'].iloc[0])
    r2 = df['r2_avg'].iloc[0]
    sig = (df['p'] < 0.05).sum()
    print(f'{model_label}')
    print(f'  N={n_obs:,}, countries={n_ctry}, R^2={r2:.3f}, sig vars (p<0.05): {sig}')
    print()

print('Files written:')
print(f'  intermediary/e5_results_long.csv   (long format)')
print(f'  intermediary/e5_results_wide.csv   (wide format for paper tables)')
print(f'  Graphics/e5_coefficients.png       (coefficient plot)')
print()
print('Next steps:')
print('  - Re-run e0 with PRODVAL_PATH fix to produce ProductionValue columns')
print('  - Re-run e1, e2 to propagate to Master_v2')
print('  - Add Model 6 (Production-Value spec) to e5')
print('  - e6: forecasting')
print('  - e3/e4: clustering and ML')
'''))

# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.4'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Wrote {NB_PATH}')
print(f'Cells: {len(cells)} ({sum(1 for c in cells if c["cell_type"] == "code")} code, '
      f'{sum(1 for c in cells if c["cell_type"] == "markdown")} markdown)')
