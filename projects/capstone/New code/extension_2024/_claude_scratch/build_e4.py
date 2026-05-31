"""Build e4_ml.ipynb.

Supervised learning to predict Economic Complexity Index from resource and
macro features. Pooled across M MICE imputations.

Models:
  - Lasso (feature selection)
  - Ridge (regularized linear)
  - Random Forest (non-linear interactions)
  - XGBoost (gradient boosting, optional if installed)

CV: PanelTemporalCV (expanding window, panel-aware)

Outputs:
  - intermediary/e4_results.csv     -- per-model CV R^2, MAE, RMSE
  - intermediary/e4_predictions.csv -- per-country-year predictions (pooled)
  - intermediary/e4_shap.csv        -- pooled SHAP importance
  - Graphics/NB4/feature_importance.png
  - Graphics/NB4/predictions_scatter.png
"""
import json
from pathlib import Path

NB_PATH = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e4_ml.ipynb')


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

cells.append(md('''# e4 — Supervised ML for Economic Complexity

Predicts Economic Complexity Index from resource exposure, macro controls,
and institutions. Pooled across M MICE imputations.

Models:
- **Lasso** (CV-tuned regularization for feature selection)
- **Ridge** (CV-tuned regularization, linear baseline)
- **Random Forest** (non-linear, captures interactions)
- **XGBoost** (gradient boosting, if available)

CV: `PanelTemporalCV` — expanding-window, panel-aware, gap=1 between train/val.

Targets:
- **Level**: Economic Complexity Index
- **Delta**: 1-year change (more demanding test of explanatory power)

Pooling across M imputations:
- Predictions: average across M (per row).
- Feature importance: average SHAP values across M.
- R²/RMSE: report mean and between-imputation SD.

Features:
- Trade-side resource shares (`wide_resource_share`, `hydrocarbon_share`, etc.)
- Per-HS-chapter shares
- COVID interactions
- Macro controls (HCI, GFCF, GDP per capita, governance, etc.)
- Rolling 5-year inflation and interest rates
'''))

cells.append(md('## 1. Setup'))

cells.append(code('''import sys, time, warnings
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

EXT  = Path('.').resolve()
INTER = EXT / 'intermediary'
GRAPHICS = EXT / 'Graphics' / 'NB4'
GRAPHICS.mkdir(parents=True, exist_ok=True)

print(f'Working dir: {EXT}')
print(f'M imputations available: {n_imputations()}')
'''))

cells.append(md('## 2. Feature engineering (applied to each imputation)'))

cells.append(code('''def engineer_features(panel):
    """Apply feature engineering identical to each of M panels."""
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

    # Resource concentration via HHI on trade chapter shares
    # (substitute for v2's WB-rents-based HHI)
    hs_share_cols = [c for c in df.columns if c.startswith('hs') and c.endswith('_share')]
    if hs_share_cols:
        # Normalise so shares sum to 1 (within the resource basket)
        share_sum = df[hs_share_cols].sum(axis=1).replace(0, np.nan)
        normalised = df[hs_share_cols].div(share_sum, axis=0)
        df['Resource_HHI_trade'] = (normalised ** 2).sum(axis=1)

    # Log transforms
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

    return df


FEATURE_COLS = [
    # Trade-side resource exposure (headline)
    'wide_resource_share',
    'hydrocarbon_share',
    'ores_share',
    'base_metals_share',
    'precious_share',
    'Resource_HHI_trade',
    # COVID interactions
    'post2019_x_hydrocarbon_share',
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
print(f'Features available: {len(present)} / {len(FEATURE_COLS)}')
if missing:
    print(f'Missing: {missing}')
FEATURES = present
'''))

cells.append(md('## 3. Panel-aware temporal cross-validation'))

cells.append(code('''class PanelTemporalCV:
    """Expanding-window CV for panels: train on years <= cutoff, validate on years > cutoff + gap."""
    def __init__(self, years, n_splits=5, gap=1, min_train_years=8):
        self.years = np.asarray(years)
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_years = min_train_years
        uy = np.sort(np.unique(self.years))
        ec = uy[0] + min_train_years - 1
        lc = uy[-1] - gap - 1
        if ec > lc:
            raise ValueError(f'Year range too narrow')
        self.cutoffs = np.linspace(ec, lc, n_splits).astype(int)

    def split(self, X=None, y=None, groups=None):
        for c in self.cutoffs:
            tr = np.where(self.years <= c)[0]
            va = np.where(self.years > c + self.gap)[0]
            if len(tr) > 0 and len(va) > 0:
                yield tr, va

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


print('CV class defined.')
'''))

cells.append(md('## 4. Train models on a single panel'))

cells.append(code('''def train_models(df_engineered, target='Economic Complexity Index'):
    """Train Lasso, Ridge, RF (+XGBoost if available) on a single engineered panel."""
    sub = df_engineered[FEATURES + [target, 'Country Code', 'Year']].dropna()
    if len(sub) == 0:
        return None
    X = sub[FEATURES].values
    y = sub[target].values
    years = sub['Year'].values

    # Standardise
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cv = PanelTemporalCV(years, n_splits=5, gap=1, min_train_years=8)

    models = {}

    # Lasso
    lasso = LassoCV(cv=cv, max_iter=10000, n_alphas=20)
    lasso.fit(Xs, y)
    models['Lasso'] = lasso

    # Ridge
    ridge = RidgeCV(cv=cv, alphas=np.logspace(-3, 3, 25))
    ridge.fit(Xs, y)
    models['Ridge'] = ridge

    # RF
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    rf.fit(Xs, y)
    models['RandomForest'] = rf

    # XGBoost (optional)
    if HAS_XGB:
        xgbm = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgbm.fit(Xs, y)
        models['XGBoost'] = xgbm

    # Out-of-fold predictions
    oof = {name: np.full(len(y), np.nan) for name in models}
    fold_scores = {name: [] for name in models}
    for tr, va in cv.split(Xs, y):
        for name, model in models.items():
            mod_clone = type(model)(**{k: v for k, v in model.get_params().items()})
            mod_clone.fit(Xs[tr], y[tr])
            pred = mod_clone.predict(Xs[va])
            oof[name][va] = pred
            fold_scores[name].append(r2_score(y[va], pred))

    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': FEATURES,
        'X': Xs, 'y': y,
        'sub_idx': sub.index,
        'sub': sub[['Country Code', 'Year']].reset_index(drop=True),
        'oof': oof,
        'fold_scores': fold_scores,
    }


# Sanity check on one imputation
imp1 = engineer_features(next(iter_imputations())[1])
print(f'Engineered panel: {len(imp1):,} rows, {len(FEATURES)} features')
'''))

cells.append(md('## 5. Train across M imputations'))

cells.append(code('''# Train all model on each of M imputations, collect OOF predictions and CV scores
all_results = []
all_oof = {}  # imp_id -> dict
print(f'Training {1 + (1 if HAS_XGB else 0) + 2} models across M imputations...')
t0 = time.time()

for imp_id, panel in iter_imputations():
    t_i = time.time()
    eng = engineer_features(panel)
    res = train_models(eng)
    if res is None:
        print(f'  imp {imp_id}: empty panel after dropna')
        continue

    for name in res['models']:
        scores = res['fold_scores'][name]
        oof = res['oof'][name]
        mask = ~np.isnan(oof)
        all_results.append({
            'imputation': imp_id,
            'model': name,
            'cv_r2_mean': float(np.mean(scores)),
            'cv_r2_std':  float(np.std(scores)),
            'oof_r2':     r2_score(res['y'][mask], oof[mask]) if mask.any() else np.nan,
            'oof_mae':    mean_absolute_error(res['y'][mask], oof[mask]) if mask.any() else np.nan,
            'oof_rmse':   np.sqrt(mean_squared_error(res['y'][mask], oof[mask])) if mask.any() else np.nan,
            'n_train':    int(mask.sum()),
        })
    all_oof[imp_id] = res
    print(f'  imp {imp_id}: trained in {time.time() - t_i:.1f}s')

print(f'\\nTotal: {time.time() - t0:.1f}s')
results_df = pd.DataFrame(all_results)
print()
print(results_df.to_string(index=False))
'''))

cells.append(md('## 6. Pool predictions and scores'))

cells.append(code('''# Per-model: average OOF predictions across imputations, compute pooled R^2
pooled_results = []
for name in results_df['model'].unique():
    # Stack OOF predictions: (M, N) — but each imp has different sub_idx
    # We need to align on (Country Code, Year)
    preds_list = []
    for imp_id, res in all_oof.items():
        pred_df = res['sub'].copy()
        pred_df['pred'] = res['oof'][name]
        preds_list.append(pred_df)

    # Concatenate and average by (Country Code, Year)
    stacked = pd.concat(preds_list, ignore_index=True)
    pooled_pred = stacked.groupby(['Country Code', 'Year'])['pred'].mean().reset_index()

    # Merge with ground truth from imp0 (any imp would do; ECI same across imps for observed)
    truth = next(iter_imputations())[1][['Country Code', 'Year', 'Economic Complexity Index']].dropna()
    merged = truth.merge(pooled_pred, on=['Country Code', 'Year'], how='inner')
    if len(merged) == 0:
        continue

    pooled_results.append({
        'model':      name,
        'pooled_r2':  r2_score(merged['Economic Complexity Index'], merged['pred']),
        'pooled_mae': mean_absolute_error(merged['Economic Complexity Index'], merged['pred']),
        'pooled_rmse': np.sqrt(mean_squared_error(merged['Economic Complexity Index'], merged['pred'])),
        'n_pooled':   len(merged),
    })

pooled_df = pd.DataFrame(pooled_results)
print('Pooled across M imputations:')
print(pooled_df.to_string(index=False))

# Save full results
out = INTER / 'e4_results.csv'
results_df.to_csv(out, index=False)
print(f'\\nSaved per-imputation: {out}')
'''))

cells.append(md('## 7. Feature importance (SHAP, pooled)'))

cells.append(code('''if not HAS_SHAP:
    print('SHAP not available, skipping importance computation.')
else:
    # Compute SHAP for RF on each imputation, average
    print('Computing SHAP values across M imputations...')
    t0 = time.time()
    shap_per_imp = []
    for imp_id, res in all_oof.items():
        rf = res['models']['RandomForest']
        Xs = res['X']
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(Xs)
        # Mean absolute SHAP per feature
        importances = np.abs(sv).mean(axis=0)
        shap_per_imp.append(importances)
        print(f'  imp {imp_id} done in {time.time() - t0:.1f}s')

    shap_arr = np.array(shap_per_imp)
    shap_mean = shap_arr.mean(axis=0)
    shap_std = shap_arr.std(axis=0)

    shap_df = pd.DataFrame({
        'feature': FEATURES,
        'shap_mean': shap_mean,
        'shap_std':  shap_std,
        'shap_sd_rel': shap_std / (shap_mean + 1e-8),
    }).sort_values('shap_mean', ascending=False)

    print('\\nTop-15 features by pooled SHAP importance:')
    print(shap_df.head(15).to_string(index=False))

    out_shap = INTER / 'e4_shap.csv'
    shap_df.to_csv(out_shap, index=False)
    print(f'\\nSaved: {out_shap}')
'''))

cells.append(md('## 8. Coefficient table (Lasso, Ridge averaged across M)'))

cells.append(code('''# For linear models, average coefficients across M imputations
def collect_linear_coefs(model_name):
    coef_rows = []
    for imp_id, res in all_oof.items():
        model = res['models'][model_name]
        if hasattr(model, 'coef_'):
            for feat, coef in zip(res['feature_cols'], model.coef_):
                coef_rows.append({'imp': imp_id, 'feature': feat, 'coef': coef})
    return pd.DataFrame(coef_rows)

for name in ['Lasso', 'Ridge']:
    df_coef = collect_linear_coefs(name)
    if df_coef.empty:
        continue
    pivot = (df_coef.groupby('feature')['coef']
                    .agg(['mean', 'std', 'count'])
                    .sort_values('mean', key=abs, ascending=False))
    pivot['nonzero_rate'] = df_coef.groupby('feature')['coef'].apply(lambda x: (x.abs() > 1e-8).mean())
    print(f'\\n{name} (pooled across M):')
    print(pivot.head(15).to_string())
'''))

cells.append(md('## 9. Visualisations'))

cells.append(code('''# Bar chart of feature importance
if HAS_SHAP:
    fig, ax = plt.subplots(figsize=(10, 8))
    top = shap_df.head(20)
    ax.barh(top['feature'], top['shap_mean'], xerr=top['shap_std'],
            color='#4a6fa5', alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean absolute SHAP value (pooled across M imputations)')
    ax.set_title('Feature importance — Random Forest predicting ECI')
    plt.tight_layout()
    plt.savefig(GRAPHICS / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

# Predicted vs actual ECI (RF, pooled)
truth = next(iter_imputations())[1][['Country Code', 'Year', 'Economic Complexity Index']].dropna()
rf_preds = []
for imp_id, res in all_oof.items():
    sub = res['sub'].copy()
    sub['pred'] = res['oof']['RandomForest']
    rf_preds.append(sub)
all_rf_pred = pd.concat(rf_preds, ignore_index=True)
pooled_rf = all_rf_pred.groupby(['Country Code', 'Year'])['pred'].mean().reset_index()
merged = truth.merge(pooled_rf, on=['Country Code', 'Year'], how='inner')

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(merged['Economic Complexity Index'], merged['pred'], alpha=0.4, s=15)
lims = [min(merged['Economic Complexity Index'].min(), merged['pred'].min()),
        max(merged['Economic Complexity Index'].max(), merged['pred'].max())]
ax.plot(lims, lims, 'r--', alpha=0.5, label='45° line')
ax.set_xlabel('Actual ECI')
ax.set_ylabel('Predicted ECI (Random Forest, pooled M=5)')
r2 = r2_score(merged['Economic Complexity Index'], merged['pred'])
ax.set_title(f'OOF predictions (R² = {r2:.3f}, N = {len(merged):,})')
ax.legend()
plt.tight_layout()
plt.savefig(GRAPHICS / 'predictions_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# Save predictions
out_preds = INTER / 'e4_predictions.csv'
pooled_rf.to_csv(out_preds, index=False)
print(f'Saved: {out_preds}')
'''))

cells.append(md('## 10. Summary'))

cells.append(code('''print('=' * 70)
print('e4 — ML summary')
print('=' * 70)
print(f'Imputations:  {results_df["imputation"].nunique()}')
print(f'Models:       {", ".join(results_df["model"].unique())}')
print()
print('Pooled performance (predictions averaged across M):')
print(pooled_df.to_string(index=False))
print()
print('Files:')
print(f'  intermediary/e4_results.csv     (per-imputation CV scores)')
print(f'  intermediary/e4_predictions.csv (pooled per-country-year)')
print(f'  intermediary/e4_shap.csv        (pooled feature importance)')
print(f'  Graphics/NB4/feature_importance.png')
print(f'  Graphics/NB4/predictions_scatter.png')
print()
print('Next: e6 (forecasting), e7+ (robustness/visualization)')
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
