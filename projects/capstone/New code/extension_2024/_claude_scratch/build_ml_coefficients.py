"""
Fit Lasso, Ridge, ElasticNet, RandomForest on the structural-v2 feature
set (drops L1_ECI and log_GDPpc) across MICE imputations. Save pooled
coefficients and RF importances to intermediary/e4_ml_coefficients.csv.

Matches build_e4.py's engineer_features pipeline and uses the 3 combined
post-2019 interactions (hydrocarbon, ores, base metals) consistent with
the rest of the project.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model  import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
sys.path.insert(0, str(EXT))
from _mice_pool import iter_imputations

# --- feature engineering, mirrored from build_e4.py ------------------------
def engineer(df):
    df = df.sort_values(['Country Code', 'Year']).copy()

    if 'Inflation, consumer prices (annual %)' in df.columns:
        df['Inflation_roll5'] = (df.groupby('Country Code')
            ['Inflation, consumer prices (annual %)']
            .transform(lambda x: x.rolling(5, min_periods=3).mean()))
    if 'Real interest rate (%)' in df.columns:
        df['RealRate_roll5'] = (df.groupby('Country Code')
            ['Real interest rate (%)']
            .transform(lambda x: x.rolling(5, min_periods=3).mean()))

    hs_share_cols = [c for c in df.columns if c.startswith('hs') and c.endswith('_share')]
    if hs_share_cols:
        share_sum = df[hs_share_cols].sum(axis=1).replace(0, np.nan)
        normalised = df[hs_share_cols].div(share_sum, axis=0)
        df['Resource_HHI_trade'] = (normalised ** 2).sum(axis=1)

    for raw, lognm in [
        ('Human capital index', 'log_HCI'),
        ('Gross fixed capital formation, all, Constant prices, Percent of GDP', 'log_GFCF'),
        ('GDP per capita (constant prices, PPP)', 'log_GDPpc'),
        ('Population', 'log_Pop'),
    ]:
        if raw in df.columns:
            df[lognm] = np.log(df[raw].clip(lower=0) + 1)

    return df

# Structural v2: drops L1_ECI AND log_GDPpc
STRUCT_FEATURES = [
    'wide_resource_share',
    'hydrocarbon_share', 'ores_share', 'base_metals_share', 'precious_share',
    'Resource_HHI_trade',
    'post2019_x_hydrocarbon_share',
    'post2019_x_ores_share',
    'post2019_x_base_metals_share',
    'log_Pop', 'log_HCI', 'log_GFCF',
    'Trade (% of GDP)', 'Domestic credit to private sector (% of GDP)',
    'Agriculture', 'Industry', 'Manufacturing', 'Services',
    'Urban population (% of total population)',
    'Political stability — estimate', 'Rule of law index',
    'Political corruption index',
    'Inflation_roll5', 'RealRate_roll5',
]
TARGET = 'Economic Complexity Index'

# --- fit per imputation ----------------------------------------------------
coefs_l, coefs_r, coefs_e, imps_rf = [], [], [], []

for imp_id, panel in iter_imputations():
    eng = engineer(panel)
    missing = [c for c in STRUCT_FEATURES if c not in eng.columns]
    if missing:
        raise SystemExit(f'missing columns: {missing}')

    sub = eng[STRUCT_FEATURES + [TARGET, 'Country Code', 'Year']].dropna()
    print(f'imp {imp_id}: N={len(sub):,}', flush=True)
    X  = sub[STRUCT_FEATURES].values
    y  = sub[TARGET].values
    Xs = StandardScaler().fit_transform(X)

    lasso = LassoCV(cv=5, max_iter=20000, n_alphas=20).fit(Xs, y)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xs, y)
    en    = ElasticNetCV(cv=5, max_iter=20000,
                         l1_ratio=[0.1, 0.5, 0.9], n_alphas=20).fit(Xs, y)
    rf    = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                                  random_state=42, n_jobs=-1).fit(Xs, y)

    coefs_l.append(lasso.coef_)
    coefs_r.append(ridge.coef_)
    coefs_e.append(en.coef_)
    imps_rf.append(rf.feature_importances_)
    print(f'  lasso a={lasso.alpha_:.4f}  ridge a={ridge.alpha_:.4f}  '
          f'EN a={en.alpha_:.4f} l1={en.l1_ratio_}', flush=True)

# --- pool & save -----------------------------------------------------------
out = pd.DataFrame({
    'feature': STRUCT_FEATURES,
    'lasso':   np.mean(coefs_l, axis=0),
    'ridge':   np.mean(coefs_r, axis=0),
    'en':      np.mean(coefs_e, axis=0),
    'rf':      np.mean(imps_rf, axis=0),
})
out['abs_avg'] = out[['lasso', 'ridge', 'en']].abs().mean(axis=1)
mx = out['rf'].max()
out['rf_norm'] = out['rf'] / mx if mx > 0 else out['rf']

out_path = EXT / 'intermediary' / 'e4_ml_coefficients.csv'
out.to_csv(out_path, index=False)
print(f'\nWrote: {out_path}')
print('\nTop 10 by absolute average linear coefficient:')
print(out.sort_values('abs_avg', ascending=False).head(10)
        [['feature','lasso','ridge','en','rf_norm']].to_string(index=False))
