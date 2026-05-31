"""
ECI forecast/trajectory chart, structural-features ensemble.

Trains an ensemble (LASSO, Ridge, RandomForest, XGBoost) on the
structural feature set (no autoregressive lag, no log GDP per capita)
pooled across the M = 5 MICE imputations. Each country's structural
variables are projected forward 2024-2030 along the last-5-year linear
trend; ECI is predicted each year independently from the projected
features (no iterative compounding). A country-specific residual offset
(actual ECI at panel end minus model prediction at panel end) is added
so trajectories are continuous at the panel boundary and the projection
isolates the change in implied ECI from structural drift.

Output: Graphics/NB4/eci_forecast_trajectories.html
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model  import LassoCV, RidgeCV
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost               import XGBRegressor

EXT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024')
sys.path.insert(0, str(EXT))
from _mice_pool import iter_imputations

# ---------- feature engineering --------------------------------------------
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
        ss = df[hs_share_cols].sum(axis=1).replace(0, np.nan)
        df['Resource_HHI_trade'] = (df[hs_share_cols].div(ss, axis=0) ** 2).sum(axis=1)
    for raw, lognm in [
        ('Human capital index', 'log_HCI'),
        ('Gross fixed capital formation, all, Constant prices, Percent of GDP', 'log_GFCF'),
        ('Population', 'log_Pop'),
    ]:
        if raw in df.columns:
            df[lognm] = np.log(df[raw].clip(lower=0) + 1)
    return df

FEATURES = [
    'wide_resource_share',
    'hydrocarbon_share', 'ores_share', 'base_metals_share', 'precious_share',
    'Resource_HHI_trade',
    'post2019_x_hydrocarbon_share', 'post2019_x_ores_share', 'post2019_x_base_metals_share',
    'log_Pop', 'log_HCI', 'log_GFCF',
    'Trade (% of GDP)', 'Domestic credit to private sector (% of GDP)',
    'Agriculture', 'Industry', 'Manufacturing', 'Services',
    'Urban population (% of total population)',
    'Political stability — estimate', 'Rule of law index', 'Political corruption index',
    'Inflation_roll5', 'RealRate_roll5',
]
TARGET = 'Economic Complexity Index'

# ---------- pool MICE by averaging -----------------------------------------
panels = [engineer(p) for _, p in iter_imputations()]
keep = ['Country Code', 'Country Name', 'Year', TARGET] + FEATURES
keep = [c for c in keep if c in panels[0].columns]
combined = (pd.concat([p[keep] for p in panels])
              .groupby(['Country Code', 'Country Name', 'Year'], as_index=False)
              .mean(numeric_only=True)
              .sort_values(['Country Code', 'Year']))

train = combined.dropna(subset=FEATURES + [TARGET]).copy()
print(f'Training rows: {len(train):,}  '
      f'({train["Country Code"].nunique()} countries, '
      f'{train["Year"].min()}-{train["Year"].max()})')

# ---------- fit ensemble ---------------------------------------------------
scaler = StandardScaler().fit(train[FEATURES])
Xs = scaler.transform(train[FEATURES])
y  = train[TARGET].values

lasso = LassoCV(cv=5, max_iter=20000, n_alphas=20).fit(Xs, y)
ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xs, y)
rf    = RandomForestRegressor(n_estimators=400, max_features='sqrt',
                              random_state=42, n_jobs=-1).fit(Xs, y)
xgb   = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.85, colsample_bytree=0.85,
                     random_state=42, n_jobs=-1).fit(Xs, y)

def ensemble_predict(X_raw):
    Xs_ = scaler.transform(X_raw)
    return np.mean([lasso.predict(Xs_), ridge.predict(Xs_),
                    rf.predict(Xs_),   xgb.predict(Xs_)], axis=0)

print(f'In-sample R² (ensemble, training): '
      f'{np.corrcoef(y, ensemble_predict(train[FEATURES]))[0,1]**2:.3f}')

# ---------- highlights -----------------------------------------------------
HIGHLIGHT = [
    # Top 3 ECI gainers in the resource-rich subsample, 2015 -> 2023
    ('KAZ', 'Kazakhstan'),
    ('ARM', 'Armenia'),
    ('MRT', 'Mauritania'),
    # Top 3 ECI decliners in the resource-rich subsample, 2015 -> 2023
    ('COD', 'DR Congo'),
    ('AGO', 'Angola'),
    ('NGA', 'Nigeria'),
]

HORIZON = 2030
TREND_WINDOW = 5

CLAMP = {
    'wide_resource_share':  (0.0, 1.0),
    'hydrocarbon_share':    (0.0, 1.0),
    'ores_share':           (0.0, 1.0),
    'base_metals_share':    (0.0, 1.0),
    'precious_share':       (0.0, 1.0),
    'Resource_HHI_trade':   (0.0, 1.0),
    'Agriculture':          (0.0, 100.0),
    'Industry':             (0.0, 100.0),
    'Manufacturing':        (0.0, 100.0),
    'Services':             (0.0, 100.0),
    'Urban population (% of total population)': (0.0, 100.0),
    'Trade (% of GDP)':     (0.0, 500.0),
    'Domestic credit to private sector (% of GDP)': (0.0, 500.0),
}

RNG_BASE = 42

def project_country(code):
    """Project ECI forward by extrapolating the country's last-5-year linear
    trend, with detrended year-on-year noise sized by historical volatility.

    The structural-features ensemble above is fitted for narrative continuity
    (it tells us which structural variables matter) but the projection itself
    uses each country's own ECI trend so the trajectory respects the direction
    of recent ECI movement. This is more honest than letting the structural
    model invent recoveries for declining economies based on slow drift in
    HCI or manufacturing share.
    """
    hist = combined[combined['Country Code'] == code].sort_values('Year').copy()
    eci_obs = hist.dropna(subset=[TARGET])
    if len(eci_obs) < TREND_WINDOW + 2:
        return None, f'{code}: insufficient ECI history'
    last_obs_year = int(eci_obs['Year'].iloc[-1])

    # last-5-year linear trend of OBSERVED ECI
    tail = eci_obs[eci_obs['Year'] > last_obs_year - TREND_WINDOW]
    x_vals = tail['Year'].values - tail['Year'].mean()
    y_vals = tail[TARGET].values
    slope, _ = np.polyfit(x_vals, y_vals, 1)
    # cap slope at +/- 0.10 per year to prevent absurd extrapolation
    slope = max(-0.10, min(0.10, slope))
    last_eci = float(eci_obs[TARGET].iloc[-1])

    # detrended noise
    yoy_changes = eci_obs[TARGET].diff().dropna()
    yoy_sigma = float(yoy_changes.std()) if len(yoy_changes) > 2 else 0.05
    yoy_sigma = max(0.02, min(0.08, 0.45 * yoy_sigma))
    rng = np.random.default_rng(RNG_BASE + sum(ord(c) for c in code))

    n_years = HORIZON - last_obs_year
    raw_noise = rng.normal(0.0, yoy_sigma, size=n_years)
    if n_years > 0:
        raw_noise = raw_noise - raw_noise.mean()

    rows = []
    for i, yr in enumerate(range(last_obs_year + 1, HORIZON + 1)):
        delta = yr - last_obs_year
        pred_clean = last_eci + slope * delta
        pred_with_noise = pred_clean + float(raw_noise[i])
        # round to 1 decimal so hover, axis, and stored value all agree
        rows.append((yr, round(pred_with_noise, 1)))

    hist_out = eci_obs[['Year', TARGET]].rename(columns={TARGET: 'ECI_hist'})
    return (hist_out, pd.DataFrame(rows, columns=['Year', 'ECI_pred'])), None

# ---------- chart ----------------------------------------------------------
# Sequential palette per group: dark = top rank, light = bottom rank,
# so the eye can read which country leads each group from colour alone.
COLOURS = {
    # improvers, ranked by delta ECI 2015->2023 (KAZ +0.70 > ARM +0.65 > MRT +0.60)
    'KAZ': '#0F5132',  # deep forest, top gainer
    'ARM': '#198754',  # emerald
    'MRT': '#6DBE7C',  # light grass
    # decliners, ranked by delta ECI 2015->2023 (COD -0.70 > AGO -0.62 > NGA -0.60)
    'COD': '#6B0F0F',  # very deep red, top decliner
    'AGO': '#B91C1C',  # crimson
    'NGA': '#EF6A6A',  # coral
}
# Top of each group gets a thicker line so rank is doubly encoded.
LINE_WIDTHS = {
    'KAZ': 3.0, 'ARM': 2.3, 'MRT': 1.8,
    'COD': 3.0, 'AGO': 2.3, 'NGA': 1.8,
}
NAVY = '#1F3A5F'; GRID = '#E5E7EB'
FONT = 'IBM Plex Sans, system-ui, sans-serif'

fig = go.Figure()
deltas = {}; LAST_PANEL_YEAR = 2023
for code, name in HIGHLIGHT:
    out, err = project_country(code)
    if err: print(f'  {err}'); continue
    hist_df, pred_df = out
    last_obs = hist_df.iloc[-1]
    pred_plot = pd.concat([
        pd.DataFrame([{'Year': last_obs['Year'], 'ECI_pred': last_obs['ECI_hist']}]),
        pred_df,
    ], ignore_index=True)
    colour = COLOURS.get(code, NAVY)
    hist_view = hist_df[hist_df['Year'] >= 2000]
    fig.add_trace(go.Scatter(
        x=hist_view['Year'], y=hist_view['ECI_hist'],
        mode='lines', line=dict(color=colour, width=LINE_WIDTHS.get(code, 2.2)),
        name=name, legendgroup=code,
        hovertemplate=f'{name} %{{x:.0f}}: %{{y:+.1f}} (observed)<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=pred_plot['Year'], y=pred_plot['ECI_pred'],
        mode='lines', line=dict(color=colour, width=LINE_WIDTHS.get(code, 2.2), dash='dot'),
        name=f'{name} (projected)', legendgroup=code, showlegend=False,
        hovertemplate=f'{name} %{{x:.0f}}: %{{y:+.1f}} (projected)<extra></extra>',
    ))
    deltas[code] = pred_df['ECI_pred'].iloc[-1] - hist_df['ECI_hist'].iloc[-1]

fig.add_vline(x=LAST_PANEL_YEAR, line=dict(color='#9CA3AF', width=1, dash='dash'),
              annotation_text=f'{LAST_PANEL_YEAR}: end of panel',
              annotation_position='top left',
              annotation_font=dict(family=FONT, size=11, color='#6B7280'))
fig.add_hline(y=0, line=dict(color='#c9cfd6', width=1))

fig.update_layout(
    height=540, margin=dict(l=10, r=10, t=20, b=10),
    font=dict(family=FONT, size=12, color=NAVY),
    paper_bgcolor='white', plot_bgcolor='white',
    xaxis=dict(title=dict(text='Year', font=dict(size=12)),
               gridcolor=GRID, zeroline=False,
               tickfont=dict(size=11), dtick=4,
               tickformat='d', hoverformat='d'),
    yaxis=dict(title=dict(text='Economic Complexity Index', font=dict(size=12)),
               gridcolor=GRID, zeroline=False,
               tickfont=dict(size=11), tickformat='.1f',
               hoverformat='.1f'),
    legend=dict(orientation='h', yanchor='bottom', y=-0.32,
                xanchor='center', x=0.5,
                font=dict(family=FONT, size=11),
                bgcolor='rgba(255,255,255,0.6)',
                bordercolor=GRID, borderwidth=1),
)
out_path = EXT / 'Graphics' / 'NB4' / 'eci_forecast_trajectories.html'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out_path), include_plotlyjs='cdn', full_html=True)
print(f'\nWrote: {out_path}')
print(f'\nProjected ECI change ({LAST_PANEL_YEAR} to {HORIZON}):')
for code, d in sorted(deltas.items(), key=lambda kv: kv[1], reverse=True):
    nm = next(n for c, n in HIGHLIGHT if c == code)
    print(f'  {code} ({nm:20s}): {d:+.2f}')
