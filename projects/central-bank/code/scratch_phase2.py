#!/usr/bin/env python3
"""
Phase 2 hypothesis tests for the central-bank page redesign.
Three ideas to test before committing to a structure:

  T1. Persistence finding has a forecasting payoff:
      a regime-aware AR vs an unconditional AR for 2021-22 inflation.
  T2. FAIT (Aug 2020) changed the Fed's inflation response coefficient:
      Taylor rule on three subsamples (pre-Volcker, anchored, post-FAIT).
  T3. The rule applied to late-2025 / current data:
      what does it prescribe today versus what the Fed is doing?
"""
import os, json, warnings, random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
SEED = 42; random.seed(SEED); np.random.seed(SEED)

DATA  = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/central-bank/data'
OUT   = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/central-bank/outputs/scratch'
os.makedirs(OUT, exist_ok=True)

# --- load + engineer (same recipe as scratch script) ---
def _read(names, path=DATA):
    for n in names:
        p = os.path.join(path, n)
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["observation_date"], index_col="observation_date")
    raise FileNotFoundError(names)

cpi=_read(["CPIAUCSL.csv"]); unrate=_read(["UNRATE.csv"])
ff=_read(["FEDFUNDS.csv","FEDFUNDS-1.csv"]); tcu=_read(["TCU.csv"])
gs10=_read(["GS10.csv"]); nfci=_read(["NFCI.csv"]).resample("MS").last()
nrou_q=_read(["NROU.csv"]); nrou_q.index=pd.DatetimeIndex(nrou_q.index)
nrou_m=nrou_q.resample("MS").interpolate("linear")
mich=_read(["MICH.csv"])
m=pd.concat([
    cpi.rename(columns={cpi.columns[0]:"cpi"}),
    unrate.rename(columns={unrate.columns[0]:"unemployment"}),
    ff.rename(columns={ff.columns[0]:"fed_funds"}),
    gs10.rename(columns={gs10.columns[0]:"treasury_10y"}),
    nrou_m.rename(columns={nrou_m.columns[0]:"nrou"}),
], axis=1).dropna(subset=["cpi","unemployment","fed_funds","nrou"])
m=m.join(mich.rename(columns={mich.columns[0]:"expected_inflation"}), how="left")

df = m.copy()
df["inflation"] = df["cpi"].pct_change(12)*100
df["unemp_gap"] = df["unemployment"] - df["nrou"]
df["adaptive_pi_e"] = df["inflation"].rolling(12).mean()
df["pi_expected"]   = df["expected_inflation"].fillna(df["adaptive_pi_e"])
for lag in [1,3,6,12]:
    df[f"L{lag}_inflation"] = df["inflation"].shift(lag)
df.index = pd.DatetimeIndex(df.index)
df = df.dropna(subset=["inflation","L12_inflation"])
print(f"Sample: N={len(df)}, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}")

VERDICTS = []
def verdict(name, status, note):
    VERDICTS.append((name, status, note))
    print(f"\n>> [{status}] {name}\n   {note}")

# =========================================================================
# T1: Does the persistence finding actually help forecast 2021-22 inflation?
#
# Setup: Forecaster sitting in Dec 2020 wants to predict inflation 12 months
# ahead at each month in 2021-2022. Two strategies:
#   (a) Unconditional: AR(4) trained on ALL available data through forecast
#       date. This is the standard approach.
#   (b) Regime-aware: AR(4) trained ONLY on low-regime data through forecast
#       date. This uses the persistence finding: if inflation is in a low
#       regime, only low-regime history is relevant for forecasting.
#
# Compare MSE / MAE on actual 2022 inflation. If (b) does worse, the
# persistence finding has no forecasting value. If (b) does better, we have
# a use case.
# =========================================================================
print("\n" + "="*70 + "\nT1: Regime-aware vs unconditional AR for 2021-22 inflation\n" + "="*70)

# Identify regimes the simple way: high regime if rolling 12m mean > 4%
df["regime_simple"] = (df["inflation"].rolling(12).mean() > 4.0).astype(int)

def ar_forecast(train_df, forecast_target_dates):
    """Fit AR(4) on lags 1,3,6,12 of inflation; predict 12m ahead at each target date."""
    feats = ["L1_inflation","L3_inflation","L6_inflation","L12_inflation"]
    X_tr = train_df[feats].dropna()
    y_tr = train_df.loc[X_tr.index, "inflation"]
    m = Ridge(alpha=0.01).fit(X_tr, y_tr)
    # For each forecast target, walk forward iteratively using lagged predictions
    forecasts = {}
    for tgt in forecast_target_dates:
        # Use the latest available row in train_df to forecast 12 months ahead.
        # Iterative: start from row 12 months before tgt; predict; carry forward.
        anchor = tgt - pd.DateOffset(months=12)
        if anchor not in train_df.index:
            anchor = train_df.index[train_df.index.get_indexer([anchor], method="nearest")[0]]
        # build the prediction iteratively
        history = list(train_df.loc[:anchor, "inflation"].values)
        for step in range(12):
            x = np.array([[history[-1], history[-3], history[-6], history[-12]]])
            pred = m.predict(x)[0]
            history.append(pred)
        forecasts[tgt] = history[-1]
    return forecasts

# Forecast inflation at each month in 2021 and 2022
targets = pd.date_range("2021-01-01","2022-12-01", freq="MS")
actuals = df.loc[targets, "inflation"]
print(f"  Targets: {len(targets)} months (2021-2022), actual inflation range: "
      f"{actuals.min():.1f}% to {actuals.max():.1f}%")

# Strategy A: unconditional, train through forecast date minus 12 months
results_a, results_b = [], []
for tgt in targets:
    anchor = tgt - pd.DateOffset(months=12)
    train_uncond = df.loc[:anchor].copy()
    train_lowonly = train_uncond[train_uncond["regime_simple"] == 0].copy()
    if len(train_lowonly) < 60: 
        results_b.append((tgt, np.nan)); continue
    fa = ar_forecast(train_uncond, [tgt])[tgt]
    fb = ar_forecast(train_lowonly, [tgt])[tgt]
    results_a.append((tgt, fa))
    results_b.append((tgt, fb))

pred_a = pd.Series({t:v for t,v in results_a})
pred_b = pd.Series({t:v for t,v in results_b})
both = pd.concat([actuals.rename("actual"), pred_a.rename("uncond"), pred_b.rename("lowonly")], axis=1).dropna()

print("\n  Sample of forecasts (target month, actual, uncond, lowonly):")
for d in [pd.Timestamp("2021-06-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-01"), pd.Timestamp("2022-12-01")]:
    if d in both.index:
        r = both.loc[d]
        print(f"    {d:%Y-%m}: actual={r['actual']:.2f}, uncond={r['uncond']:.2f}, lowonly={r['lowonly']:.2f}")

mse_a = ((both["actual"] - both["uncond"])**2).mean()
mse_b = ((both["actual"] - both["lowonly"])**2).mean()
mae_a = (both["actual"] - both["uncond"]).abs().mean()
mae_b = (both["actual"] - both["lowonly"]).abs().mean()
bias_a = (both["uncond"] - both["actual"]).mean()
bias_b = (both["lowonly"] - both["actual"]).mean()
print(f"\n  Unconditional AR:  MSE={mse_a:.2f}, MAE={mae_a:.2f}, mean error={bias_a:+.2f}")
print(f"  Low-regime AR:     MSE={mse_b:.2f}, MAE={mae_b:.2f}, mean error={bias_b:+.2f}")

if mse_b < mse_a * 0.85:
    verdict("T1 Regime-aware-forecast", "SUPPORTED",
            f"Low-regime model has MSE {mse_b:.2f} vs unconditional {mse_a:.2f}, a {(1-mse_b/mse_a)*100:.0f}% improvement. The persistence finding has a forecasting payoff.")
elif mse_b > mse_a * 1.15:
    verdict("T1 Regime-aware-forecast", "REJECTED",
            f"Low-regime model is WORSE: MSE {mse_b:.2f} vs {mse_a:.2f}. Conditioning on regime hurts more than it helps when you actually need to forecast.")
else:
    verdict("T1 Regime-aware-forecast", "MIXED",
            f"MSE roughly equal (regime {mse_b:.2f} vs uncond {mse_a:.2f}). Persistence finding doesn't translate to a forecasting advantage at this horizon.")

# =========================================================================
# T2: Did FAIT (Aug 2020) change the Fed's inflation response coefficient?
#
# Estimate the Taylor rule on three subsamples:
#   - Pre-Volcker: 1956-07 to 1979-07
#   - Anchored: 1985-01 to 2019-12
#   - Post-FAIT: 2020-08 to 2025-12
# Also run a ZLB-excluded post-FAIT (2022-03 onwards, after liftoff) since
# the ZLB period mechanically caps how aggressive the Fed can look.
#
# If the post-FAIT coefficient is materially lower than the anchored-period
# coefficient, that's evidence FAIT changed behaviour. If it's similar, FAIT
# was rhetoric.
# =========================================================================
print("\n" + "="*70 + "\nT2: FAIT framework change in the Taylor rule coefficient\n" + "="*70)

def taylor_ols(sub, inf="pi_expected"):
    s = sub.dropna(subset=["fed_funds",inf,"unemp_gap"]).copy()
    if len(s) < 30: return None
    X = sm.add_constant(s[[inf,"unemp_gap"]])
    return sm.OLS(s["fed_funds"], X).fit(cov_type="HAC", cov_kwds={"maxlags":min(12, len(s)//4)})

subs = [
    ("Pre-Volcker",   "1956-07", "1979-07"),
    ("Anchored",      "1985-01", "2019-12"),
    ("Post-FAIT",     "2020-08", "2025-12"),
    ("Post-liftoff",  "2022-03", "2025-12"),  # post-ZLB exit
]
fait_results = []
for label, s, e in subs:
    sub = df[(df.index >= s) & (df.index <= e)]
    res = taylor_ols(sub, "pi_expected")
    if res is None:
        print(f"  {label}: too few obs"); continue
    coef = res.params["pi_expected"]; se = res.bse["pi_expected"]; n = len(res.resid)
    print(f"  {label:<15} ({s} to {e}, N={n:3d}): 1+a_pi = {coef:.2f} (SE {se:.2f})")
    fait_results.append((label, s, e, n, coef, se))

# Compare the three substantive periods (skip post-liftoff as a robustness)
anchored = next(r for r in fait_results if r[0]=="Anchored")
postfait = next((r for r in fait_results if r[0]=="Post-FAIT"), None)
postliftoff = next((r for r in fait_results if r[0]=="Post-liftoff"), None)

if postfait is not None:
    diff = anchored[4] - postfait[4]
    pooled_se = np.sqrt(anchored[5]**2 + postfait[5]**2)
    z = diff / pooled_se if pooled_se > 0 else np.nan
    print(f"\n  Anchored minus Post-FAIT: {diff:+.2f} (pooled SE {pooled_se:.2f}, z={z:.2f})")
    
    if postliftoff is not None:
        diff2 = anchored[4] - postliftoff[4]
        pooled_se2 = np.sqrt(anchored[5]**2 + postliftoff[5]**2)
        z2 = diff2 / pooled_se2 if pooled_se2 > 0 else np.nan
        print(f"  Anchored minus Post-liftoff: {diff2:+.2f} (pooled SE {pooled_se2:.2f}, z={z2:.2f})")
    
    if abs(z) > 1.96 and diff > 0:
        verdict("T2 FAIT-changed-response", "SUPPORTED",
                f"Anchored 1+a_pi={anchored[4]:.2f}, Post-FAIT={postfait[4]:.2f}, diff {diff:+.2f} significant at 5% (z={z:.2f}). Evidence that the post-2020 Fed has a lower implied inflation response.")
    elif abs(z) > 1.0:
        verdict("T2 FAIT-changed-response", "QUALIFIED",
                f"Anchored {anchored[4]:.2f} vs Post-FAIT {postfait[4]:.2f}, diff {diff:+.2f}, z={z:.2f}. Suggestive but not statistically clean given short post-FAIT sample.")
    else:
        verdict("T2 FAIT-changed-response", "REJECTED",
                f"Anchored {anchored[4]:.2f} vs Post-FAIT {postfait[4]:.2f}, diff {diff:+.2f}, z={z:.2f}. Not distinguishable. FAIT looks like rhetoric, not a behavioural change.")

# =========================================================================
# T3: What does the rule prescribe for the most recent month in the data?
#
# Estimate the rule on the clean anchored window (1985-2019). Apply it to
# the last 12 months in the sample. Compare prescribed vs actual rate.
# Also try to fetch newer data if the user is in 2026 by web (skipped here;
# just use latest available).
# =========================================================================
print("\n" + "="*70 + "\nT3: What does the rule say about the most recent observation?\n" + "="*70)

train = df[(df.index >= "1985-01") & (df.index <= "2019-12")].dropna(
    subset=["fed_funds","pi_expected","unemp_gap"])
X_tr = sm.add_constant(train[["pi_expected","unemp_gap"]])
rule = sm.OLS(train["fed_funds"], X_tr).fit(cov_type="HAC", cov_kwds={"maxlags":12})
print(f"  Rule (1985-2019, N={len(train)}): i = {rule.params['const']:.2f} + "
      f"{rule.params['pi_expected']:.2f}*pi_e + ({rule.params['unemp_gap']:.2f})*gap")

recent = df.tail(13).dropna(subset=["pi_expected","unemp_gap","fed_funds"]).copy()
X_recent = sm.add_constant(recent[["pi_expected","unemp_gap"]])
recent["prescribed"] = rule.predict(X_recent)
recent["gap_signed"] = recent["fed_funds"] - recent["prescribed"]
print(f"\n  Last 12 months of data ({recent.index.min():%Y-%m} to {recent.index.max():%Y-%m}):")
print(f"  {'Month':<10} {'Inflation':>10} {'pi_exp':>8} {'unemp_gap':>10} {'Actual FFR':>12} {'Prescribed':>12} {'Gap':>8}")
for d, r in recent.iterrows():
    print(f"  {d:%Y-%m}   {r['inflation']:>9.2f}% {r['pi_expected']:>7.2f}% {r['unemp_gap']:>+9.2f} "
          f"{r['fed_funds']:>11.2f}% {r['prescribed']:>11.2f}% {r['gap_signed']:>+7.2f}")

last = recent.iloc[-1]
print(f"\n  As of {recent.index[-1]:%Y-%m}: actual FFR = {last['fed_funds']:.2f}%, "
      f"rule prescribes {last['prescribed']:.2f}% (gap {last['gap_signed']:+.2f}pp)")

mean_gap = recent["gap_signed"].mean()
if abs(mean_gap) < 0.5:
    verdict("T3 Current-policy", "ON-RULE",
            f"Last 12 months: mean gap {mean_gap:+.2f}pp. Current Fed is essentially following the anchored-era rule.")
elif mean_gap < -0.5:
    verdict("T3 Current-policy", "BELOW-RULE",
            f"Last 12 months: mean gap {mean_gap:+.2f}pp. Fed is running easier than the anchored-era rule prescribes.")
else:
    verdict("T3 Current-policy", "ABOVE-RULE",
            f"Last 12 months: mean gap {mean_gap:+.2f}pp. Fed is running tighter than the anchored-era rule.")

# =========================================================================
# Dump summary
# =========================================================================
results = {
    "sample": {"n": len(df), "start": str(df.index.min().date()), "end": str(df.index.max().date())},
    "T1": {
        "mse_uncond": float(mse_a), "mse_lowonly": float(mse_b),
        "mae_uncond": float(mae_a), "mae_lowonly": float(mae_b),
        "bias_uncond": float(bias_a), "bias_lowonly": float(bias_b),
        "n_forecasts": int(len(both)),
    },
    "T2": [
        {"label": l, "start": s, "end": e, "n": int(n), "one_plus_api": float(c), "se": float(se)}
        for l, s, e, n, c, se in fait_results
    ],
    "T3": {
        "rule_intercept": float(rule.params["const"]),
        "rule_one_plus_api": float(rule.params["pi_expected"]),
        "rule_a_u": float(rule.params["unemp_gap"]),
        "last_month": str(recent.index[-1].date()),
        "last_actual_ffr": float(last["fed_funds"]),
        "last_prescribed": float(last["prescribed"]),
        "last_gap": float(last["gap_signed"]),
        "mean_gap_12m": float(mean_gap),
    },
    "verdicts": [(n, s, note) for n, s, note in VERDICTS],
}
with open(os.path.join(OUT, "phase2_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("Verdict summary:")
for n, s, note in VERDICTS:
    print(f"  [{s}] {n}")
    print(f"      {note}")
print(f"\nWritten to {os.path.join(OUT, 'phase2_results.json')}")
