#!/usr/bin/env python3
"""
Sandbox: hypothesis testing for proposed central-bank page restructure.
Each test outputs a verdict line that gets collected at the end.
"""
import os, json, warnings, random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
warnings.filterwarnings("ignore")
SEED = 42; random.seed(SEED); np.random.seed(SEED)
DATA_PATH = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/central-bank/data'
OUT = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/central-bank/outputs/scratch'
os.makedirs(OUT, exist_ok=True)

# ── load + engineer (copied from notebook cell 0, path-fixed) ─────────
def load_data(path=DATA_PATH):
    def _read(names):
        for n in names:
            p = os.path.join(path, n)
            if os.path.exists(p):
                return pd.read_csv(p, parse_dates=["observation_date"], index_col="observation_date")
        raise FileNotFoundError(names)
    cpi    = _read(["CPIAUCSL.csv"])
    unrate = _read(["UNRATE.csv"])
    ff     = _read(["FEDFUNDS.csv", "FEDFUNDS-1.csv"])
    tcu    = _read(["TCU.csv"])
    gs10   = _read(["GS10.csv"])
    nfci   = _read(["NFCI.csv"]).resample("MS").last()
    nrou_q = _read(["NROU.csv"]); nrou_q.index = pd.DatetimeIndex(nrou_q.index)
    nrou_m = nrou_q.resample("MS").interpolate("linear")
    mich = _read(["MICH.csv"])
    m = pd.concat([
        cpi.rename(columns={cpi.columns[0]:"cpi"}),
        unrate.rename(columns={unrate.columns[0]:"unemployment"}),
        ff.rename(columns={ff.columns[0]:"fed_funds"}),
        tcu.rename(columns={tcu.columns[0]:"capacity_util"}),
        gs10.rename(columns={gs10.columns[0]:"treasury_10y"}),
        nfci.rename(columns={nfci.columns[0]:"fin_conditions"}),
        nrou_m.rename(columns={nrou_m.columns[0]:"nrou"}),
    ], axis=1).dropna(subset=["cpi","unemployment","fed_funds","nrou"])
    m = m.join(mich.rename(columns={mich.columns[0]:"expected_inflation"}), how="left")
    return m

RECESSIONS = [("1973-11","1975-03"),("1980-01","1980-07"),("1981-07","1982-11"),
    ("1990-07","1991-03"),("2001-03","2001-11"),("2007-12","2009-06"),("2020-02","2020-04")]

def engineer(data):
    df = data.copy()
    df["inflation"]    = df["cpi"].pct_change(12)*100
    df["unemp_gap_cbo"]   = df["unemployment"] - df["nrou"]
    df["unemp_gap_fixed"] = df["unemployment"] - 5.0
    df["term_spread"]  = df["treasury_10y"] - df["fed_funds"]
    df["adaptive_pi_e"] = df["inflation"].rolling(12).mean()
    df["pi_expected"]   = df["expected_inflation"].fillna(df["adaptive_pi_e"])
    df["ff_lag1"] = df["fed_funds"].shift(1)
    for v in ["inflation","unemp_gap_cbo","unemp_gap_fixed","fed_funds","term_spread"]:
        for lag in [1,3,6,12]:
            df[f"L{lag}_{v}"] = df[v].shift(lag)
    df["recession"] = 0
    for s,e in RECESSIONS:
        df.loc[(df.index >= s) & (df.index <= e), "recession"] = 1
    df.index = pd.DatetimeIndex(df.index)
    return df.dropna(subset=["inflation","L12_inflation"])

print("Loading data...")
df = engineer(load_data())
print(f"  N={len(df)}, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}")

# ========== fit Markov-switching once, freeze regimes ==================
print("\nFitting 2-state MS...")
endog = df["inflation"]; exog = sm.add_constant(df[["unemployment"]])
mod = sm.tsa.MarkovRegression(endog, k_regimes=2, exog=exog,
        switching_variance=True, switching_exog=True)
best, best_llf = None, -np.inf
for k in range(3):
    try:
        r = mod.fit(search_reps=8, random_state=SEED+k, disp=False, maxiter=300)
        if r.llf > best_llf: best_llf, best = r.llf, r
    except Exception: continue
sm_prob = best.smoothed_marginal_probabilities
# Order so 0 = low-inflation
means = {r: df.loc[sm_prob[r]>0.5, "inflation"].mean() for r in [0,1]}
order = sorted(means, key=means.get)
df["p_low"]  = sm_prob[order[0]].values
df["p_high"] = sm_prob[order[1]].values
df["regime"] = (df["p_high"] > df["p_low"]).astype(int)
print(f"  Low mean={df.loc[df.regime==0,'inflation'].mean():.2f}, High mean={df.loc[df.regime==1,'inflation'].mean():.2f}")
print(f"  N_low={int((df.regime==0).sum())}, N_high={int((df.regime==1).sum())}")

VERDICTS = []
def verdict(name, status, note):
    VERDICTS.append((name, status, note))
    print(f"\n>> [{status}] {name}\n   {note}")

# =====================================================================
# H1: Does the structural break disappear because of NAIRU, or just
#     because of the test? Run the same Chow test under three
#     specifications: fixed 5%, CBO NAIRU, and NO unemployment gap.
# =====================================================================
print("\n" + "="*65 + "\nH1: NAIRU specification and the 1980 break\n" + "="*65)

def chow_taylor(df_in, gap_col, break_date="1979-08"):
    """Simple reduced-form Taylor reaction with a gap. Run pre / post / pooled OLS, compute Chow F."""
    sub = df_in.dropna(subset=["fed_funds","inflation",gap_col]).copy()
    sub["c"] = 1.0
    X_cols = ["c","inflation",gap_col]
    pre  = sub[sub.index <  break_date]
    post = sub[sub.index >= break_date]
    full_rss = sm.OLS(sub["fed_funds"],  sub[X_cols]).fit().ssr
    pre_rss  = sm.OLS(pre["fed_funds"],  pre[X_cols]).fit().ssr
    post_rss = sm.OLS(post["fed_funds"], post[X_cols]).fit().ssr
    k = len(X_cols); n = len(sub)
    F = ((full_rss - (pre_rss+post_rss))/k) / ((pre_rss+post_rss)/(n-2*k))
    p = 1 - stats.f.cdf(F, k, n-2*k)
    return dict(F=F, p=p, pre_n=len(pre), post_n=len(post))

# (a) fixed 5%
r_fixed = chow_taylor(df, "unemp_gap_fixed")
# (b) CBO NAIRU
r_cbo   = chow_taylor(df, "unemp_gap_cbo")
# (c) NO gap at all (compare a tighter Taylor with only inflation)
def chow_no_gap(df_in, break_date="1979-08"):
    sub = df_in.dropna(subset=["fed_funds","inflation"]).copy()
    sub["c"]=1.0; X=["c","inflation"]
    pre = sub[sub.index < break_date]; post = sub[sub.index >= break_date]
    full = sm.OLS(sub["fed_funds"], sub[X]).fit().ssr
    pre_s= sm.OLS(pre["fed_funds"], pre[X]).fit().ssr
    post_s=sm.OLS(post["fed_funds"], post[X]).fit().ssr
    k=len(X); n=len(sub)
    F = ((full - (pre_s+post_s))/k) / ((pre_s+post_s)/(n-2*k))
    return dict(F=F, p=1 - stats.f.cdf(F,k,n-2*k))
r_none = chow_no_gap(df)

print(f"  Fixed 5% gap : F={r_fixed['F']:.2f}, p={r_fixed['p']:.4f}")
print(f"  CBO NAIRU gap: F={r_cbo['F']:.2f},   p={r_cbo['p']:.4f}")
print(f"  No gap       : F={r_none['F']:.2f}, p={r_none['p']:.4f}")

if r_fixed['p'] < 0.01 and r_cbo['p'] > 0.05:
    verdict("H1 NAIRU-drives-break", "SUPPORTED",
            f"Fixed 5% Chow p={r_fixed['p']:.3f} (sig), CBO NAIRU p={r_cbo['p']:.3f} (not sig). NAIRU choice flips the result.")
elif r_fixed['p'] > 0.05 and r_cbo['p'] > 0.05:
    verdict("H1 NAIRU-drives-break", "REJECTED",
            f"Neither spec finds a significant break at 1979-08. Original claim overstated.")
elif r_fixed['p'] < 0.05 and r_cbo['p'] < 0.05:
    verdict("H1 NAIRU-drives-break", "QUALIFIED",
            f"Both specs still find a significant break (fixed p={r_fixed['p']:.3f}, CBO p={r_cbo['p']:.3f}). NAIRU matters less than claimed.")
else:
    verdict("H1 NAIRU-drives-break", "MIXED",
            f"fixed p={r_fixed['p']:.3f}, CBO p={r_cbo['p']:.3f}, no-gap p={r_none['p']:.3f}")

# =====================================================================
# H2: Compositional persistence (Simpson's-paradox claim).
#     Decompose AC(12) into within-regime and between-regime parts.
# =====================================================================
print("\n" + "="*65 + "\nH2: Compositional persistence (within vs between regime)\n" + "="*65)

def ac(x, h):
    x = pd.Series(x).dropna()
    return x.autocorr(lag=h)

ac12_all  = ac(df["inflation"], 12)
ac12_low  = ac(df.loc[df.regime==0, "inflation"], 12)
ac12_high = ac(df.loc[df.regime==1, "inflation"], 12)
# Variance decomposition: Var(x) = E[Var(x|r)] + Var(E[x|r])
within  = df.groupby("regime")["inflation"].var().mean()
between = df.groupby("regime")["inflation"].mean().var()
total   = df["inflation"].var()
between_share = between / total

print(f"  AC(12) full sample : {ac12_all:.3f}")
print(f"  AC(12) low regime  : {ac12_low:.3f}")
print(f"  AC(12) high regime : {ac12_high:.3f}")
print(f"  Variance decomp: within={within:.2f}, between={between:.2f}, total={total:.2f}")
print(f"  Between-regime share of total variance: {between_share:.1%}")

if ac12_all > 0.6 and (ac12_low < 0.4 or ac12_high < 0.6) and between_share > 0.25:
    verdict("H2 Compositional-persistence", "SUPPORTED",
            f"AC(12) collapses from {ac12_all:.2f} (full) to {ac12_low:.2f} (low) / {ac12_high:.2f} (high); regime mean shift accounts for {between_share:.0%} of variance.")
else:
    verdict("H2 Compositional-persistence", "QUALIFIED",
            f"AC(12) full={ac12_all:.2f}, low={ac12_low:.2f}, high={ac12_high:.2f}, between-share={between_share:.0%}.")

# =====================================================================
# H3: Forecast horse race is mostly noise.
#     Compare AR(12), Ridge with all features, random walk, mean.
# =====================================================================
print("\n" + "="*65 + "\nH3: Long-horizon forecast horse race\n" + "="*65)

def expand_cv(df_in, target, features, model_fn, min_train=120):
    """Walk-forward expanding CV: at each t, train on [:t], predict t."""
    y = df_in[target].dropna()
    X = df_in[features].loc[y.index].fillna(0.0)
    preds, truths = [], []
    for i in range(min_train, len(y)):
        m = model_fn()
        m.fit(X.iloc[:i], y.iloc[:i])
        preds.append(m.predict(X.iloc[i:i+1])[0])
        truths.append(y.iloc[i])
    truths = np.array(truths); preds = np.array(preds)
    rmse = np.sqrt(np.mean((truths-preds)**2))
    # baseline = expanding mean
    base = np.array([y.iloc[:min_train+i].mean() for i in range(len(truths))])
    base_rmse = np.sqrt(np.mean((truths-base)**2))
    r2 = 1 - mean_squared_error(truths,preds)/mean_squared_error(truths,base)
    return dict(rmse=rmse, base_rmse=base_rmse, r2_vs_mean=r2, n=len(truths))

# build h-step ahead targets
df["inflation_12m"] = df["inflation"].shift(-12)
df["inflation_1m"]  = df["inflation"].shift(-1)

feats_simple = [f"L{l}_inflation" for l in [1,3,6,12]]
feats_full   = feats_simple + ["L1_unemp_gap_cbo","L1_term_spread","L1_fed_funds","p_high"]

print("  1-month horizon:")
r1_ar    = expand_cv(df.dropna(subset=["inflation_1m"]), "inflation_1m", feats_simple, lambda: Ridge(alpha=0.01))
r1_full  = expand_cv(df.dropna(subset=["inflation_1m"]), "inflation_1m", feats_full,   lambda: Ridge(alpha=0.01))
print(f"    AR(simple)   R2 vs mean = {r1_ar['r2_vs_mean']:.3f}, RMSE={r1_ar['rmse']:.3f}")
print(f"    Full Ridge   R2 vs mean = {r1_full['r2_vs_mean']:.3f}, RMSE={r1_full['rmse']:.3f}")

print("  12-month horizon:")
r12_ar   = expand_cv(df.dropna(subset=["inflation_12m"]), "inflation_12m", feats_simple, lambda: Ridge(alpha=0.01))
r12_full = expand_cv(df.dropna(subset=["inflation_12m"]), "inflation_12m", feats_full,   lambda: Ridge(alpha=0.01))
print(f"    AR(simple)   R2 vs mean = {r12_ar['r2_vs_mean']:.3f}, RMSE={r12_ar['rmse']:.3f}")
print(f"    Full Ridge   R2 vs mean = {r12_full['r2_vs_mean']:.3f}, RMSE={r12_full['rmse']:.3f}")

if r1_ar['r2_vs_mean'] > 0.3 and r12_ar['r2_vs_mean'] < 0:
    verdict("H3 Forecast-horse-race-is-noise", "SUPPORTED",
            f"1m AR R^2={r1_ar['r2_vs_mean']:.2f} (real signal). 12m AR R^2={r12_ar['r2_vs_mean']:.2f}, full Ridge R^2={r12_full['r2_vs_mean']:.2f} (both useless). Drop the multi-model table.")
else:
    verdict("H3 Forecast-horse-race-is-noise", "MIXED",
            f"1m AR R^2={r1_ar['r2_vs_mean']:.2f}, 12m AR R^2={r12_ar['r2_vs_mean']:.2f}, 12m full Ridge R^2={r12_full['r2_vs_mean']:.2f}.")

# =====================================================================
# H4: The realized-vs-expected-inflation flip in Taylor rule.
#     Estimate a clean OLS Taylor rule on each subsample with realized
#     and expected inflation. Report 1+a_pi each way.
# =====================================================================
print("\n" + "="*65 + "\nH4: Realized vs expected inflation in Taylor rule\n" + "="*65)

def taylor_ols(df_in, inf_col):
    """Static Taylor: i = c + (1+a_pi)*pi + a_u*(u-u*). Returns 1+a_pi and SE."""
    sub = df_in.dropna(subset=["fed_funds", inf_col, "unemp_gap_cbo"]).copy()
    X = sub[[inf_col, "unemp_gap_cbo"]].copy()
    X = sm.add_constant(X)
    m = sm.OLS(sub["fed_funds"], X).fit(cov_type="HAC", cov_kwds={"maxlags":12})
    return dict(one_plus_api=m.params[inf_col], se=m.bse[inf_col], n=len(sub))

print("  Full sample:")
fr_real = taylor_ols(df, "inflation")
fr_exp  = taylor_ols(df, "pi_expected")
print(f"    realized : 1+a_pi = {fr_real['one_plus_api']:.2f} ({fr_real['se']:.2f})")
print(f"    expected : 1+a_pi = {fr_exp['one_plus_api']:.2f} ({fr_exp['se']:.2f})")

print("  High-inflation regime:")
hi = df[df.regime==1]
hr_real = taylor_ols(hi, "inflation")
hr_exp  = taylor_ols(hi, "pi_expected")
print(f"    realized : 1+a_pi = {hr_real['one_plus_api']:.2f} ({hr_real['se']:.2f})")
print(f"    expected : 1+a_pi = {hr_exp['one_plus_api']:.2f} ({hr_exp['se']:.2f})")

print("  Low-inflation regime:")
lo = df[df.regime==0]
lr_real = taylor_ols(lo, "inflation")
lr_exp  = taylor_ols(lo, "pi_expected")
print(f"    realized : 1+a_pi = {lr_real['one_plus_api']:.2f} ({lr_real['se']:.2f})")
print(f"    expected : 1+a_pi = {lr_exp['one_plus_api']:.2f} ({lr_exp['se']:.2f})")

# Does using expected inflation flip the high-regime principle?
flip_high = (hr_real['one_plus_api'] < 1 and hr_exp['one_plus_api'] > 1)
if flip_high:
    verdict("H4 Real-vs-expected-flip", "SUPPORTED",
            f"High regime 1+a_pi: realized={hr_real['one_plus_api']:.2f}, expected={hr_exp['one_plus_api']:.2f}. Principle flips from violated to satisfied with expected inflation, as page claims.")
elif abs(hr_real['one_plus_api'] - hr_exp['one_plus_api']) > 0.3:
    verdict("H4 Real-vs-expected-flip", "QUALIFIED",
            f"Big shift in high-regime estimate (realized {hr_real['one_plus_api']:.2f} -> expected {hr_exp['one_plus_api']:.2f}) but doesn't cleanly cross 1.")
else:
    verdict("H4 Real-vs-expected-flip", "REJECTED",
            f"High-regime 1+a_pi barely moves (realized {hr_real['one_plus_api']:.2f} -> expected {hr_exp['one_plus_api']:.2f}).")

# =====================================================================
# H5: The Taylor-rule estimation is partly tautological because regimes
#     are inflation-defined. Test: how much of the high-regime "Taylor
#     principle is satisfied" is mechanical from regime construction?
#     Compare to a date-based 1979-2024 split (Volcker onward) on raw data.
# =====================================================================
print("\n" + "="*65 + "\nH5: Regime-based vs date-based Taylor rule\n" + "="*65)
post79 = df[df.index >= "1979-08"]
pre79  = df[df.index <  "1979-08"]
p79_real = taylor_ols(post79, "inflation")
p79_exp  = taylor_ols(post79, "pi_expected")
pre_real = taylor_ols(pre79,  "inflation")
pre_exp  = taylor_ols(pre79,  "pi_expected")
print(f"  Pre-1979-08  realized 1+a_pi = {pre_real['one_plus_api']:.2f} ({pre_real['se']:.2f}); expected = {pre_exp['one_plus_api']:.2f} ({pre_exp['se']:.2f})")
print(f"  Post-1979-08 realized 1+a_pi = {p79_real['one_plus_api']:.2f} ({p79_real['se']:.2f}); expected = {p79_exp['one_plus_api']:.2f} ({p79_exp['se']:.2f})")
print(f"  (For comparison: high-regime realized = {hr_real['one_plus_api']:.2f}, expected = {hr_exp['one_plus_api']:.2f})")

# date-based vs regime-based: do they tell the same story?
if abs(p79_exp['one_plus_api'] - hr_exp['one_plus_api']) < 0.25:
    verdict("H5 Regime-vs-date-equivalence", "REJECTED",
            f"Date-based post-79 expected 1+a_pi = {p79_exp['one_plus_api']:.2f} ~ high-regime {hr_exp['one_plus_api']:.2f}. Regime conditioning adds little over date split.")
else:
    verdict("H5 Regime-vs-date-equivalence", "SUPPORTED",
            f"Date split: post-79 expected 1+a_pi={p79_exp['one_plus_api']:.2f}, vs high-regime {hr_exp['one_plus_api']:.2f}. Regime conditioning gives a materially different answer.")

# =====================================================================
# H6: Counterfactual payoff. Estimate a Taylor rule on post-1985 data
#     (anchored-expectations period). Apply it back to 1972-75, 1979-82,
#     2021-23 and compare prescribed vs actual FFR. Is the deviation
#     large enough to be the page's "so what?".
# =====================================================================
print("\n" + "="*65 + "\nH6: Counterfactual rule (anchored era applied to crisis episodes)\n" + "="*65)

train = df[(df.index >= "1985-01") & (df.index < "2008-01")].dropna(
    subset=["fed_funds","pi_expected","unemp_gap_cbo"])
X_train = sm.add_constant(train[["pi_expected","unemp_gap_cbo"]])
rule = sm.OLS(train["fed_funds"], X_train).fit(cov_type="HAC", cov_kwds={"maxlags":12})
print(f"  Trained on 1985-2007 (n={len(train)}):")
print(f"    intercept = {rule.params['const']:.2f}")
print(f"    1+a_pi    = {rule.params['pi_expected']:.2f}")
print(f"    a_u       = {rule.params['unemp_gap_cbo']:.2f}")

episodes = [("Burns 1972-75","1972-01","1975-12"),
            ("Volcker 1979-82","1979-08","1982-12"),
            ("Post-COVID 2021-23","2021-01","2023-12")]
episode_results = []
for name, s, e in episodes:
    ep = df[(df.index >= s) & (df.index <= e)].dropna(
        subset=["fed_funds","pi_expected","unemp_gap_cbo"])
    Xep = sm.add_constant(ep[["pi_expected","unemp_gap_cbo"]])
    pred = rule.predict(Xep)
    actual = ep["fed_funds"]
    gap = actual - pred  # positive = tighter than rule
    avg_gap = gap.mean()
    print(f"  {name}: actual mean FFR={actual.mean():.2f}, rule mean={pred.mean():.2f}, gap={avg_gap:+.2f}")
    episode_results.append((name, actual.mean(), pred.mean(), avg_gap))

# Was Burns more lax than the rule? Volcker tighter? Post-COVID lax?
burns_gap = episode_results[0][3]
volcker_gap = episode_results[1][3]
covid_gap = episode_results[2][3]
if burns_gap < -1 and volcker_gap > 1:
    verdict("H6 Counterfactual-payoff", "SUPPORTED",
            f"Burns {burns_gap:+.1f}pp below rule, Volcker {volcker_gap:+.1f}pp above, post-COVID {covid_gap:+.1f}pp. Clean narrative: Burns underreacted, Volcker overcorrected.")
else:
    verdict("H6 Counterfactual-payoff", "QUALIFIED",
            f"Episode gaps: Burns {burns_gap:+.1f}pp, Volcker {volcker_gap:+.1f}pp, post-COVID {covid_gap:+.1f}pp. Story exists but isn't as crisp.")

# =====================================================================
# H7: LORO recession analysis is orphaned. Quick check whether the
#     term spread alone really gives AUC ~0.74 LORO, and whether
#     1990-91 and 2007-09 are the failure cases.
# =====================================================================
print("\n" + "="*65 + "\nH7: LORO recession prediction\n" + "="*65)
df["rec_12m_ahead"] = df["recession"].shift(-12).fillna(0)
sub = df.dropna(subset=["term_spread","rec_12m_ahead"]).copy()
sub["c"] = 1.0
recession_starts = [pd.Timestamp(s) for s,_ in RECESSIONS]
labels_per_rec = ["1973","1980","1981","1990","2001","2007","2020"]
aucs = []
for label, (s,e) in zip(labels_per_rec, RECESSIONS):
    s_t = pd.Timestamp(s); e_t = pd.Timestamp(e)
    in_window = (sub.index >= s_t - pd.Timedelta(days=400)) & (sub.index <= e_t + pd.Timedelta(days=400))
    train = sub[~in_window]
    test  = sub[in_window]
    if train["rec_12m_ahead"].sum() < 2 or test["rec_12m_ahead"].nunique() < 2:
        aucs.append((label, np.nan)); continue
    m = LogisticRegression(max_iter=500).fit(train[["term_spread"]], train["rec_12m_ahead"])
    pp = m.predict_proba(test[["term_spread"]])[:,1]
    try:
        a = roc_auc_score(test["rec_12m_ahead"], pp)
    except Exception:
        a = np.nan
    aucs.append((label, a))
    print(f"  Hold out {label}: AUC = {a:.3f}")

valid = [a for _, a in aucs if not np.isnan(a)]
mean_auc = np.mean(valid) if valid else np.nan
print(f"  Mean LORO AUC = {mean_auc:.3f}")
worst = sorted([(lbl,a) for lbl,a in aucs if not np.isnan(a)], key=lambda x: x[1])[:2]
if mean_auc > 0.6 and any(lbl in ["1990","2007"] for lbl,_ in worst):
    verdict("H7 LORO-orphaned", "SUPPORTED",
            f"Mean LORO AUC = {mean_auc:.2f}; worst episodes = {worst}. Story exists but is a standalone paper, not part of monetary-policy-rules thesis.")
else:
    verdict("H7 LORO-orphaned", "QUALIFIED",
            f"Mean LORO AUC = {mean_auc:.2f}; worst = {worst}.")

# =====================================================================
# Dump verdicts + numbers to JSON so the scratch HTML page can read them
# =====================================================================
results = {
    "n_obs": len(df),
    "date_range": f"{df.index.min():%Y-%m} to {df.index.max():%Y-%m}",
    "n_low_regime": int((df.regime==0).sum()),
    "n_high_regime": int((df.regime==1).sum()),
    "H1": {
        "fixed_F": float(r_fixed["F"]), "fixed_p": float(r_fixed["p"]),
        "cbo_F": float(r_cbo["F"]),     "cbo_p": float(r_cbo["p"]),
        "nogap_F": float(r_none["F"]),  "nogap_p": float(r_none["p"]),
    },
    "H2": {
        "ac12_full": float(ac12_all),
        "ac12_low": float(ac12_low),
        "ac12_high": float(ac12_high),
        "within_var": float(within),
        "between_var": float(between),
        "total_var": float(total),
        "between_share": float(between_share),
    },
    "H3": {
        "h1_ar_r2":   float(r1_ar["r2_vs_mean"]),  "h1_ar_rmse": float(r1_ar["rmse"]),
        "h1_full_r2": float(r1_full["r2_vs_mean"]),"h1_full_rmse": float(r1_full["rmse"]),
        "h12_ar_r2":   float(r12_ar["r2_vs_mean"]), "h12_ar_rmse": float(r12_ar["rmse"]),
        "h12_full_r2": float(r12_full["r2_vs_mean"]), "h12_full_rmse": float(r12_full["rmse"]),
    },
    "H4": {
        "full_real": float(fr_real["one_plus_api"]), "full_real_se": float(fr_real["se"]),
        "full_exp":  float(fr_exp["one_plus_api"]),  "full_exp_se":  float(fr_exp["se"]),
        "high_real": float(hr_real["one_plus_api"]), "high_real_se": float(hr_real["se"]),
        "high_exp":  float(hr_exp["one_plus_api"]),  "high_exp_se":  float(hr_exp["se"]),
        "low_real":  float(lr_real["one_plus_api"]), "low_real_se":  float(lr_real["se"]),
        "low_exp":   float(lr_exp["one_plus_api"]),  "low_exp_se":   float(lr_exp["se"]),
    },
    "H5": {
        "pre79_real": float(pre_real["one_plus_api"]), "pre79_exp": float(pre_exp["one_plus_api"]),
        "post79_real": float(p79_real["one_plus_api"]), "post79_exp": float(p79_exp["one_plus_api"]),
        "highregime_real": float(hr_real["one_plus_api"]), "highregime_exp": float(hr_exp["one_plus_api"]),
    },
    "H6": {
        "intercept": float(rule.params["const"]),
        "one_plus_api": float(rule.params["pi_expected"]),
        "a_u": float(rule.params["unemp_gap_cbo"]),
        "episodes": [(n, float(am), float(pm), float(g)) for n, am, pm, g in episode_results],
    },
    "H7": {
        "loro_auc": [(lbl, None if np.isnan(a) else float(a)) for lbl, a in aucs],
        "mean_loro_auc": float(mean_auc) if not np.isnan(mean_auc) else None,
    },
    "VERDICTS": [(n, s, note) for n, s, note in VERDICTS],
}

with open(os.path.join(OUT, "hypothesis_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n" + "="*65)
print("Verdict summary:")
for n, s, note in VERDICTS:
    print(f"  [{s}] {n}")
    print(f"      {note}")
print(f"\nResults written to: {os.path.join(OUT, 'hypothesis_results.json')}")
