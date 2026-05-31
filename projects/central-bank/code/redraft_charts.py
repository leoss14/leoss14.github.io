#!/usr/bin/env python3
"""
Generate new charts for the redrafted central-bank page.
Two charts:
  1. taylor_date_split.png   - pre/post-1979 Taylor coefficients (H5)
  2. taylor_counterfactual.png - three episodes vs 1985-2007 rule (H6)
"""
import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import statsmodels.api as sm
warnings.filterwarnings("ignore")
SEED = 42; random.seed(SEED); np.random.seed(SEED)

DATA  = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/central-bank/data'
OUT   = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/central-bank/outputs'

# Matplotlib styling - same family as existing PNGs
C_BG = "#fafafa"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Public Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 11,
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9cfd6", "axes.linewidth": 0.8,
    "axes.labelcolor": "#4b5563",
    "xtick.color": "#5a6675", "ytick.color": "#5a6675",
    "legend.frameon": True, "legend.edgecolor": "#dde1e7",
    "legend.facecolor": "#fafafa", "legend.fontsize": 10,
})
NAVY = "#1a2744"
ACCENT = "#c23a3a"
GREEN = "#2e7d4a"
BLUE = "#2563eb"
SLATE = "#3d4f5f"

# --- load data (same recipe) ---
RECESSIONS = [("1973-11","1975-03"),("1980-01","1980-07"),("1981-07","1982-11"),
    ("1990-07","1991-03"),("2001-03","2001-11"),("2007-12","2009-06"),("2020-02","2020-04")]

def load_data():
    def _read(names):
        for n in names:
            p = os.path.join(DATA, n)
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
        tcu.rename(columns={tcu.columns[0]:"capacity_util"}),
        gs10.rename(columns={gs10.columns[0]:"treasury_10y"}),
        nfci.rename(columns={nfci.columns[0]:"fin_conditions"}),
        nrou_m.rename(columns={nrou_m.columns[0]:"nrou"}),
    ], axis=1).dropna(subset=["cpi","unemployment","fed_funds","nrou"])
    m=m.join(mich.rename(columns={mich.columns[0]:"expected_inflation"}), how="left")
    return m

def engineer(d):
    df=d.copy()
    df["inflation"]=df["cpi"].pct_change(12)*100
    df["unemp_gap"]=df["unemployment"]-df["nrou"]
    df["adaptive_pi_e"]=df["inflation"].rolling(12).mean()
    df["pi_expected"]=df["expected_inflation"].fillna(df["adaptive_pi_e"])
    df["L12_inflation"]=df["inflation"].shift(12)
    df.index=pd.DatetimeIndex(df.index)
    return df.dropna(subset=["inflation","L12_inflation"])

print("Loading data..."); df=engineer(load_data())
print(f"  N={len(df)}, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}")

def shade_recessions(ax):
    for s, e in RECESSIONS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.10, color="#fca5a5", zorder=0)

def taylor_ols(d, inf="pi_expected"):
    s=d.dropna(subset=["fed_funds",inf,"unemp_gap"]).copy()
    X=sm.add_constant(s[[inf,"unemp_gap"]])
    return sm.OLS(s["fed_funds"], X).fit(cov_type="HAC", cov_kwds={"maxlags":12})

# ============================================================
# CHART 1: Date-split Taylor coefficients (H5)
# ============================================================
print("\nGenerating chart 1: date-split Taylor coefficients...")
pre=df[df.index < "1979-08"]; post=df[df.index >= "1979-08"]
m_pre_r=taylor_ols(pre,"inflation"); m_pre_e=taylor_ols(pre,"pi_expected")
m_post_r=taylor_ols(post,"inflation"); m_post_e=taylor_ols(post,"pi_expected")

# Layout: 2x1 bar charts for 1+a_pi pre vs post, real vs expected
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
specs = [("Realized inflation", "inflation", m_pre_r, m_post_r),
         ("Expected inflation",  "pi_expected", m_pre_e, m_post_e)]
for ax, (label, col, pre_m, post_m) in zip(axes, specs):
    vals = [pre_m.params[col], post_m.params[col]]
    errs = [1.96*pre_m.bse[col], 1.96*post_m.bse[col]]
    xs = [0, 1]
    bars = ax.bar(xs, vals, color=[SLATE, NAVY], width=0.55,
                  yerr=errs, capsize=8, error_kw={"ecolor":"#4b5563","linewidth":1.2})
    ax.axhline(1.0, color=ACCENT, linestyle="--", linewidth=1.4, alpha=0.85,
               label="Taylor principle threshold (1+a$_\\pi$ = 1)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Pre-1979-08\n(N=" + str(len(pre)) + ")", "Post-1979-08\n(N=" + str(len(post)) + ")"])
    ax.set_title(label, color=NAVY, fontweight="bold", fontsize=12)
    ax.set_ylabel("1 + a$_\\pi$ (long-run inflation response)") if ax is axes[0] else None
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.06, f"{v:.2f}", ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_ylim(0, max(2.2, max(vals)+max(errs)+0.2))
    ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Taylor rule estimated by date split, not by inflation regime",
             fontsize=12.5, fontweight="bold", color=NAVY, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "taylor_date_split.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  saved taylor_date_split.png")

# ============================================================
# CHART 2: Counterfactual rule applied to three episodes (H6)
# Train rule on 1985-2007, plot prescribed vs actual on each episode
# ============================================================
print("\nGenerating chart 2: counterfactual rule, three episodes...")
train = df[(df.index >= "1985-01") & (df.index < "2008-01")].dropna(
    subset=["fed_funds","pi_expected","unemp_gap"])
X_train = sm.add_constant(train[["pi_expected","unemp_gap"]])
rule = sm.OLS(train["fed_funds"], X_train).fit(cov_type="HAC", cov_kwds={"maxlags":12})
print(f"  rule: i = {rule.params['const']:.2f} + {rule.params['pi_expected']:.2f}*pi_e "
      f"+ ({rule.params['unemp_gap']:.2f})*gap")

episodes = [
    ("Burns 1972-75",   "1972-01", "1975-12", "#dbeafe"),  # light blue
    ("Volcker 1979-82", "1979-08", "1982-12", "#ede9fe"),  # light purple
    ("Post-COVID 2021-23", "2021-01", "2023-12", "#fee2e2"),  # light red
]

# Pre-compute predictions to determine a common y-range
all_actual, all_pred = [], []
for name, s, e, _ in episodes:
    ep = df[(df.index >= s) & (df.index <= e)].dropna(
        subset=["fed_funds","pi_expected","unemp_gap"])
    Xep = sm.add_constant(ep[["pi_expected","unemp_gap"]])
    all_actual.append(ep["fed_funds"])
    all_pred.append(rule.predict(Xep))
y_max = max(max(a.max(), p.max()) for a, p in zip(all_actual, all_pred)) + 1.5
y_min = 0

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
gap_summary = []
for ax, (name, s, e, shade) in zip(axes, episodes):
    ep = df[(df.index >= s) & (df.index <= e)].dropna(
        subset=["fed_funds","pi_expected","unemp_gap"])
    Xep = sm.add_constant(ep[["pi_expected","unemp_gap"]])
    pred = rule.predict(Xep)
    actual = ep["fed_funds"]
    avg_gap = (actual - pred).mean()
    gap_summary.append((name, actual.mean(), pred.mean(), avg_gap))
    ax.fill_between(ep.index, y_min, y_max, color=shade, alpha=0.4, zorder=0)
    ax.plot(ep.index, actual, color=NAVY, linewidth=2.2, label="Actual fed funds rate")
    ax.plot(ep.index, pred, color=ACCENT, linewidth=2.2, linestyle="--",
            label="Rule prescription (1985-2007 fit)")
    ax.set_title(name, color=NAVY, fontweight="bold", fontsize=11.5)
    if ax is axes[0]:
        ax.set_ylabel("Fed funds rate (percent)")
    sign = "+" if avg_gap >= 0 else ""
    ax.text(0.04, 0.96, f"Mean gap: {sign}{avg_gap:.1f} pp",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=NAVY, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=NAVY, linewidth=0.8))
    ax.legend(loc="lower right", fontsize=8.5)
    # Yearly major ticks, formatted as YYYY
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.set_ylim(y_min, y_max)

fig.suptitle("Three episodes compared to a Taylor rule trained on 1985-2007",
             fontsize=12.5, fontweight="bold", color=NAVY, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "taylor_counterfactual.png"), dpi=130, bbox_inches="tight")
plt.close()
print("  saved taylor_counterfactual.png")

print("\nSummary table:")
print(f"  {'Episode':<22} {'Actual':>8} {'Rule':>8} {'Gap':>8}")
for name, a, p, g in gap_summary:
    print(f"  {name:<22} {a:>8.2f} {p:>8.2f} {g:>+8.2f}")
print("\nDone.")
