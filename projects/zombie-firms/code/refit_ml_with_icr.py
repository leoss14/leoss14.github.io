#!/usr/bin/env python3
"""
refit_ml_with_icr.py

Runs the XGBoost early-warning model WITH ICR features included, to give
the comparison number against the leakage-free spec already saved in
nb5_ml_results.csv. Output: two-row CSV with AUPRC for both specs.

Time: ~30 seconds on a 2024 laptop.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

SRC = Path("/Users/leoss/Desktop/Thesis Replication/output")
OUT = Path("/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/zombie-firms/outputs/scratch")
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_ZOMBIE = "zombie_mcgowan"
TRAIN_YEARS = list(range(2016, 2022))
VAL_YEARS   = [2022]
TEST_YEARS  = [2023]
HORIZON     = 1

print("Loading panel...")
df = pd.read_parquet(SRC / "zombie_panel_classified.parquet")
print(f"  {len(df):,} rows, {df['bvd_id'].nunique():,} firms")

def winsorise(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

df = df.sort_values(["bvd_id", "year"]).copy()
df["log_ta"]        = np.log(df["total_assets"].clip(lower=1))
df["icr_w"]         = winsorise(df["icr"])
df["roa_w"]         = winsorise(df["roa"])
df["inv_rate_w"]    = winsorise(df["investment_rate"])
df["fin_intensity"] = df["financial_expenses"] / df["total_assets"].replace(0, np.nan)
df["fin_int_w"]     = winsorise(df["fin_intensity"])
df["leverage"]      = df["total_liabilities"] / df["total_assets"].replace(0, np.nan)
df["leverage_w"]    = winsorise(df["leverage"])
df["log_emp"]       = np.log(df["employees"].clip(lower=1))
df["emp_growth"]    = df.groupby("bvd_id")["log_emp"].diff()
df["emp_growth_w"]  = winsorise(df["emp_growth"])
df["sales_growth"]  = df.groupby("bvd_id")["turnover"].pct_change().clip(-2, 2)
for col in ["icr_w", "roa_w", "leverage_w"]:
    df[f"{col}_l1"] = df.groupby("bvd_id")[col].shift(1)
    df[f"{col}_l2"] = df.groupby("bvd_id")[col].shift(2)
df["d_icr"]      = df.groupby("bvd_id")["icr_w"].diff()
df["d_roa"]      = df.groupby("bvd_id")["roa_w"].diff()
df["d_leverage"] = df.groupby("bvd_id")["leverage_w"].diff()
df["neg_equity_f"] = df["neg_equity"].fillna(0)
df["log_age"]      = np.log(df["firm_age"].clip(lower=1))
nace_dummies = pd.get_dummies(df["nace_2digit"].astype(str), prefix="nace", drop_first=True)
df = pd.concat([df, nace_dummies.astype(float)], axis=1)
nace_cols = list(nace_dummies.columns)

BASE_NO_ICR = ["leverage_w","fin_int_w","inv_rate_w","emp_growth_w","sales_growth",
               "log_ta","log_emp","neg_equity_f",
               "leverage_w_l1","leverage_w_l2","d_leverage"]
ICR_COLS    = ["icr_w","d_icr","icr_w_l1","icr_w_l2"]

results = []
for label, features in [("Without ICR (leakage-free)", BASE_NO_ICR),
                        ("With ICR lags (leaky baseline)", BASE_NO_ICR + ICR_COLS)]:
    print(f"\n=== {label} ===")
    feats = features + nace_cols
    df["target"] = df.groupby("bvd_id")[PRIMARY_ZOMBIE].shift(-HORIZON)
    ml = df[df["target"].notna() & df[features].notna().all(axis=1)].copy()
    train = ml[ml["year"].isin(TRAIN_YEARS)]
    val   = ml[ml["year"].isin(VAL_YEARS)]
    test  = ml[ml["year"].isin(TEST_YEARS)]
    print(f"  N train={len(train):,}, val={len(val):,}, test={len(test):,}")
    print(f"  Zombie rate in test: {test['target'].mean():.4f}")

    X_tr, y_tr = train[feats].values, train["target"].values.astype(int)
    X_va, y_va = val[feats].values,   val["target"].values.astype(int)
    X_te, y_te = test[feats].values,  test["target"].values.astype(int)
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # Logistic
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1, random_state=42)
    sc = StandardScaler().fit(X_tr)
    lr.fit(sc.transform(X_tr), y_tr)
    pr_lr = lr.predict_proba(sc.transform(X_te))[:,1]
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    pr_rf = rf.predict_proba(X_te)[:,1]
    # XGBoost
    xg = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5,
                            scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
                            eval_metric="aucpr", early_stopping_rounds=30, random_state=42,
                            verbosity=0)
    xg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pr_xg = xg.predict_proba(X_te)[:,1]

    for name, pr in [("Logistic", pr_lr), ("Random Forest", pr_rf), ("XGBoost", pr_xg)]:
        auprc = average_precision_score(y_te, pr)
        roc = roc_auc_score(y_te, pr)
        results.append({"spec": label, "model": name, "test_auprc": auprc, "test_roc": roc,
                        "test_zombie_rate": float(test["target"].mean())})
        print(f"  {name:<15} test AUPRC = {auprc:.3f}, ROC-AUC = {roc:.3f}")

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "ml_icr_comparison.csv", index=False)
print(f"\nSaved {OUT/'ml_icr_comparison.csv'}")
print("\nSummary:")
print(res_df.to_string(index=False))
