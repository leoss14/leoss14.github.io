"""
patch_e2_covid_interactions.py
==============================
Applied: 2026-05-21

Adds 4 disaggregated COVID-interaction variables to e2_data_prep.ipynb,
mirroring the new HS27 sub-code shares produced by e1b cells 17-20.

Changes in e2_data_prep.ipynb
-----------------------------

Edit 1 (Section 8, post_2019 interactions loop):
  Extends the (share_col, name) list with four entries:
    ('coal_share',         'post2019_x_coal_share')
    ('crude_oil_share',    'post2019_x_crude_oil_share')
    ('refined_oil_share',  'post2019_x_refined_oil_share')
    ('gas_share',          'post2019_x_gas_share')
  Pre-existing entries (hydrocarbon, ores, base_metals) kept unchanged
  so the aggregate-HS27 interaction remains available as a robustness
  specification in e5.

Edit 2 (Section 10, POST_MICE_CLIPS dict):
  Adds 8 entries clipping the new variables to [0, 1]:
    coal_share, crude_oil_share, refined_oil_share, gas_share
    post2019_x_coal_share, post2019_x_crude_oil_share,
    post2019_x_refined_oil_share, post2019_x_gas_share

Why this is needed
------------------
e1b now writes coal_share, crude_oil_share, refined_oil_share, gas_share
to trade_metrics.csv. Without Edit 1, e2 builds Master_v2 without their
post_2019 interactions, and e5 cannot estimate disaggregated COVID
elasticities. Without Edit 2, MICE may overshoot the [0, 1] support for
demeaned share imputations.

How the edits were made
-----------------------
Applied in-place via Desktop Commander's edit_block on the notebook JSON,
inserting new lines immediately before the closing list/dict delimiters
of the relevant cells. Original lines were not modified, only appended to.

Verification
------------
After both edits the notebook still parses as valid JSON. The new variables
flow through:
  Section 8         -> 4 new interaction columns added to df
  Section 9         -> TRADE_USD_COLS exclusion does NOT catch share cols
                       (they don't end in '_usd' and aren't all-digit),
                       so they ARE interpolated normally
  Section 10 MICE   -> share cols are in mice_cols and get imputed
                       under the country-demeaned random-forest model;
                       post-MICE clip pulls any overshoots back to [0, 1]
  Outputs           -> Master_v2_imputations.parquet now contains 4 new
                       share columns + 4 new interaction columns

Next downstream patches still queued
------------------------------------
- e5_regressions: add the 4 new interactions to CONTROLS_FULL and
  INTERACTIONS_PARSIMONIOUS specs
- e3_clusters: decide whether to replace hydrocarbon_share with the 4
  sub-shares in RR_FEATURES (Option A: 7 features) or add alongside
  (Option B: 8 features), then update e3 section 13
"""
