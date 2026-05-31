"""
_config.py for extension_2024/

Mirrors the schema of New code/_config.py but with extended year range and
local ECI source. Downstream notebooks (once forked) should import from this
file rather than from the parent _config.py to keep the two pipelines
fully independent.
"""

import os

# Paths
HERE         = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS    = os.path.join(HERE, "artifacts")
INTERMEDIARY = os.path.join(HERE, "intermediary")
RAWDATA      = os.path.join(INTERMEDIARY, "rawdata")
CACHE        = os.path.join(INTERMEDIARY, "cache")

os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(INTERMEDIARY, exist_ok=True)
os.makedirs(RAWDATA, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# Year range. Capped at 2023 (PWT 11.0 ceiling). Atlas and trade_metrics
# extend to 2024 but PWT does not, so the panel is held complete by
# dropping 2024. Revisit once PWT publishes 2024.
YEAR_MIN, YEAR_MAX = 1995, 2023

# Local Atlas file (downloaded directly from atlas.cid.harvard.edu, May 2026).
# Contains eci_sitc, eci_hs92, eci_hs12, and growth_proj for 145 countries.
ATLAS_FILE = os.path.join(RAWDATA, "growth_proj_eci_rankings-1.csv")

# Which Atlas classification to use as the ECI series.
# hs92 chosen because (a) complete 1995-2024 coverage, (b) closest correlation
# to the live pipeline's current ECI column at 0.957, minimising coefficient
# drift when migrating regressions over.
ECI_CLASSIFICATION = "hs92"  # one of: "hs92", "hs12", "sitc"

# Whether to also retain Atlas's own growth projection column. Used by the
# forecast notebook as a benchmark against which to validate the in-house
# random forest forecasts.
KEEP_GROWTH_PROJ = True

# Sample definition (copied verbatim from New code/_config.py for parity)
GULF_STATES = ["BHR", "KWT", "OMN", "QAT", "SAU", "ARE"]
NR_THRESHOLD = float(os.environ.get("NR_THRESHOLD", 0.0))
CLUSTER_NR_THRESHOLD = float(os.environ.get("CLUSTER_NR_THRESHOLD", 1.0))

HIGH_INCOME_1995 = {
    "AUS", "AUT", "BEL", "CAN", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA",
    "GBR", "GRC", "IRL", "ISL", "ITA", "JPN", "LUX", "NLD", "NOR", "NZL",
    "PRT", "SWE", "USA",
    "ARE", "BHR", "BHS", "BRB", "BRN", "CYP", "HKG", "ISR", "KWT", "MLT",
    "QAT", "SGP",
}

# Variable lists (matches parent for downstream compatibility)
ECI_COL = "Economic Complexity Index"

LOG_COLS = [
    "Human capital index",
    "Total_Production_Value_Per_Capita",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP",
    "Government revenue",
    "Use of IMF credit (DOD, current US$)",
]

# Design decisions for the extended panel.
# These default to "permissive" (keep all observations) and are intended to be
# overridden by environment variable when running diagnostic comparisons.

# Balanced vs unbalanced panel. False = keep all observations including
# country-years with predictor gaps. True = drop country-years missing any
# predictor in IMPORTANT_VARS. Default permissive; revisit after seeing
# coverage diagnostics.
BALANCED_PANEL = bool(int(os.environ.get("BALANCED_PANEL", "0")))

# Add a post-2019 dummy interacted with resource-dependence to absorb the
# COVID structural break. Recommended default True for any post-2019
# regression. Has no effect if YEAR_MAX <= 2019.
ADD_COVID_DUMMY = bool(int(os.environ.get("ADD_COVID_DUMMY", "1")))

# PWT upgrade flag. The "1950-2019, HCI stops at 2019" claim in earlier
# comments was wrong; PWT 11.0 covers every variable in pwt_variables
# through 2023. Flag retained for compatibility but has no effect on the
# current pipeline. Revisit when PWT 11.1 ships with 2024.
USE_PWT_11_1 = bool(int(os.environ.get("USE_PWT_11_1", "0")))
