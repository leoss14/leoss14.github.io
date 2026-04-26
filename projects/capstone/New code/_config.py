"""
_config.py — Shared constants for the capstone visualization pipeline.

All prep notebooks import from this file, so paths, sample definitions,
feature lists, name maps, and colour palettes live in exactly one place.

If the data location moves, change INTERMEDIARY here and nothing else.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(HERE)
ARTIFACTS    = os.path.join(ROOT, "artifacts")

# Default location of the main capstone CLEAN/intermediary folder.
# Change this if your local checkout lives elsewhere.
INTERMEDIARY = os.environ.get(
    "CAPSTONE_INTERMEDIARY",
    "/Users/leoss/Desktop/GitHub/Portfolio/projects/capstone/New code/intermediary",
)

os.makedirs(ARTIFACTS, exist_ok=True)


# ── Sample definition ───────────────────────────────────────────────────────
# Forest-adjusted: subtract forest rents from total NR rents, apply 1% threshold,
# guarantee Gulf states inclusion. This is the headline sample used by every
# chart. Other thresholds (3%, 2%, 1%) are computed inside notebook 4 for the
# across-samples robustness charts.
GULF_STATES = ["BHR", "KWT", "OMN", "QAT", "SAU", "ARE"]
NR_THRESHOLD = 1.0  # percent of GDP, after subtracting forestry
YEAR_MIN, YEAR_MAX = 1995, 2019


# ── Variable lists ──────────────────────────────────────────────────────────
ECI_COL = "Economic Complexity Index"

LOG_COLS = [
    "Human capital index",
    "Total_Production_Value_Per_Capita",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP",
    "Government revenue",
    "Use of IMF credit (DOD, current US$)",
]

# Features used by the ML models (LASSO/Ridge/EN/RF) — matches V5 exactly.
BASE_FEATS = [
    "Total_Production_Value_Per_Capita",
    "Human capital index",
    "Rule of law index",
    "Political stability \u2014 estimate",
    "Trade (% of GDP)",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP",
    "Share of investment in GDP",
    "Domestic credit to private sector (% of GDP)",
    "Landlocked",
    "Urban population (% of total population)",
    "Government revenue",
    "Capital depreciation rate",
    "Use of IMF credit (DOD, current US$)",
    "Real interest rate (%)",
    "Inflation, consumer prices (annual %)",
    "Access to electricity (% of population)",
    "Adjusted savings: gross savings (% of GNI)",
    "L1_ECI",
    "Forestry rents (% of GDP)",
]
INTERACTION_FEATS = ["HCI_x_ProductionValue", "GFCF_x_ProductionValue"]
ALL_FEATS = BASE_FEATS + INTERACTION_FEATS

# Variables used by the headline regression (Model 3b).
REG_VARS = [
    "log_HCI",
    "log_GFCF",
    "Political stability \u2014 estimate",
    "Rule of law index",
    "log_Production_Value",
    "Forestry rents (% of GDP)",
    "Trade (% of GDP)",
]
INTERACT = [
    "log_HCI_x_log_Production",
    "log_GFCF_x_log_Production",
    "log_HCI_x_forestry_rents",
    "log_GFCF_x_forestry_rents",
]


# ── Display short names (used by every chart that shows a coefficient) ──────
NAME_MAP = {
    "Total_Production_Value_Per_Capita": "Production Value",
    "Human capital index": "Human Capital",
    "Rule of law index": "Rule of Law",
    "Political stability \u2014 estimate": "Political Stability",
    "Trade (% of GDP)": "Trade",
    "Gross fixed capital formation, all, Constant prices, Percent of GDP": "Capital Formation",
    "Share of investment in GDP": "Investment Share",
    "Domestic credit to private sector (% of GDP)": "Domestic Credit",
    "Landlocked": "Landlocked",
    "Urban population (% of total population)": "Urban Population",
    "Government revenue": "Gov Revenue",
    "Capital depreciation rate": "Depreciation",
    "Use of IMF credit (DOD, current US$)": "IMF Credit",
    "Real interest rate (%)": "Real Rate",
    "Inflation, consumer prices (annual %)": "Inflation",
    "Access to electricity (% of population)": "Electricity",
    "Adjusted savings: gross savings (% of GNI)": "Gross Savings",
    "L1_ECI": "Lagged ECI",
    "Forestry rents (% of GDP)": "Forestry rents (% GDP)",
    "HCI_x_ProductionValue": "HC \u00d7 Production",
    "GFCF_x_ProductionValue": "GFCF \u00d7 Production",
    "log_HCI": "Human Capital",
    "log_GFCF": "Capital Formation",
    "log_Production_Value": "Production Value",
    "log_HCI_x_log_Production": "HCI \u00d7 Production",
    "log_GFCF_x_log_Production": "GFCF \u00d7 Production",
    "log_HCI_x_forestry_rents": "HCI \u00d7 Forestry",
    "log_GFCF_x_forestry_rents": "GFCF \u00d7 Forestry",
    "ECI_lag1": "Lagged ECI",
}


# ── Correlation chart category map ──────────────────────────────────────────
# Each entry: original column → (short label, category). The set of categories
# is closed (5 items). If a column is missing from Master.csv it is silently
# skipped, so the chart only shows what is actually computable.
CAT_MAP = {
    "Total natural resources rents (% of GDP)": ("NR Rents", "Resource Rents"),
    "Oil rents (% of GDP)":                      ("Oil Rents", "Resource Rents"),
    "Mineral rents (% of GDP)":                  ("Mineral Rents", "Resource Rents"),
    "Natural gas rents (% of GDP)":              ("Gas Rents", "Resource Rents"),
    "GDP per capita (constant prices, PPP)":     ("GDP per Capita", "Macro & Structure"),
    "Manufacturing":                             ("Manufacturing", "Macro & Structure"),
    "Agriculture":                               ("Agriculture", "Macro & Structure"),
    "Services":                                  ("Services", "Macro & Structure"),
    "Industry":                                  ("Industry", "Macro & Structure"),
    "Trade (% of GDP)":                          ("Trade", "Macro & Structure"),
    "Urban population (% of total population)":  ("Urban Population", "Macro & Structure"),
    "Domestic credit to private sector (% of GDP)": ("Domestic Credit", "Finance & Investment"),
    "Adjusted savings: gross savings (% of GNI)":("Savings", "Finance & Investment"),
    "Gross fixed capital formation, all, Constant prices, Percent of GDP":
                                                 ("Capital Formation", "Finance & Investment"),
    "Share of investment in GDP":                ("Investment Share", "Finance & Investment"),
    "Real interest rate (%)":                    ("Interest Rate", "Finance & Investment"),
    "Lending interest rate (%)":                 ("Lending Rate", "Finance & Investment"),
    "Inflation, consumer prices (annual %)":     ("Inflation", "Finance & Investment"),
    "Capital depreciation rate":                 ("Depreciation", "Finance & Investment"),
    "Human capital index":                       ("Human Capital", "Human Capital & Infra"),
    "Life expectancy at birth, total (years)":   ("Life Expectancy", "Human Capital & Infra"),
    "Access to electricity (% of population)":   ("Electricity Access", "Human Capital & Infra"),
    "Mobile cellular subscriptions (per 100 people)": ("Mobile Subs", "Human Capital & Infra"),
    "Death rates, crude per 1000 people":        ("Death Rates", "Human Capital & Infra"),
    "Rule of law index":                         ("Rule of Law", "Governance"),
    "Political stability \u2014 estimate":       ("Political Stability", "Governance"),
    "Property rights":                           ("Property Rights", "Governance"),
    "Political corruption index":                ("Pol. Corruption", "Governance"),
    "Government revenue":                        ("Gov Revenue", "Governance"),
    "Primary net lending, General government, Percent of GDP":
                                                 ("Primary Lending", "Governance"),
    "Landlocked":                                ("Landlocked", "Macro & Structure"),
}


# ── Colours (kept here so the viz notebooks import them too) ────────────────
CLUSTER_COLORS = {
    "Petrostates":        "#E63946",
    "Oil Exporters":      "#457B9D",
    "Major Producers":    "#2A9D8F",
    "Mining Exporters":   "#E9C46A",
    "Forestry Intensive": "#8B5CF6",
}

CAT_COLORS = {
    "Resource Rents":         "#E74C3C",
    "Macro & Structure":      "#8B5CF6",
    "Finance & Investment":   "#E67E22",
    "Human Capital & Infra":  "#1ABC9C",
    "Governance":             "#3498DB",
}

SAMPLE_COLORS = {
    "Main sample": "#4a6fa5",
    "Adj \u22653%": "#c23a3a",
    "Adj \u22652%": "#2e7d4a",
    "Adj \u22651%": "#c9a227",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def artifact_path(name):
    """Return the full path to an artifact file."""
    return os.path.join(ARTIFACTS, name)


def load_master():
    """Load Master.csv from INTERMEDIARY."""
    import pandas as pd
    return pd.read_csv(os.path.join(INTERMEDIARY, "Master.csv"))


def load_clusters_1995():
    """Load the k=5 cluster assignments based on the 1995 snapshot."""
    import pandas as pd
    return pd.read_csv(os.path.join(INTERMEDIARY, "clusters1995.csv"))


def load_natural_resource():
    """Load the long-format NaturalResource.csv (used by NB6 portfolio chart)."""
    import pandas as pd
    return pd.read_csv(os.path.join(INTERMEDIARY, "NaturalResource.csv"))
