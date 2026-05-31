"""Replace NR production cell 22 to read e0's local output instead of broken URL.

e0_NR_extraction.ipynb produces intermediary/natural_resources_production_values.csv
with schema [Country, Year, Value, Resource, Metric, iso3]. This patch
rewires the cell to read that file, rename to e1's expected schema, and
apply the same aggregation logic.

Falls back to the (broken) teammate URL only if local file is missing — keeps
the cell parseable for anyone who hasn't run e0.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'

NEW_CELL = '''_prod_cache = CACHE_DIR / 'nr_production.csv'

if not FORCE_REFRESH and _prod_cache.exists():
    production = pd.read_csv(_prod_cache, dtype={'Country Code': str})
    production['Year'] = production['Year'].astype(int)
    print(f"NR production loaded from cache: {production.shape[0]:,} rows, "
          f"{production['Variable'].nunique()} resource-metric combinations")
else:
    # Local e0 output (preferred). Run e0_NR_extraction.ipynb to (re)produce.
    _e0_local = CACHE_DIR.parent / 'natural_resources_production_values.csv'

    if _e0_local.exists():
        print(f"NR production source: local e0 output ({_e0_local.name})")
        production = pd.read_csv(_e0_local)
        # e0 schema: Country, Year, Value, Resource, Metric, iso3
        # Rename to e1 schema: Country Name, Country Code
        production = production.rename(columns={
            'Country': 'Country Name',
            'iso3':    'Country Code',
        })
        # Drop rows where ISO3 mapping failed (regional aggregates, etc.)
        production = production[production['Country Code'].notna()]
        # WB-canonical country names (in case downstream merges expect them)
        production['Country Name'] = production['Country Code'].map(ISO3_TO_WB).fillna(production['Country Name'])
    else:
        # Fallback to teammate URL — this is known broken (404 at the time of
        # writing) but kept for compatibility if a working alternative emerges.
        print("WARNING: e0 output not found locally. Falling back to teammate URL.")
        production = (
            pd.read_csv(
                "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/refs/heads/main/"
                "../rawdata/production_values_w_prices-EM.csv"
            )
            .drop(columns="Unnamed: 0")
            .rename(columns={"Country": "Country Name"})
        )
        production = add_iso3(production, name_col="Country Name", iso3_col="Country Code")
        pre = len(production)
        production = production[production["Country Code"].notna()]
        print(f"Production: dropped {pre - len(production)} rows with unmatched country names")
        production["Country Name"] = production["Country Code"].map(ISO3_TO_WB)

    # Filter: drop consumption rows, clip to year range
    production = production[
        (production["Metric"] != "Consumption")
        & (production["Year"] >= cfg.YEAR_MIN)
        & (production["Year"] <= cfg.YEAR_MAX)
    ]

    # Categorize resources into Hydrocarbons / Metals / Others (matches v2 schema)
    hydrocarbons = ["Oil", "Natural Gas", "Coal"]
    metals = [
        "Lithium", "Cobalt", "Nickel", "Tin", "Bauxite", "Natural Graphite",
        "Copper", "Aluminium", "Zinc", "Manganese", "Rare Earth",
        "Platinum Group", "Vanadium",
    ]

    def classify_resource(x):
        if x in hydrocarbons:
            return "Hydrocarbons"
        elif x in metals:
            return "Metals"
        else:
            return "Others"

    production["Resource Category"] = production["Resource"].apply(classify_resource)
    production["Variable"] = production["Metric"] + "_" + production["Resource Category"]
    production = (
        production
        .groupby(["Country Code", "Year", "Variable"])["Value"]
        .sum()
        .reset_index()
    )

    production.to_csv(_prod_cache, index=False)
    print(f"NR production cached: {production.shape[0]:,} rows, "
          f"{production['Variable'].nunique()} resource-metric combinations, "
          f"{production['Year'].min()}-{production['Year'].max()}")
'''

with open(NB) as f:
    nb = json.load(f)

target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if "_prod_cache" in s and 'production_values_w_prices' in s:
        target = i
        break

if target is None:
    print('ERROR: NR production cell not found.')
    sys.exit(1)

# Idempotency check
current = ''.join(nb['cells'][target]['source'])
if '_e0_local' in current and 'natural_resources_production_values.csv' in current:
    print(f'cell {target}: already patched.')
    sys.exit(0)

nb['cells'][target]['source'] = NEW_CELL.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: replaced.')
