"""Surgical edit of cell 18 in e1_data_pull.ipynb.

Replaces the PWT pull block to:
  1. Remove the hardcoded `<= 2019` filter (PWT 11.0 actually covers
     through 2023 across all eight variables we use).
  2. Update the stale comment that claimed PWT 11.0 stops at 2019.
  3. Prefer a local PWT file over the third-party GitHub mirror.

Preserves cell metadata and outputs structure. Idempotent: re-running on an
already-patched notebook is a no-op.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'
CELL_IDX = 18

NEW_SOURCE = '''_pwt_cache = CACHE_DIR / 'pwt.csv'

if not FORCE_REFRESH and _pwt_cache.exists():
    pwt_df = pd.read_csv(_pwt_cache, dtype={'Country Code': str})
    pwt_df['Year'] = pwt_df['Year'].astype(int)
    print(f"PWT loaded from cache: {pwt_df.shape[0]:,} rows, {pwt_df['Variable'].nunique()} indicators")
else:
    # PWT 11.0 (release March 2025) covers 1950-2023 across every variable
    # listed in pwt_variables. Non-null counts per year through 2023:
    #   hc=145, cn=180, ctfp/cwtfp=120, csh_c/i/g=185, delta=180.
    # The 145/180/120 caps are country-coverage limits, not year limits.
    # Prefer local copy; fall back to the original GitHub mirror.
    local_pwt = "/Users/leoss/Desktop/GitHub/capstone-client-submission/main_analysis/rawdata/pwt110.xlsx"
    if os.path.exists(local_pwt):
        pwt_src = local_pwt
        print(f"PWT: reading local file {local_pwt}")
    else:
        pwt_src = "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/main/rawdata/pwt110.xlsx"
        print(f"PWT: local file missing, falling back to GitHub mirror")

    pwt_df = (
        pd.read_excel(pwt_src, engine="openpyxl", sheet_name="Data")
        .rename(columns={"countrycode": "Country Code", "country": "Country Name", "year": "Year"})
    )
    pwt_df = pwt_df[["Country Code", "Country Name", "Year"] + pwt_variables]
    pwt_df = pwt_df.melt(
        id_vars=["Country Code", "Year"],
        value_vars=pwt_variables,
        var_name="Variable",
        value_name="Value",
    )
    pwt_df = pwt_df[(pwt_df["Year"] >= cfg.YEAR_MIN) & (pwt_df["Year"] <= cfg.YEAR_MAX)]
    pwt_df.to_csv(_pwt_cache, index=False)
    print(f"PWT cached: {pwt_df.shape[0]:,} rows, {pwt_df['Variable'].nunique()} indicators, "
          f"{pwt_df['Year'].min()}-{pwt_df['Year'].max()}")
'''

with open(NB) as f:
    nb = json.load(f)

old_src = ''.join(nb['cells'][CELL_IDX].get('source', []))

if 'pwt_df["Year"] <= 2019' not in old_src and 'pwt_df["Year"] <= cfg.YEAR_MAX' in old_src:
    print('Cell 18 already patched. No-op.')
    sys.exit(0)

# Sanity check we're editing the right cell
if nb['cells'][CELL_IDX]['cell_type'] != 'code' or '_pwt_cache' not in old_src:
    print(f'ERROR: cell {CELL_IDX} is not the PWT cell. Aborting.')
    print(f'  cell_type={nb["cells"][CELL_IDX]["cell_type"]}')
    print(f'  first 100 chars: {old_src[:100]!r}')
    sys.exit(1)

# Apply edit: source is stored as a list of lines, each line ends with \n
# except possibly the last.
new_lines = NEW_SOURCE.splitlines(keepends=True)
nb['cells'][CELL_IDX]['source'] = new_lines

# Clear cell output and execution_count since the code has changed
nb['cells'][CELL_IDX]['outputs'] = []
nb['cells'][CELL_IDX]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched cell {CELL_IDX}. New source is {len(new_lines)} lines.')
print('---')
print(NEW_SOURCE)
