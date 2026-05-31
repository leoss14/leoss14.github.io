# Zombie Firms Replication — Project Documentation

**Project:** Replication and expansion of *Geographical Analysis of the Italian Industrial North: Zombie Firms and Spillover Effects*  
**Original:** Bocconi BSc thesis, advisor Prof. Roberto Perotti, 2023–2024  
**Expansion:** All 20 Italian regions, 2016–2024, three zombie definitions, FE regression with spillover analysis  
**Status:** Lombardia pilot complete; full national panel in progress (10 of 20 regions downloaded as of 06/04/2026)

---

## 1. Directory structure

```
/Users/leoss/Desktop/Thesis Replication/
│
├── Data/
│   ├── CAP_track/
│   │   └── IT.txt                          GeoNames Italy postcode centroids (CC-BY 4.0)
│   │   └── readme.txt                      GeoNames format documentation
│   │
│   ├── Lombardia/                          Orbis batch exports — 8 files, ~3,058 firms each
│   │   └── Export_*.xlsx
│   ├── Veneto/                             In progress
│   ├── Emilia-Romagna/
│   ├── Piemonte/
│   ├── Toscana/
│   ├── FVG/
│   ├── Liguria/
│   ├── Lazio/
│   ├── Marche/
│   └── Campania/
│       (remaining 10 regions to download)
│
└── output/
    ├── zombie_panel_wide.parquet           One row per firm (all years as columns)
    ├── zombie_panel_long.parquet           One row per firm-year (17 years × N firms)
    ├── zombie_panel_classified.parquet     Long panel + zombie indicators + geo columns
    ├── zombie_province_year.csv            Province × year zombie share summary
    ├── nb0_coverage_report.csv             Variable coverage by year
    ├── nb3_regression_results.csv          Regression coefficients table
    ├── nb3_descriptive_stats.csv           Descriptive statistics
    ├── nb3_regression_tables.txt           Formatted results text
    │
    ├── geo/
    │   ├── NUTS_RG_20M_2021_4326.geojson   Eurostat NUTS3 shapefile (downloaded once)
    │   ├── nominatim_cache.json            OSM Nominatim geocoding cache
    │   └── firm_geocoded.csv               One row per firm with lat/lon + method
    │
    └── *.png                               All figures (listed in Section 5)
```

---

## 2. Data source — Orbis (Bureau van Dijk)

**Access:** LSE account via `member@lse.ac.uk`  
**Platform:** https://orbis.bvdinfo.com  
**Data update:** 03/04/2026 (update n° 386002)

### Search criteria (saved as strategy `Leoss-Replication.strategy`)

| Step | Filter | Result |
|------|--------|--------|
| 1 | Country = Italy | 9,224,657 |
| 2 | NACE Rev. 2 primary codes 10–33 (manufacturing) | 612,920 |
| 3 | Employees ≥ 10, any year 2016–2025 | 88,690 |
| 4 | Region filter applied per export batch | varies |

### Saved view: `Leoss-replication`

~300 columns covering:

- **Identification:** Company name, address, postcode, city, country, NUTS2/3, BvD ID, legal form, status, incorporation date, consolidation code, listing status
- **NACE:** Primary + secondary codes and descriptions, BvD sectors
- **P&L (2008–2024):** Operating revenue, added value, employee costs, material costs, EBITDA, EBIT, depreciation, financial expenses, net income
- **Balance sheet (2008–2024):** Total assets (Key Financials version, 91% coverage), tangible fixed assets, shareholders funds, current liabilities, non-current liabilities, loans & short-term debt
- **Employment (2008–2024):** Number of employees
- **Ownership:** GUO name, BvD ID, country; BvD Independence Indicator

### Known structural issues in exports

| Issue | Description | Resolution |
|-------|-------------|------------|
| Loans gap | Years 2011–2014 appear as orphan columns at end of sheet instead of in main block | Handled in NB0: variable map built by year label, not position |
| Duplicate company name column | Two columns named `Company name Latin alphabet` (capitalisation difference) | First kept, second dropped in NB0 `deduplicate_columns()` |
| 2025 columns | Tangible FA 2025 and Total assets 2025 present but mostly empty | Dropped in NB0 (`DROP_2025_COLS = True`) |
| openpyxl stylesheet bug | `CellStyle.__init__() got unexpected keyword argument 'applyNumFmt'` | All exports read via direct XML parsing (bypass openpyxl stylesheet) |
| Date of incorporation | Stored as Excel serial number (days since 1899-12-30), not a date string | Decoded in NB1 with serial→year conversion (values ≥ 2958 treated as serial) |
| Total liabilities | Balance sheet version has 3–4% coverage | Derived as Total assets − Shareholders funds (91% coverage) |
| LF consolidation code | ~4.4% of firms have limited financial data code | Excluded in NB1 sample filter (kept only U1, U2, C1, C2) |

### Export logistics

- Maximum firms per export: **3,058** (constrained by 300-column width × Excel row limit)
- Lombardia: 21,895 firms → 8 exports (7 × 3,058 + 1 × 489)
- Other large regions: Veneto ~4 exports, Emilia-Romagna ~4 exports
- Small regions (FVG, Liguria, Molise, etc.): 1 export each
- Total planned exports: ~40 (tracked in `orbis_export_plan.xlsx`)
- Naming convention: `Export_DD_MM_YYYY_RegionName_N.xlsx`
  - NB0 glob pattern: `Export_*.xlsx` per region subfolder

---

## 3. Pipeline notebooks

All notebooks located in `/Users/leoss/Desktop/Thesis Replication/` (same folder as region subfolders). Run in sequence: NB0 → NB1 → NB2 → NB3.

---

### NB0 — Data Merging and Cleaning (`NB0_data_merge_clean.ipynb`)

**Input:** All `Export_*.xlsx` files in `EXPORT_DIR`  
**Output:** `zombie_panel_wide.parquet`, `zombie_panel_long.parquet`, `nb0_coverage_report.csv`

**Key settings:**
```python
EXPORT_DIR          = Path("/Users/leoss/Desktop/Thesis Replication")
OUTPUT_DIR          = Path("/Users/leoss/Desktop/Thesis Replication/output")
PANEL_YEARS         = list(range(2008, 2025))
DROP_2025_COLS      = True
CONSOLIDATION_FILTER = "ALL"          # filtering done in NB1
EXPORT_PATTERN      = "Export_*.xlsx"
```

**Steps:**
1. Reads exports via XML parsing (bypasses openpyxl bug)
2. Deduplicates columns (removes second company name col)
3. Drops 2025 columns
4. Deduplicates firms by BvD ID (keeps first occurrence across batch files)
5. Builds variable map: `{variable_name: {year: column_index}}`
6. Reshapes wide → long (one row per firm-year)
7. Computes derived variables (see table below)
8. Saves wide and long parquet files

**Derived variables computed in NB0:**

| Variable | Formula | Notes |
|----------|---------|-------|
| `total_liabilities` | Total assets − Shareholders funds | Proxy; direct var has 3–4% coverage |
| `icr` | EBIT / Financial expenses | NaN when financial expenses ≤ 0 |
| `roa` | Net income / Total assets | NaN when total assets = 0 |
| `investment_rate` | Δ(Tangible FA) / lagged Total assets | Requires 2 consecutive years |
| `neg_equity` | 1 if Shareholders funds < 0 | Float, NaN where missing |
| `nace_2digit` | First 2 digits of NACE primary code | Integer |

**Lombardia pilot results:**
- 8 exports → 21,895 rows → 21,247 unique firms after deduplication (648 duplicates at batch boundaries)
- Long panel: 361,199 firm-year observations (21,247 × 17 years)
- Financial coverage 2016–2024: 80–84% (all firms, including those entering/exiting mid-panel)
- Financial coverage in test sample (active firms only): ~91%

---

### NB1 — Zombie Firm Classification (`NB1_zombie_classification.ipynb`)

**Input:** `zombie_panel_long.parquet`  
**Output:** `zombie_panel_classified.parquet`

**Key settings:**
```python
ANALYSIS_YEARS     = list(range(2016, 2025))
MCGOWAN_CONSEC     = 3        # consecutive years ICR < 1
MCGOWAN_ICR_THR    = 1.0
MCGOWAN_MIN_AGE    = 10       # years since incorporation
KEEP_CONSOL        = ['U1', 'U2', 'C1', 'C2']
MIN_EMPLOYEES      = 10
```

**Sample filters (applied in order):**

| Step | Filter | Firms removed |
|------|--------|--------------|
| Year filter | 2016–2024 only | 0 |
| Consolidation | Drop LF (limited financials) | 936 |
| NACE | Verify 10–33 | 0 |
| Employees | Drop firms never observed ≥ 10 | 8 |
| Min financial data | ≥ 1 year with both total assets and EBIT | 0 |
| **Final sample** | | **20,303 firms** |

**Firm age:**
- Parsed from `date_incorporation` (Excel serial number format)
- Conversion: values ≥ 2958 → days since 1899-12-30; values 1800–2030 → plain year
- Coverage: 100% after fix

**Zombie definitions:**

| Label | Definition | Lombardia 2022 share |
|-------|------------|---------------------|
| `zombie_weak` | ICR < 1, current year | 15.4% |
| `zombie_mcgowan` | ICR < 1 for ≥ 3 consecutive years + age ≥ 10 | 3.4% |
| `zombie_storz` | ICR < 1 AND ROA < 0 AND neg. equity, same year | 3.2% |

**McGowan implementation:** consecutive run length computed per firm via `consec_run_length()` function. A missing ICR in any year breaks the run. Age filter applied where observable; firms with missing age pass through.

**Lombardia zombie time series (McGowan):**

| Year | Zombie share | Entry | Exit |
|------|-------------|-------|------|
| 2016–2017 | 0.0% | — | — |
| 2018 | 2.9% | 584 | — |
| 2019 | 3.1% | 216 | 167 |
| 2020 | 3.4% | 245 | 184 |
| 2021 | 3.4% | 277 | 271 |
| 2022 | 3.4% | 275 | 275 |
| 2023 | 2.8% | 151 | 280 |
| 2024 | 3.0% | 235 | 194 |

Note: 2016–2017 zero is mechanical — 3 consecutive years of data not available until 2018 (panel starts 2016).

**Output:** `zombie_panel_classified.parquet` — 182,727 rows, 20,303 firms, 53 columns

---

### NB2 — Geographic Analysis (`NB2_geographic_analysis.ipynb`)

**Input:** `zombie_panel_classified.parquet`  
**Output:** updated `zombie_panel_classified.parquet` (adds `nuts3_code`, `nuts2_code`, `province`), `zombie_province_year.csv`, maps, `geo/firm_geocoded.csv`

**Key settings:**
```python
MAP_YEAR           = 2022
PRIMARY_ZOMBIE     = 'zombie_mcgowan'
MIN_FIRMS_PROVINCE = 5
```

**NUTS3 parsing:**
- `nuts3` column format: `'ITC4C - Milano'`
- Split on ` - ` → `nuts3_code = 'ITC4C'`, `province = 'Milano'`
- NUTS2 code = first 4 characters: `'ITC4'`
- Coverage: 100%

**Geocoding:**
- Primary: GeoNames IT.txt — 4,735 unique CAP codes, loaded from `/Users/leoss/Desktop/Thesis Replication/Data/CAP_track/IT.txt`
- Multiple localities per CAP → mean lat/lon centroid
- Fallback: OSM Nominatim (address+city first, then postcode), cached at `geo/nominatim_cache.json`
- Lombardia result: **100% coverage**, all 20,303 firms geocoded via GeoNames CAP (zero Nominatim calls needed)

**Choropleths:**
- Eurostat NUTS3 shapefile: `NUTS_RG_20M_2021_4326.geojson` (downloaded once, cached in `geo/`)
- URL: `https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326.geojson`

**Province outputs (Lombardia, 2022, McGowan):**

| Province | Firms | Zombie share |
|----------|-------|-------------|
| Como | 1,084 | 4.8% |
| Mantova | 824 | 4.6% |
| Varese | 1,667 | 4.3% |
| Sondrio | 193 | 4.1% |
| Milano | 6,399 | 3.9% |
| Pavia | 610 | 3.6% |
| Cremona | 586 | 3.2% |
| Bergamo | 2,812 | 3.1% |
| Monza e della Brianza | 1,662 | 3.0% |
| Brescia | 3,426 | 2.3% |
| Lecco | 825 | 2.3% |
| Lodi | 215 | 1.9% |

---

### NB3 — Regression Analysis (`NB3_regression.ipynb`)

**Input:** `zombie_panel_classified.parquet`  
**Output:** `nb3_regression_results.csv`, `nb3_regression_tables.txt`, `nb3_descriptive_stats.csv`, `nb3_coefficient_plot.png`

**Key settings:**
```python
REG_YEARS      = list(range(2018, 2025))   # 2018+ for full McGowan history
PRIMARY_ZOMBIE = 'zombie_mcgowan'
WINSOR_P       = 0.01                       # 1% winsorisation each tail
MIN_OBS_CELL   = 5                          # min firms per province-sector cell
```

**Empirical strategy:**

The regression framework follows McGowan, Andrews & Millot (2018) and Caballero et al. (2008). The key variable of interest is zombie congestion: the share of zombie firms in the same province × NACE 2-digit sector × year cell. This is computed as a leave-one-out measure to avoid a firm mechanically contributing to its own congestion term.

**Leave-one-out formula:**
$$\text{ZombieShare}^{-i}_{pst} = \frac{N_{pst} \cdot \bar{Z}_{pst} - Z_{it}}{N_{pst} - 1}$$

**Models:**

| Model | Dependent variable | Sample | Key regressors |
|-------|-------------------|--------|---------------|
| M1 | Investment rate | Non-zombie firms | Zombie share (province-sector) |
| M2 | Employment growth (log diff) | Non-zombie firms | Zombie share (province-sector) |
| M3 | Investment rate | Non-zombie firms | Zombie share × high credit intensity |
| M4 | Investment rate | All firms | Own zombie status (McGowan) |

All models include firm fixed effects and year fixed effects. Standard errors clustered at province × sector level.

**Controls:**
- `log_ta`: log total assets (size)
- `icr_lag_w`: lagged ICR, winsorised (financial health)
- `high_credit`: above-median financial expenses / total assets (credit intensity, M3 only)

**Dependencies:**
```bash
pip install linearmodels    # PanelOLS with clustered SEs
pip install statsmodels     # fallback OLS
```

---

## 4. Figures produced

| File | Notebook | Description |
|------|----------|-------------|
| `zombie_share_over_time.png` | NB1 | Zombie share 2016–2024, all three definitions |
| `zombie_share_by_sector_2022.png` | NB1 | McGowan zombie share by NACE 2-digit sector, 2022 |
| `zombie_share_by_province.png` | NB2 | McGowan zombie share by province over time |
| `map_zombie_italy_2022.png` | NB2 | National choropleth, all NUTS3 (grey = no data) |
| `map_zombie_zoom_2022.png` | NB2 | Choropleth zoomed to regions in panel |
| `map_dotmap_2022.png` | NB2 | Firm-level dot map (red = zombie, blue = non-zombie) |
| `province_scatter.png` | NB2 | Province zombie share vs. median ICR and investment rate |
| `nb3_coefficient_plot.png` | NB3 | Coefficient plot with 95% CI for zombie congestion effect |

---

## 5. Key methodological decisions

**Why `zombie_mcgowan` is the primary definition:**  
The 3-year consecutive requirement distinguishes persistently distressed firms from those experiencing a temporary shock (e.g. COVID 2020, where `zombie_weak` spikes to 26%). The age filter (≥ 10 years) excludes young firms legitimately running deficits during growth phases.

**Why consolidation code `LF` is excluded:**  
LF (Limited Financials) firms have insufficient financial data for zombie classification — they typically lack EBIT and financial expenses. Retaining them would inflate the non-zombie denominator while contributing no zombie classifications, understating the zombie share.

**Why total liabilities is derived rather than sourced directly:**  
The direct total liabilities variable in Orbis (Balance Sheet version) has only 3–4% coverage in this dataset. The identity Total assets − Shareholders funds gives 91% coverage and is accounting-consistent.

**Why the panel starts at 2016 for classification:**  
Coverage for 2008–2015 is 5–14%, too sparse for representative zombie analysis. The regression sample starts at 2018 (first year where 3 consecutive years of ICR data are available from the 2016 panel start).

**Why leave-one-out for the congestion term:**  
A firm classified as zombie in year $t$ mechanically contributes to the province-sector zombie share in year $t$. Using a raw share would conflate the own-zombie effect (Model 4) with the spillover effect (Models 1–3). The LOO correction removes this.

---

## 6. Replication checklist for full national panel

When all 20 regions are downloaded:

- [ ] Move all region subfolders into `/Users/leoss/Desktop/Thesis Replication/`
- [ ] Confirm all files match naming pattern `Export_*.xlsx`
- [ ] Re-run **NB0** — set `EXPORT_DIR` to root folder, will glob all exports automatically
- [ ] Check NB0 output: verify firm count, coverage report, loans gap handling per new region
- [ ] Re-run **NB1** — sample construction table will show updated attrition; check LF share
- [ ] Re-run **NB2** — choropleth will now show full Italy; check NUTS3 parse coverage = 100%
- [ ] Run **NB3** — regression now identified across provinces and sectors nationally
- [ ] Check `zombie_province_year.csv` for regional heterogeneity before interpreting regression

**Expected national panel size:**  
~88,690 firms (Orbis search total) → after filters approximately 70,000–75,000 firms → ~630,000 firm-year observations (2016–2024)

---

## 7. Data citations and licenses

| Source | Use | License |
|--------|-----|---------|
| Orbis (Bureau van Dijk) | Firm-level financial panel | LSE institutional licence |
| GeoNames IT.txt | Postcode → coordinates lookup | CC-BY 4.0 (credit: geonames.org) |
| Eurostat NUTS3 shapefile | Province boundary polygons for maps | CC-BY 4.0 |
| OpenStreetMap Nominatim | Address geocoding fallback | ODbL |

---

## 8. Software and environment

```
Python         3.10 (local: /usr/local/bin/python3.10)
pandas         latest
numpy          latest
geopandas      1.x
linearmodels   latest      (pip install linearmodels)
statsmodels    latest
matplotlib     latest
pyarrow        23.x        (parquet I/O)
```
