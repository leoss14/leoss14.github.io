# extension_2024/

Parallel workspace that extends the capstone panel from 1995-2019 to 1995-2024 and adopts the current Atlas of Economic Complexity vintage as the ECI source. Mirrors the layout of `../threshold_sweep/` so it can be developed and validated without touching the live `New code/` pipeline.

## What this is

A self-contained re-build of the data pull stage of the capstone, with three substantive changes vs the live pipeline:

1. **Year range** extended from `1995-2019` to `1995-2024`.
2. **ECI source** is `eci_hs92` from the local Atlas growth-projection file `intermediary/rawdata/growth_proj_eci_rankings-1.csv`, rather than the older vintage pulled from a teammate's GitHub in the v2 notebooks. This locks in current Atlas methodology and matches the file Leonardo downloaded directly from atlas.cid.harvard.edu.
3. **Atlas growth_proj** retained as a separate column (was previously dropped at the rename step) so it can be used as a benchmark in the forecast notebook.

Downstream notebooks (data prep, clustering, ML, regression, forecast) are not in this folder yet. They will be forked from `New code/1_data_prep.ipynb` etc. once the extended panel is built and coverage is known.

## Source of the data-pull notebooks

The two data-pull notebooks here are adapted from `../../code/v2/0_NR_extraction_FINAL.ipynb` and `../../code/v2/1_cleaning_master_data_FINAL.ipynb`. The v2 versions already use `eci_hs92` from a `growth_proj_eci_rankings.csv` file, but they pull it from a teammate's GitHub which holds an older Atlas vintage. The current `intermediary/Master.csv` was built from that older vintage, which is why its ECI values do not match the file Leonardo just downloaded.

## File layout

```
extension_2024/
├── README.md                           this file
├── _config.py                          year range, paths, design decisions
├── e0_NR_extraction.ipynb              adapted v2/NB0, extended to 2024
├── e1_data_pull.ipynb                  adapted v2/NB1, extended to 2024 with local ECI
├── intermediary/
│   ├── cache/                          per-source cache (set FORCE_REFRESH=True to bypass)
│   ├── rawdata/
│   │   └── growth_proj_eci_rankings-1.csv   uploaded Atlas file, 1995-2024 hs92 + growth_proj
│   ├── master_data_long.csv            output of e1
│   ├── master_data_wide.csv            output of e1, wide form
│   └── Master_extended.csv             headline output, ready for downstream notebooks
└── artifacts/                          downstream output directory (panel.parquet etc.)
```

## Year-range bumps applied (relative to v2/NB1)

| Cell | Variable           | Live pipeline      | This pipeline       | Note                                              |
|------|--------------------|--------------------|---------------------|---------------------------------------------------|
| 6    | World Bank API     | end_year=2019      | end_year=2024       | Returns whatever WDI has; typically 1-2 year lag |
| 10   | IMF WEO            | `download(2024)`   | `download(2025)`    | April 2025 vintage; falls back to 2024 if missing |
| 12   | IMF ICSD           | range(1995, 2020)  | range(1995, 2025)   | CSV from teammate's GitHub may itself be capped   |
| 16   | V-Dem              | unfiltered upper   | unfiltered upper    | unchanged; V-Dem releases ~April each year        |
| 18   | PWT                | <= 2019            | <= 2019 (unchanged) | PWT 11.0 stops at 2019; would need 11.1 for more  |
| 20   | CEPII              | range(1995, 2020)  | range(1995, 2025)   | time-invariant; this is just cross-product width  |
| 22   | NR production      | <= 2019            | depends on e0       | Needs e0 to be re-run with updated source files   |
| 24   | Final filter       | <= 2019            | <= 2024             | Master cutoff                                     |

## Known coverage limits

- **PWT 11.0** is the bottleneck for `Human capital index` (HCI). It hard-stops at 2019. Two options:
  - Accept: HCI stays NaN for 2020-2024 in the extended panel. Regressions with HCI as a predictor would either lose those years or need to interpolate.
  - Upgrade to PWT 11.1: pulls from a different URL, may have schema changes. Worth a one-cell test.
- **IMF ICSD** is loaded from a snapshot CSV in the teammate's GitHub, dated 2025-12-23. Its actual year coverage may or may not extend through 2024 — needs verification at run time.
- **NR production** comes from `e0` (v2/NB0). That notebook pulls from EI, OWID, USGS source files which are versioned by year of release. Whether post-2019 data is available depends on which source files are present in `intermediary/rawdata/`. The notebook has explicit version checks.
- **V-Dem v15** (the most recent annual release) covers up to 2024 in the most recent vintage. Should be fine.
- **WDI indicators**: post-2019 coverage is uneven, especially for `Use of IMF credit (DOD, current US$)` and the resource-rents indicators. The coverage diagnostic cell at the end of e1 reports per-variable non-null counts by year.

## How to use this tomorrow

1. Open `e1_data_pull.ipynb` and run all cells. Expected time: 5-15 minutes (V-Dem download is the slowest at ~100MB).
2. Inspect the coverage diagnostic at the end. Decide:
   - Are post-2019 gaps acceptable as-is, or do you want to upgrade PWT and re-source ICSD?
   - Use balanced or unbalanced panel for regressions?
   - Add a `post_2019` dummy + resource-dependence interaction term to absorb COVID-era structural break?
3. If the panel looks healthy, fork `../1_data_prep.ipynb` into `e2_data_prep.ipynb` and point it at `intermediary/Master_extended.csv`.
4. Continue downstream: e3 clusters, e4 ML, e5 regressions, e6 forecast (with `growth_proj` benchmark).

`e0_NR_extraction.ipynb` is only needed if you want to pull fresh NR production data beyond 2019. Otherwise e1 will reuse the existing `nr_production.csv` cache. Skip unless required.

## Decisions still open (will be filled in once data is in)

- Balanced vs unbalanced panel for the post-2019 period
- Whether to include a `post_2019 x resource_dependence` interaction
- Whether to upgrade PWT 11.0 to 11.1 for HCI coverage through 2023
- Whether to bring product-level Atlas data into the case studies (decided later, per Leonardo)

## What this does NOT touch

- Live `New code/` pipeline files
- Live `projects/capstone/page.html` text or numerical claims
- Live `outputs/` files
- `threshold_sweep/` workspace

This workspace is read-only with respect to everything else in the repo until results are validated and a decision is made to migrate.
