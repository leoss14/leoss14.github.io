# Uber NYC: Code Layout

Analysis code for the Uber NYC FHV market study (Feb 2019 – Apr 2026). Active
scripts at the top level; one-shot migration scripts in `archive/`.

## Active scripts

### Shared infrastructure

| File | Purpose |
|---|---|
| `trip_queries.py` | DuckDB connect, zone classification, palette, `save_chart` helper. Imported by most other scripts. |
| `_panel_loader.py` | Monthly-zone aggregate loader, attaches cluster classification. |
| `taxi_zones.geojson` | NYC TLC zone shapefile (263 zones), used for all maps. |

### Data extraction

| File | Purpose |
|---|---|
| `download_tlc_data.py` | Pulls monthly HVFHV parquet files from the TLC public bucket. |
| `trip_sample_extraction.py` | Builds the 8.34M-trip stratified sample (`outputs/tables/trip_sample_full.parquet`) used in Parts 3 and 4. |
| `run_all_trip_queries.py` | Runner for the trip-query sweep. |

### Tables

| File | Purpose |
|---|---|
| `export_tables.py` | Writes monthly aggregated CSV tables to `outputs/tables/`. |

### Chart generation

| File | Charts produced (output dir) |
|---|---|
| `rebuild_op_clusters_simple.py` | Part 1 cluster map (`sample/op_clusters.html`). |
| `edge_vs_core_test.py` | Appendix edge-vs-core analysis, writes `tables/edge_vs_core_test.csv`. |
| `edge_vs_core_viz_switch.py` | Appendix edge-vs-core scatter chart (`sample/edge_vs_core_scatter.html`). |
| `build_margin_charts.py` | Part 2 margin charts (`margin/trip_margin_fan_monthly.html`, `trip_margin_by_zone_class_p50.html`, `trip_margin_by_length_year.html`). |
| `sample_charts.py` | All Part 3 trip-level charts (`sample/op_*.html`). |
| `analyze_cbd.py` | Part 4 CBD charts (`cbd/cbd_volume_by_zone_class.html`, `cbd_share_inside.html`, `cbd_buffer_share.html`, `cbd_passthrough.html`). |
| `cbd_overlay_charts.py` | Dual-line (pickup-only vs either-end) overlay variant of `cbd_share_inside.html`, written to `cbd/cbd_share_inside_both.html`. Reads from the trip sample so both lines come from the same source. Originals untouched. |
| `cbd_did_analysis.py` | DiD specification on Uber and Lyft base fare around 5 Jan 2025. Prints coefficients to stdout, writes `tables/cbd_did_event_study.csv`. |
| `cbd_did_event_study_viz.py` | Part 4 DiD event-study chart (`cbd/cbd_did_event_study.html`). |
| `cbd_event_study.py` | Canonical event-study regression: log(base_fare) on event-time × treated interactions, zone + month FE, cluster SEs at pickup zone. Reports pre-trends F-test, post-period ATT, and static DiD coefficient. Treatment definition uses either-end CBD. Writes `tables/cbd_event_study_coefs.csv` and `tables/cbd_event_study_tests.csv`. |
| `cbd_event_study_plot.py` | Renders the event-time coefficient plot with 95% CIs from the regression outputs above. Writes `cbd/cbd_event_study_proper.html`. |
| `analyze_drivers.py` | Part 5 driver-side charts (`drivers/drv_pay_trajectory.html`, `drv_margin.html`). |

## archive/

One-shot scripts that were run during the K=6 → K=4 hybrid clustering
migration in May 2026. Kept for reference; not re-run under normal use.

| File | Purpose |
|---|---|
| `cluster_compare_b_c.py` | Compared lat/lon-only vs OD-hybrid vs OD-only clustering options. |
| `cluster_compare_b_c_map.py` | Map version of the above. |
| `rollover_to_C.py` | Backed up the old lat/lon-only clustering and installed the OD-hybrid as canonical. |
| `regen_k4_charts.py` | Regenerated the main-body charts under K=4 hybrid. |
| `regen_k4_appendix.py` | Regenerated the appendix charts under K=4 hybrid. |
| `rev_weighted_and_verify.py` | Switched the operator-margin charts to the revenue-weighted definition (sum of platform-retained dollars over sum of rider-paid dollars). |

If any of these need to be re-run, the import paths already point to the
parent directory's `trip_queries.py` via `Path(__file__).parent.parent`.

## Pre-generated appendix charts (no regen script)

Seven charts in the appendix were generated on 18 May 2026 from an earlier
analysis layer that has since been deleted. They are static artifacts:

- `outputs/market/ext9_bootstrap_gini.html`
- `outputs/market/ext10_mismatch_ratios.html`
- `outputs/spatial/lisa_map_2018.html`
- `outputs/spatial/lisa_map_2025.html`
- `outputs/geographic/lorenz_curve.html`
- `outputs/geographic/pickup_density_change.html`
- `outputs/clusters/fix_borough_pct.html`

The underlying numbers are stable (single-month Jan 2018 vs Jan 2025 snapshots)
and the page text reports the values directly, so regeneration is not normally
needed. If it becomes necessary, restore the relevant analysis script from git
history.

## Setup

```
pip install duckdb pandas plotly pyarrow scikit-learn linearmodels geopandas libpysal esda
```

The scripts expect raw FHVHV parquets in:

```
/Users/leoss/Library/CloudStorage/OneDrive-LondonSchoolofEconomics/Uber-data/
    fhvhv_tripdata_YYYY-MM.parquet
```

Override with `UBER_DATA_DIR` in the shell or edit `DEFAULT_DATA_DIR` in
`trip_queries.py`.

## Outputs

| Dir | Content |
|---|---|
| `outputs/sample/` | Part 3 trip-level charts + cluster map + edge-vs-core scatter |
| `outputs/margin/` | Part 2 margin charts |
| `outputs/cbd/` | Part 4 congestion-fee charts |
| `outputs/drivers/` | Part 5 driver charts |
| `outputs/market/` | Pre-generated bootstrap Gini + mismatch ratios |
| `outputs/spatial/` | Pre-generated LISA maps |
| `outputs/geographic/` | Pre-generated Lorenz curve + pickup density map |
| `outputs/clusters/` | Pre-generated borough mix |
| `outputs/tables/` | Intermediate CSV tables produced during analysis |

## Build order (full rebuild)

1. `download_tlc_data.py` – pull source parquets
2. `trip_sample_extraction.py` – build the 8.34M stratified sample
3. `run_all_trip_queries.py` – produce intermediate tables
4. `export_tables.py` – write the additional aggregate tables
5. `rebuild_op_clusters_simple.py` – build the Part 1 map
6. `build_margin_charts.py` – Part 2
7. `sample_charts.py` – Part 3
8. `edge_vs_core_test.py` then `edge_vs_core_viz_switch.py` – appendix edge-vs-core
9. `analyze_cbd.py` – Part 4 main charts
10. `cbd_overlay_charts.py` – the dual-line variant of `cbd_share_inside`
11. `cbd_did_analysis.py` then `cbd_did_event_study_viz.py` – Part 4 static DiD
12. `cbd_event_study.py` then `cbd_event_study_plot.py` – Part 4 canonical event study
13. `analyze_drivers.py` – Part 5

The pre-generated appendix charts listed above are not regenerated by this flow.
