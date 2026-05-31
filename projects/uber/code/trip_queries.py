"""
trip_queries.py

Shared infrastructure for trip-level analysis of NYC TLC FHVHV data.

Queries the raw parquets directly via DuckDB without materialising them.
The 40 GB on disk reduces to a few seconds of CPU per simple aggregate,
typically 10 to 30 minutes for queries that scan the full panel.

Conventions:
    - Uber: hvfhs_license_num = 'HV0003'
    - Lyft: hvfhs_license_num = 'HV0005'
    - All times in America/New_York implicit in the source data
    - Zone classification matches _panel_loader.py

Public functions:
    connect(memory_limit, threads) -> duckdb connection
    register_fhvhv(con, data_dir, pattern) -> registers `fhvhv` view
    zone_class_cte() -> SQL fragment classifying PULocationID
    base_filters_sql() -> SQL WHERE clause for valid Uber trips
    save_chart(fig, path) -> writes plotly HTML with shared style
    PLOTLY_TEMPLATE, PALETTE -> styling constants
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional
import duckdb
import plotly.graph_objects as go
import plotly.io as pio

# Force unbuffered output so progress prints show up immediately, even when
# the script is run via a pipe, redirect, or some Mac terminal setups.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def log(msg: str) -> None:
    """Print a timestamped, flushed message. Use everywhere instead of print()."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(
    "/Users/leoss/Library/CloudStorage/OneDrive-LondonSchoolofEconomics/Uber-data"
)
DEFAULT_FHVHV_GLOB = "fhvhv_tripdata_*.parquet"

UBER_LICENSE = "HV0003"
LYFT_LICENSE = "HV0005"

# ---------------------------------------------------------------------------
# Styling: matches _panel_loader.py
# ---------------------------------------------------------------------------

PALETTE = {
    "navy":  "#1f2a44",
    "slate": "#3b4a6b",
    "steel": "#6b7d9e",
    "rose":  "#b85c5c",
    "gold":  "#c9a45c",
    "sage":  "#6b8e6b",
    "grey":  "#9aa3b2",
    "ink":   "#0f1626",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=PALETTE["ink"]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[PALETTE[k] for k in ("navy", "rose", "steel", "gold", "sage", "slate")],
        xaxis=dict(showgrid=True, gridcolor="#e8eaf0", zeroline=False, ticks="outside",
                   tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e8eaf0", zeroline=False, ticks="outside",
                   tickfont=dict(size=11)),
        title=dict(font=dict(size=18, color=PALETTE["ink"]), x=0.02, xanchor="left"),
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#dadde6", borderwidth=1),
    )
)
pio.templates["uber_portfolio"] = go.layout.Template(PLOTLY_TEMPLATE)


def save_chart(fig: go.Figure, path: Path, suppress_title: bool = True) -> None:
    """Write a plotly figure to HTML using the portfolio template.

    The HTML caption box on the project page already shows titles, so we
    blank the in-figure title by default.
    """
    fig.update_layout(template="uber_portfolio")
    if suppress_title:
        fig.update_layout(title_text=None, margin=dict(l=60, r=30, t=30, b=50))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=False)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(memory_limit: str = "8GB", threads: int = 4,
            temp_dir: Optional[str] = None,
            enable_progress_bar: bool = True) -> duckdb.DuckDBPyConnection:
    """Open an in-process DuckDB connection with sensible defaults.

    Adjust memory_limit and threads to your laptop. On 16 GB RAM with
    other apps open, 8 GB is the safe ceiling. Bump threads to 8 if you
    have spare cores. Set temp_dir to an SSD path if memory spills are
    expected on long queries.

    enable_progress_bar (default True) makes DuckDB print a progress
    bar to stderr during queries longer than ~2 seconds. This is the
    single most useful signal that the query is actually doing work.
    """
    log(f"Opening DuckDB connection (memory={memory_limit}, threads={threads})")
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    if temp_dir is not None:
        con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    if enable_progress_bar:
        try:
            con.execute("SET enable_progress_bar=true")
            con.execute("SET progress_bar_time=2000")  # show after 2s
        except Exception as e:
            log(f"Could not enable progress bar: {e}")
    return con


def register_fhvhv(con: duckdb.DuckDBPyConnection,
                   data_dir: Optional[Path] = None,
                   pattern: str = DEFAULT_FHVHV_GLOB,
                   view_name: str = "fhvhv",
                   verify: bool = True) -> int:
    """Register a view over all FHVHV monthly parquets matching the pattern.

    If data_dir is None, falls back to the module-level DEFAULT_DATA_DIR
    or the UBER_DATA_DIR env var. Returns the count of parquet files
    matched. union_by_name handles the schema break that added
    cbd_congestion_fee in late 2024.

    If verify=True (default), runs a tiny COUNT(*) on the view after
    registering it. This forces DuckDB to read the parquet metadata
    upfront so you find out NOW if OneDrive is going to be slow,
    rather than 10 minutes into the analysis.
    """
    import os
    if data_dir is None:
        env = os.environ.get("UBER_DATA_DIR")
        data_dir = Path(env) if env else DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    log(f"Enumerating parquets in {data_dir} ...")
    t0 = time.time()
    files = sorted(data_dir.glob(pattern))
    log(f"  found {len(files)} files in {time.time() - t0:.1f}s")
    if not files:
        raise FileNotFoundError(
            f"No parquets matching {pattern} found in {data_dir}. "
            "Set DEFAULT_DATA_DIR, set UBER_DATA_DIR env var, "
            "or pass data_dir explicitly."
        )
    total_bytes = sum(f.stat().st_size for f in files)
    log(f"  total size on disk: {total_bytes / 1e9:.1f} GB")
    log(f"  first: {files[0].name}")
    log(f"  last:  {files[-1].name}")
    glob_path = str(data_dir / pattern).replace("\\", "/")
    log(f"Creating DuckDB view '{view_name}' (this reads metadata from each "
        f"parquet; takes a while on OneDrive) ...")
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT * FROM read_parquet('{glob_path}', union_by_name=true)
    """)
    log(f"  view created in {time.time() - t0:.1f}s")
    if verify:
        log("Sanity-check: counting rows (forces a full metadata read) ...")
        t0 = time.time()
        n = con.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
        log(f"  {n:,} trips total in {time.time() - t0:.1f}s")
    return len(files)


# ---------------------------------------------------------------------------
# Chunked iteration: process one parquet at a time
# ---------------------------------------------------------------------------
#
# Use case: the raw FHVHV parquets live in a OneDrive Files-On-Demand
# folder and are stored as cloud stubs. Reading all 87 files at once via
# read_parquet glob makes DuckDB request all of them simultaneously, which
# triggers OneDrive to start downloading 36 GB at once and disk fills up.
#
# This iterator pulls one file at a time, registers it as a view called
# `chunk`, yields control to the caller's query, and moves on. OneDrive
# downloads each file as it's accessed and auto-evicts the oldest when
# disk pressure rises. The function does not delete files; eviction is
# OneDrive's job.

def iter_fhvhv_chunked(con: duckdb.DuckDBPyConnection,
                       data_dir: Optional[Path] = None,
                       pattern: str = DEFAULT_FHVHV_GLOB,
                       view_name: str = "chunk",
                       limit: Optional[int] = None,
                       should_process=None):
    """Yield (filename, registered_view_name) for each FHVHV parquet.

    The view `chunk` is recreated on each iteration to point at exactly
    one parquet. The caller's query body runs once per file.

    Args:
        should_process: optional callable taking (fname, file_path) and
            returning True if the file should be opened and yielded,
            False to skip without touching the file. This is critical
            for resumable runs: opening a OneDrive Files-On-Demand stub
            triggers a multi-minute cloud download even if we plan to
            skip it. Pass a function that checks for cached results.
        limit: if set, process only the first N files (for testing).

    Example:
        cache_dir = Path("...")
        def is_cached(fname, path):
            return (cache_dir / f"{fname}_done").exists()
        for fname, view in iter_fhvhv_chunked(con, should_process=lambda f, p: not is_cached(f, p)):
            ...
    """
    import os
    if data_dir is None:
        env = os.environ.get("UBER_DATA_DIR")
        data_dir = Path(env) if env else DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquets matching {pattern} in {data_dir}")
    if limit is not None:
        files = files[:limit]
        log(f"NOTE: limit={limit}, processing only the first {limit} files")
    log(f"Will iterate over {len(files)} parquets, one at a time")
    n_skipped_cheap = 0
    for i, f in enumerate(files, 1):
        # CHEAP skip path: check before we touch the file. Touching a
        # OneDrive stub triggers a download we don't want.
        if should_process is not None and not should_process(f.name, f):
            n_skipped_cheap += 1
            # Print every 10 to keep output readable
            if n_skipped_cheap % 10 == 1 or i == len(files):
                log(f"  [{i}/{len(files)}] {f.name} [SKIP cached] "
                    f"(cumulative cheap skips: {n_skipped_cheap})")
            continue
        t0 = time.time()
        log(f"  [{i}/{len(files)}] {f.name} "
            f"(this may take 10-30s if OneDrive needs to download it)")
        # Register one parquet as the view. DuckDB will trigger the
        # download via the OS file open call.
        path = str(f).replace("\\", "/")
        con.execute(f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{path}')")
        yield f.name, view_name
        log(f"      done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Zone classification: matches _panel_loader.py categories
# ---------------------------------------------------------------------------
#
# CBD: NYC zones south of 60th Street in Manhattan (the MTA congestion zone)
# Buffer: zones immediately north of CBD between 60th and 65th Street
# Airport: JFK, LGA, EWR
# Upper Manhattan: Manhattan zones north of 65th Street
# Outer: everything else (Brooklyn, Queens, Bronx, Staten Island)
#
# Source: NYC TLC taxi zone lookup table cross-referenced with MTA's
# congestion relief zone boundary (5 January 2025 rule).

AIRPORT_ZONES = (132, 138, 1)  # JFK, LGA, EWR

CBD_ZONES = (
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 79, 87, 88, 90,
    100, 103, 104, 105, 107, 113, 114, 125, 137, 140, 141, 142,
    143, 144, 148, 151, 152, 158, 161, 162, 163, 164, 166, 170,
    186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249,
    261, 262, 263,
)

# 60th to 65th Street buffer: Lincoln Square (142), Lenox Hill North (140),
# Yorkville (262 inside CBD already), Central Park (43 inside CBD already).
# The buffer is a narrow strip; we treat Lincoln Square West, Upper West Side,
# Lenox Hill North as the buffer for the arbitrage test.
BUFFER_ZONES = (142, 239, 140, 237)  # Lincoln Sq W, UWS S, Lenox Hill N, UES S

UPPER_MANHATTAN_ZONES = (
    41, 42, 116, 120, 127, 128, 152, 153, 166, 194, 202, 243, 244,
)


def zone_class_cte(pickup_col: str = "PULocationID") -> str:
    """Return a SQL CASE expression mapping a zone column to a zone class.

    Use inside a SELECT to add a `zone_class` column:
        SELECT *, {zone_class_cte()} AS zone_class FROM fhvhv
    """
    airport_list = ", ".join(str(z) for z in AIRPORT_ZONES)
    cbd_list = ", ".join(str(z) for z in CBD_ZONES)
    buffer_list = ", ".join(str(z) for z in BUFFER_ZONES)
    upper_list = ", ".join(str(z) for z in UPPER_MANHATTAN_ZONES)
    return f"""
    CASE
        WHEN {pickup_col} IN ({airport_list}) THEN 'airport'
        WHEN {pickup_col} IN ({cbd_list})     THEN 'cbd'
        WHEN {pickup_col} IN ({buffer_list})  THEN 'buffer'
        WHEN {pickup_col} IN ({upper_list})   THEN 'upper_manhattan'
        ELSE 'outer'
    END
    """


# ---------------------------------------------------------------------------
# Reusable filter clauses
# ---------------------------------------------------------------------------

def base_filters_sql(license: str = UBER_LICENSE,
                     min_fare: float = 1.0,
                     min_pay: float = 0.5,
                     min_miles: float = 0.1,
                     max_miles: float = 100.0,
                     max_fare: float = 500.0) -> str:
    """Return a SQL WHERE clause filtering implausible records.

    Empirical thresholds: TLC data contains a small number of trips with
    zero fare, negative pay, near-zero miles, or absurd values. These
    distort distributional statistics, so we exclude them. The filter
    drops roughly 0.2 to 0.5 percent of trips.
    """
    return f"""
        hvfhs_license_num = '{license}'
        AND base_passenger_fare > {min_fare}
        AND base_passenger_fare < {max_fare}
        AND driver_pay > {min_pay}
        AND trip_miles BETWEEN {min_miles} AND {max_miles}
        AND trip_time > 30
        AND trip_time < 18000
    """


# ---------------------------------------------------------------------------
# Quick sanity check on the connection
# ---------------------------------------------------------------------------

def quick_stats(con: duckdb.DuckDBPyConnection, view_name: str = "fhvhv") -> dict:
    """Run a tiny query to confirm the view works. Returns row count and date range."""
    r = con.execute(f"""
        SELECT COUNT(*) AS n,
               MIN(pickup_datetime) AS min_date,
               MAX(pickup_datetime) AS max_date,
               COUNT(DISTINCT hvfhs_license_num) AS n_operators
        FROM {view_name}
    """).fetchone()
    return dict(n_trips=r[0], min_date=r[1], max_date=r[2], n_operators=r[3])


if __name__ == "__main__":
    # Smoke test: verify the connection and view work
    con = connect()
    n = register_fhvhv(con)
    print(f"Registered {n} parquet files")
    stats = quick_stats(con)
    print(f"Total trips:  {stats['n_trips']:,}")
    print(f"Date range:   {stats['min_date']} -> {stats['max_date']}")
    print(f"Operators:    {stats['n_operators']}")
