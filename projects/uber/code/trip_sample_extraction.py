"""
trip_sample_extraction.py

Stratified random sampling of NYC TLC FHVHV trips across the full panel,
both Uber and Lyft, with reweighting to preserve unbiased population
estimates.

Why a sample, not aggregates: aggregates throw away individual trip
variation permanently. A well-designed sample lets us answer any new
question without re-downloading 36 GB of parquets from OneDrive.

Sampling design:
    - Per file: 60k Uber + 40k Lyft = 100k rows
    - Strata: zone_class (5) x week_of_month (5) x hour_bucket (5)
    - Allocation: square-root proportional (Neyman-lite)
        --> rare strata get more sample than pure proportional gives,
            so analyses of e.g. 3am-Bronx are still possible
    - Each sampled row has a sampling weight = pop_stratum / sample_stratum
        --> downstream analyses multiply by `sampling_weight` for unbiased
            population estimates

Outputs:
    outputs/tables/_chunks_sample/<fname>_sample.parquet   per-file shards
    outputs/tables/trip_sample_full.parquet                concatenated
    outputs/tables/trip_sample_diagnostics.csv             diagnostics

Expected runtime: ~2 hours against the full panel. Resumable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from trip_queries import (
    connect, iter_fhvhv_chunked, zone_class_cte, log,
)

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "outputs"
TABLE_DIR = OUT_DIR / "tables"
CHUNK_DIR = TABLE_DIR / "_chunks_sample"

UBER = "HV0003"
LYFT = "HV0005"
SAMPLE_SIZES = {UBER: 60_000, LYFT: 40_000}


# Metrics for which we capture full-population distribution summaries
# (one row per file x operator, with percentiles computed in DuckDB)
DIST_METRICS = [
    ("margin_proxy",        "(base_passenger_fare - driver_pay) / base_passenger_fare"),
    ("pay_per_mile",        "driver_pay / trip_miles"),
    ("pay_per_minute",      "driver_pay / (trip_time / 60.0)"),
    ("pay_per_hour",        "driver_pay * 3600.0 / trip_time"),
    ("fare_per_mile",       "base_passenger_fare / trip_miles"),
    ("fare_per_minute",     "base_passenger_fare / (trip_time / 60.0)"),
    ("trip_miles",          "trip_miles"),
    ("trip_time_min",       "trip_time / 60.0"),
    ("speed_mph",           "trip_miles / (trip_time / 3600.0)"),
    ("base_passenger_fare", "base_passenger_fare"),
    ("driver_pay",          "driver_pay"),
    ("tips",                "tips"),
    ("total_rider_payment",
     "base_passenger_fare + tolls + bcf + sales_tax + congestion_surcharge + airport_fee + tips"),
    ("response_sec",
     "EXTRACT(EPOCH FROM on_scene_datetime - request_datetime)"),
]
PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def q_full_distribution(view: str, license_code: str) -> str:
    """Compute population-level distribution summaries for every metric.

    One row returned per (file, operator). Captures percentiles, mean, sd,
    plus fraction-negative and fraction-zero flags where meaningful.
    Computed over ALL trips in the file (not the sample), so these
    percentiles are authoritative even though the sample loses tail detail.
    """
    select_parts = ["COUNT(*) AS n_trips"]
    for name, expr in DIST_METRICS:
        select_parts.append(f"AVG({expr}) AS {name}__mean")
        select_parts.append(f"STDDEV_SAMP({expr}) AS {name}__sd")
        select_parts.append(f"MIN({expr}) AS {name}__min")
        select_parts.append(f"MAX({expr}) AS {name}__max")
        for p in PERCENTILES:
            tag = f"p{int(p*100):02d}"
            # approx_quantile uses t-digest: ~10x faster than quantile_cont
            # with <1% error. Across 14 metrics x 9 percentiles this saves
            # several minutes per file. For the published numbers, the
            # margin script's exact percentiles are the source of truth;
            # these are for richer per-file context.
            select_parts.append(f"approx_quantile({expr}, {p}) AS {name}__{tag}")
        # Fraction below zero (e.g. negative margin = subsidised trip)
        select_parts.append(
            f"AVG(CASE WHEN ({expr}) < 0 THEN 1.0 ELSE 0.0 END) AS {name}__frac_neg"
        )
    select_sql = ",\n            ".join(select_parts)
    return f"""
    SELECT
        '{license_code}' AS hvfhs_license_num,
        {select_sql}
    FROM {view}
    WHERE hvfhs_license_num = '{license_code}'
      AND base_passenger_fare > 1
      AND base_passenger_fare < 500
      AND driver_pay > 0.5
      AND trip_miles BETWEEN 0.1 AND 100
      AND trip_time BETWEEN 30 AND 18000
    """


def q_trips_with_strata(view: str, license: str) -> str:
    """Fetch all valid trips for one operator, with stratum keys and a
    random number per row. Filtering keeps obvious data errors out.

    NOTE: this is the OLD version that returns the full population. It is
    kept for reference only; the main pipeline now uses q_stratified_sample
    which does the sampling inside DuckDB and returns ~target_n rows.
    """
    return f"""
    SELECT
        hvfhs_license_num,
        dispatching_base_num,
        originating_base_num,
        request_datetime,
        on_scene_datetime,
        pickup_datetime,
        dropoff_datetime,
        PULocationID,
        DOLocationID,
        trip_miles,
        trip_time,
        base_passenger_fare,
        tolls,
        bcf,
        sales_tax,
        congestion_surcharge,
        airport_fee,
        tips,
        driver_pay,
        shared_request_flag,
        shared_match_flag,
        access_a_ride_flag,
        wav_request_flag,
        wav_match_flag,
        {zone_class_cte()} AS zone_class,
        ((date_part('day', pickup_datetime)::INT - 1) / 7 + 1)::INT AS week_of_month,
        CASE
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 0 AND 5  THEN 0
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 6 AND 9  THEN 1
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 10 AND 15 THEN 2
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 16 AND 19 THEN 3
            ELSE                                                          4
        END AS hour_bucket,
        CASE WHEN EXTRACT(DOW FROM pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        random() AS u
    FROM {view}
    WHERE hvfhs_license_num = '{license}'
      AND base_passenger_fare > 1
      AND base_passenger_fare < 500
      AND driver_pay > 0.5
      AND trip_miles BETWEEN 0.1 AND 100
      AND trip_time BETWEEN 30 AND 18000
    """


def q_stratum_counts(view: str, license_code: str) -> str:
    """Pass 1: cheap aggregate. Only counts per stratum, ~125 rows out.
    Takes a few seconds. No window function. No huge data transfer."""
    return f"""
    SELECT
        {zone_class_cte()} AS zone_class,
        ((date_part('day', pickup_datetime)::INT - 1) / 7 + 1)::INT AS week_of_month,
        CASE
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 0 AND 5  THEN 0
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 6 AND 9  THEN 1
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 10 AND 15 THEN 2
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 16 AND 19 THEN 3
            ELSE                                                          4
        END AS hour_bucket,
        COUNT(*) AS N_h
    FROM {view}
    WHERE hvfhs_license_num = '{license_code}'
      AND base_passenger_fare > 1
      AND base_passenger_fare < 500
      AND driver_pay > 0.5
      AND trip_miles BETWEEN 0.1 AND 100
      AND trip_time BETWEEN 30 AND 18000
    GROUP BY 1, 2, 3
    """


def q_oversample(view: str, license_code: str, sample_pct: float) -> str:
    """Pass 2: random oversample. Uses DuckDB's USING SAMPLE which is
    O(N) and avoids any window function. Returns roughly sample_pct% of
    rows, distributed across all strata. Python does the final
    stratified selection on this smaller result."""
    return f"""
    SELECT
        hvfhs_license_num,
        dispatching_base_num,
        originating_base_num,
        request_datetime,
        on_scene_datetime,
        pickup_datetime,
        dropoff_datetime,
        PULocationID,
        DOLocationID,
        trip_miles,
        trip_time,
        base_passenger_fare,
        tolls,
        bcf,
        sales_tax,
        congestion_surcharge,
        airport_fee,
        tips,
        driver_pay,
        shared_request_flag,
        shared_match_flag,
        access_a_ride_flag,
        wav_request_flag,
        wav_match_flag,
        {zone_class_cte()} AS zone_class,
        ((date_part('day', pickup_datetime)::INT - 1) / 7 + 1)::INT AS week_of_month,
        CASE
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 0 AND 5  THEN 0
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 6 AND 9  THEN 1
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 10 AND 15 THEN 2
            WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 16 AND 19 THEN 3
            ELSE                                                          4
        END AS hour_bucket,
        CASE WHEN EXTRACT(DOW FROM pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
    FROM {view}
    WHERE hvfhs_license_num = '{license_code}'
      AND base_passenger_fare > 1
      AND base_passenger_fare < 500
      AND driver_pay > 0.5
      AND trip_miles BETWEEN 0.1 AND 100
      AND trip_time BETWEEN 30 AND 18000
    USING SAMPLE {sample_pct}%
    """


def stratified_sample_fast(con, view: str, license_code: str,
                           target_n: int) -> pd.DataFrame:
    """Two-pass stratified sample, avoiding window function over full data.

    Step 1: tiny COUNT query gets population per stratum.
    Step 2: compute square-root proportional allocation in Python.
    Step 3: ask DuckDB for a random sample of ~3x the rows we need.
    Step 4: pandas takes the first n_h rows per stratum from the oversample.

    The oversample size is chosen so each stratum gets at least its needed
    quota with high probability. For total target 60k from N=13M trips,
    we need sample_pct ~ (target * oversample_factor / N) * 100.
    """
    # Pass 1: stratum populations
    pops = con.execute(q_stratum_counts(view, license_code)).df()
    if len(pops) == 0:
        return pd.DataFrame()
    N_total = pops["N_h"].sum()

    # Allocate target_n across strata using sqrt proportional
    sqrt_N = np.sqrt(pops["N_h"].astype(float))
    raw_alloc = sqrt_N / sqrt_N.sum() * target_n
    pops["n_h"] = np.maximum(1, np.round(raw_alloc).astype(int))
    pops["n_h"] = np.minimum(pops["n_h"], pops["N_h"])

    # Adjust to hit target exactly (sometimes rounding overshoots)
    drift = target_n - pops["n_h"].sum()
    if drift != 0:
        # Distribute drift to largest strata
        idx = pops["n_h"].sort_values(ascending=False).index
        for k in range(abs(drift)):
            i = idx[k % len(idx)]
            if drift > 0 and pops.loc[i, "n_h"] < pops.loc[i, "N_h"]:
                pops.loc[i, "n_h"] += 1
            elif drift < 0 and pops.loc[i, "n_h"] > 1:
                pops.loc[i, "n_h"] -= 1

    # Sampling weights
    pops["sampling_weight"] = pops["N_h"].astype(float) / pops["n_h"]

    # Pass 2: oversample. We want enough rows that each stratum has
    # n_h available. Use 4x safety margin; clamp at 100%.
    sample_pct = min(100.0, (target_n * 4.0) / N_total * 100.0)
    log_msg = (f"      [sample] pop={N_total:,}, "
               f"target={target_n:,}, oversample={sample_pct:.2f}%, "
               f"{len(pops)} strata")
    log(log_msg)
    oversample = con.execute(q_oversample(view, license_code, sample_pct)).df()

    # Pass 3: pandas stratified selection on the much smaller oversample
    keys = ["zone_class", "week_of_month", "hour_bucket"]
    oversample = oversample.merge(pops[keys + ["n_h", "N_h", "sampling_weight"]],
                                  on=keys, how="inner")
    # Within each stratum, take the first n_h rows. The oversample is
    # already in random order from DuckDB's USING SAMPLE, so head(n_h)
    # is a valid random subset of the stratum.
    # Note: this is the cleanest pandas idiom that avoids the deprecation
    # warning and works on all pandas versions.
    parts = []
    for stratum_key, group in oversample.groupby(keys, sort=False):
        n_take = group["n_h"].iloc[0]
        parts.append(group.head(n_take))
    sample = pd.concat(parts, ignore_index=True) if parts else oversample.iloc[0:0]
    sample = sample.drop(columns=["n_h", "N_h"], errors="ignore")

    return sample


def q_stratified_sample(view: str, license_code: str, target_n: int) -> str:
    """DEPRECATED: kept for reference. The window function approach is too
    slow on large files. Use stratified_sample_fast() instead.

    Pipeline (in one SQL query):
      1. base: filtered trips with stratum keys and a random number per row
      2. strata: count and sqrt(count) per (zone_class, week_of_month, hour_bucket)
      3. allocation: square-root proportional allocation summing to target_n,
         capped at population (cant sample more than exists in a stratum)
      4. ranked: row_number per stratum sorted by random `u`
      5. final: keep rows with rank <= allocated n_h; emit sampling_weight = N_h / n_h

    The square-root allocation is Neyman-lite: gives rare strata more sample
    than pure proportional. Weights restore unbiased population estimates.
    """
    return f"""
    WITH base AS (
        SELECT
            hvfhs_license_num,
            dispatching_base_num,
            originating_base_num,
            request_datetime,
            on_scene_datetime,
            pickup_datetime,
            dropoff_datetime,
            PULocationID,
            DOLocationID,
            trip_miles,
            trip_time,
            base_passenger_fare,
            tolls,
            bcf,
            sales_tax,
            congestion_surcharge,
            airport_fee,
            tips,
            driver_pay,
            shared_request_flag,
            shared_match_flag,
            access_a_ride_flag,
            wav_request_flag,
            wav_match_flag,
            {zone_class_cte()} AS zone_class,
            ((date_part('day', pickup_datetime)::INT - 1) / 7 + 1)::INT AS week_of_month,
            CASE
                WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 0 AND 5  THEN 0
                WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 6 AND 9  THEN 1
                WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 10 AND 15 THEN 2
                WHEN EXTRACT(HOUR FROM pickup_datetime) BETWEEN 16 AND 19 THEN 3
                ELSE                                                          4
            END AS hour_bucket,
            CASE WHEN EXTRACT(DOW FROM pickup_datetime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            random() AS u
        FROM {view}
        WHERE hvfhs_license_num = '{license_code}'
          AND base_passenger_fare > 1
          AND base_passenger_fare < 500
          AND driver_pay > 0.5
          AND trip_miles BETWEEN 0.1 AND 100
          AND trip_time BETWEEN 30 AND 18000
    ),
    strata AS (
        SELECT
            zone_class, week_of_month, hour_bucket,
            COUNT(*) AS N_h,
            SQRT(COUNT(*)::DOUBLE) AS sqrt_N_h
        FROM base
        GROUP BY zone_class, week_of_month, hour_bucket
    ),
    allocation AS (
        SELECT
            zone_class, week_of_month, hour_bucket,
            N_h,
            LEAST(
                N_h,
                GREATEST(
                    1,
                    CAST(ROUND(sqrt_N_h * {target_n} / SUM(sqrt_N_h) OVER ()) AS BIGINT)
                )
            ) AS n_h
        FROM strata
    ),
    ranked AS (
        SELECT
            b.*,
            ROW_NUMBER() OVER (
                PARTITION BY b.zone_class, b.week_of_month, b.hour_bucket
                ORDER BY b.u
            ) AS rnk
        FROM base b
    )
    SELECT
        r.* EXCLUDE (u, rnk),
        a.N_h,
        a.n_h,
        a.N_h::DOUBLE / a.n_h AS sampling_weight
    FROM ranked r
    JOIN allocation a USING (zone_class, week_of_month, hour_bucket)
    WHERE r.rnk <= a.n_h
    """


def stratified_sample(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Square-root proportional stratified sample with weights.

    For each stratum (zone_class, week_of_month, hour_bucket):
        - Population N_h is the count in df
        - Allocation n_h is proportional to sqrt(N_h), summed to target_n
        - Sample is the top n_h rows by random `u` within stratum
        - Sampling weight w = N_h / n_h
    """
    df = df.copy()
    keys = ["zone_class", "week_of_month", "hour_bucket"]
    pop_counts = df.groupby(keys).size()
    sqrt_counts = np.sqrt(pop_counts.astype(float))
    raw_alloc = sqrt_counts / sqrt_counts.sum() * target_n
    alloc = raw_alloc.round().astype(int).clip(lower=1)

    # Adjust to hit target_n exactly
    drift = target_n - alloc.sum()
    if drift != 0:
        idx_sorted = alloc.sort_values(ascending=False).index
        for k in range(abs(drift)):
            alloc.loc[idx_sorted[k % len(idx_sorted)]] += 1 if drift > 0 else -1
        alloc = alloc.clip(lower=1)

    # Cap allocation at population
    alloc_df = pd.concat([alloc.rename("n_h"), pop_counts.rename("N_h")], axis=1)
    alloc_df["n_h"] = alloc_df[["n_h", "N_h"]].min(axis=1)

    df = df.merge(alloc_df.reset_index(), on=keys)
    df["_rank"] = df.groupby(keys)["u"].rank(method="first")
    sample = df[df["_rank"] <= df["n_h"]].copy()
    sample["sampling_weight"] = sample["N_h"] / sample["n_h"]
    sample = sample.drop(columns=["u", "_rank", "n_h", "N_h"])
    return sample


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute commonly-needed derived metrics on sampled rows.

    Step 0 (critical): coerce source columns to clean types BEFORE doing
    any string or arithmetic operations. DuckDB returns columns with
    inferred types: a VARCHAR column that is all-NULL in a given file
    will come back as Int32 with all <NA> values (because DuckDB sees
    no string evidence). That breaks downstream `.fillna("")` calls on
    flag columns. Force flag columns to object/string and numeric-like
    columns to float64.
    """
    df = df.copy()

    # Flag columns: must be string-like (or all-empty if missing).
    # If DuckDB gave them as Int (all-NA), convert to empty-string column.
    flag_cols = ["shared_request_flag", "shared_match_flag",
                 "wav_request_flag", "wav_match_flag",
                 "access_a_ride_flag"]
    for col in flag_cols:
        if col not in df.columns:
            df[col] = ""
            continue
        s = df[col]
        # If dtype is numeric (Int*, float*) with all-NA, replace with empty
        if pd.api.types.is_numeric_dtype(s) or str(s.dtype).startswith("Int"):
            df[col] = ""
        else:
            # object or string dtype: normalize NA -> ""
            df[col] = s.where(s.notna(), "").astype(str)

    # Numeric columns we do math on: force float64 so arithmetic is safe
    # even when DuckDB returned an Int dtype (e.g. airport_fee Int32-all-NA).
    numeric_cols = ["airport_fee", "tolls", "bcf", "sales_tax",
                    "congestion_surcharge", "tips",
                    "base_passenger_fare", "driver_pay",
                    "trip_miles", "trip_time"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df["trip_time_min"] = df["trip_time"] / 60.0
    df["trip_time_hr"]  = df["trip_time"] / 3600.0

    df["margin_proxy"]    = (df["base_passenger_fare"] - df["driver_pay"]) \
                            / df["base_passenger_fare"]
    df["pay_per_mile"]    = df["driver_pay"] / df["trip_miles"]
    df["pay_per_minute"]  = df["driver_pay"] / df["trip_time_min"]
    df["pay_per_hour"]    = df["driver_pay"] / df["trip_time_hr"]
    df["fare_per_mile"]   = df["base_passenger_fare"] / df["trip_miles"]
    df["fare_per_minute"] = df["base_passenger_fare"] / df["trip_time_min"]
    df["speed_mph"]       = df["trip_miles"] / df["trip_time_hr"]

    fare_components = ["base_passenger_fare", "tolls", "bcf", "sales_tax",
                       "congestion_surcharge", "airport_fee", "tips"]
    df["total_rider_payment"] = df[fare_components].sum(axis=1, skipna=True)

    df["response_sec"] = (df["on_scene_datetime"] - df["request_datetime"]) \
                            .dt.total_seconds()

    # Flag columns: handle NaN/missing safely. Early-2019 files don't
    # populate some of these (e.g. wav_match_flag). Using fillna("") and
    # then .eq("Y") keeps the dtype clean as int8 without NA propagation.
    for src, dst in [
        ("shared_request_flag", "is_shared_requested"),
        ("shared_match_flag",   "is_shared_matched"),
        ("wav_request_flag",    "is_wav_requested"),
        ("wav_match_flag",      "is_wav_matched"),
        ("access_a_ride_flag",  "is_access_a_ride"),
    ]:
        if src in df.columns:
            df[dst] = df[src].fillna("").eq("Y").astype("int8")
        else:
            df[dst] = 0
    df["operator"]            = df["hvfhs_license_num"].map({UBER: "Uber", LYFT: "Lyft"})

    # ---- Defensive dtype normalization before parquet write ----
    # DuckDB returns some columns as pandas nullable Int (Int32/Int64 with
    # capital I), which PyArrow's parquet writer trips over when there's
    # any value coercion. Convert to plain int64 / object string. Cheap.
    for col in df.columns:
        s = df[col]
        dt = str(s.dtype)
        if dt.startswith("Int") or dt.startswith("UInt"):
            # Nullable Int -> fill NA with 0, cast to int64. Safe because
            # NA in the flag columns means "not recorded" which we treat
            # as the same as False/0 elsewhere.
            df[col] = s.fillna(0).astype("int64")
        elif dt == "object":
            # Object columns: empty string or actual NA -> empty string.
            # This avoids mixed None / float NaN / "" cases that confuse
            # parquet schema inference.
            df[col] = s.where(s.notna(), "").astype(str)

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 68)
    log("trip_sample_extraction.py starting (stratified sample, resumable)")
    log("=" * 68)
    log(f"Per-file targets:  Uber {SAMPLE_SIZES[UBER]:,}, Lyft {SAMPLE_SIZES[LYFT]:,} rows")
    log(f"Strata:            zone_class x week_of_month x hour_bucket")
    log(f"Allocation:        square-root proportional, with weights stored")
    log(f"Per-file output:   {CHUNK_DIR}")
    log(f"Final output:      {TABLE_DIR / 'trip_sample_full.parquet'}")
    log("")

    limit = int(os.environ.get("TRIP_LIMIT", "0")) or None
    t_start = time.time()
    con = connect(memory_limit="4GB", threads=4)
    log("")

    def is_already_done(fname, _path):
        # File is done only if BOTH the sample shard and the dist shard exist.
        sample_path = CHUNK_DIR / f"{fname}_sample.parquet"
        dist_path   = CHUNK_DIR / f"{fname}_dist.parquet"
        return sample_path.exists() and dist_path.exists()

    files_processed = 0
    interrupted = False

    try:
        for fname, view in iter_fhvhv_chunked(
            con, limit=limit,
            should_process=lambda fn, p: not is_already_done(fn, p),
        ):
            shards = []
            dist_rows = []
            for license_code, target in SAMPLE_SIZES.items():
                op_name = {UBER: "Uber", LYFT: "Lyft"}[license_code]

                # ---- (a) Full-population distribution summary ----
                # Cheap: one row of percentiles + means per operator per file.
                # Same DuckDB query plan as the sample query, so reading the
                # parquet twice does not double the OneDrive cost (file is
                # already local by this point).
                try:
                    dist_df = con.execute(q_full_distribution(view, license_code)).df()
                    if len(dist_df):
                        dist_df.insert(0, "source_file", fname)
                        dist_df.insert(1, "operator", op_name)
                        dist_rows.append(dist_df)
                        log(f"      [{op_name}] dist: captured "
                            f"{len(DIST_METRICS)} metrics x "
                            f"{len(PERCENTILES)} percentiles + mean/sd/range")
                except Exception as e:
                    log(f"      [SKIP {op_name} dist] failed: {e}")

                # ---- (b) Stratified sample (two-pass, fast) ----
                # Pass 1: get stratum counts (small, cheap).
                # Pass 2: cheap USING SAMPLE oversample, then pandas does
                # the final stratified slice. Avoids the slow window-function
                # plan that scanned 13M rows just to keep ~60k.
                try:
                    sample = stratified_sample_fast(con, view, license_code, target)
                    if len(sample) == 0:
                        log(f"      [{op_name}] no rows in this file")
                        continue
                    n_strata = sample.groupby(
                        ["zone_class", "week_of_month", "hour_bucket"]).ngroups
                    log(f"      [{op_name}] sample: {len(sample):,} rows, "
                        f"strata used {n_strata}, "
                        f"weight range [{sample['sampling_weight'].min():.1f}, "
                        f"{sample['sampling_weight'].max():.1f}]")
                    shards.append(sample)
                except Exception as e:
                    log(f"      [SKIP {op_name} sample] failed: {e}")

            # Write both outputs together so partial files don't leave us
            # with a sample shard but no dist shard (or vice versa).
            if dist_rows:
                dist_combined = pd.concat(dist_rows, ignore_index=True)
                dist_path = CHUNK_DIR / f"{fname}_dist.parquet"
                dist_combined.to_parquet(dist_path, index=False, compression="snappy")
                log(f"      wrote {dist_path.name}: "
                    f"{len(dist_combined)} rows, "
                    f"{dist_path.stat().st_size / 1024:.1f} KB")

            if shards:
                try:
                    combined = pd.concat(shards, ignore_index=True)
                    log(f"      [debug] after concat: shape={combined.shape}, "
                        f"object_cols={sum(combined.dtypes == 'object')}, "
                        f"nullable_int={sum(str(t).startswith('Int') for t in combined.dtypes)}")
                    combined = add_derived_columns(combined)
                    out_path = CHUNK_DIR / f"{fname}_sample.parquet"
                    combined.to_parquet(out_path, index=False, compression="snappy")
                    size_mb = out_path.stat().st_size / 1e6
                    log(f"      wrote {out_path.name}: "
                        f"{len(combined):,} rows, {size_mb:.1f} MB")
                    files_processed += 1
                except Exception as e:
                    import traceback
                    log(f"      [SAMPLE WRITE FAILED] {type(e).__name__}: {e}")
                    log(f"      [traceback]")
                    for line in traceback.format_exc().splitlines():
                        log(f"        {line}")
                    # Dump dtypes
                    log(f"      [dtypes of combined ({combined.shape}):]")
                    for col, dt in combined.dtypes.items():
                        sample_vals = combined[col].head(3).tolist()
                        log(f"        {col:40s} {str(dt):15s} sample={sample_vals}")
                    # Try to identify problem column by writing one at a time
                    log(f"      [bisecting columns to find problem...]")
                    problem_cols = []
                    for col in combined.columns:
                        try:
                            combined[[col]].head(100).to_parquet('/tmp/test_col.parquet', index=False)
                        except Exception as ce:
                            problem_cols.append((col, str(combined[col].dtype), str(ce)[:100]))
                    if problem_cols:
                        log(f"      [PROBLEM COLUMNS:]")
                        for col, dt, err in problem_cols:
                            log(f"        {col} ({dt}): {err}")
                    raise  # re-raise to let outer handler kick in

    except KeyboardInterrupt:
        log("")
        log("INTERRUPTED by user. Already-written shards are safe.")
        log("Re-run to resume; cached files will be skipped instantly.")
        interrupted = True
    except Exception as e:
        log("")
        log(f"UNEXPECTED ERROR during file iteration: {e}")
        log("Already-written shards are safe; re-run to resume.")
        interrupted = True

    log("")
    log("=" * 68)
    log(f"File iteration done in {(time.time() - t_start) / 60:.1f} min "
        f"({'INTERRUPTED' if interrupted else 'complete'})")
    log(f"  newly processed this run: {files_processed}")
    n_shards = len(list(CHUNK_DIR.glob('*_sample.parquet')))
    log(f"  total shards on disk:     {n_shards}")
    log("=" * 68)

    if n_shards == 0:
        log("[WARN] No shards on disk. Nothing to concatenate.")
        return

    log("Concatenating all shards (this can take a few minutes) ...")
    t_concat = time.time()
    shards = [pd.read_parquet(p) for p in sorted(CHUNK_DIR.glob("*_sample.parquet"))]
    full = pd.concat(shards, ignore_index=True)
    out_path = TABLE_DIR / "trip_sample_full.parquet"
    full.to_parquet(out_path, index=False, compression="snappy")
    log(f"Wrote {out_path.name}: {len(full):,} rows, "
        f"{out_path.stat().st_size / 1e9:.2f} GB "
        f"(concat took {time.time() - t_concat:.1f}s)")

    # Concatenate distribution shards too (small file, just CSV is fine)
    dist_shards = [pd.read_parquet(p)
                   for p in sorted(CHUNK_DIR.glob("*_dist.parquet"))]
    if dist_shards:
        dist_full = pd.concat(dist_shards, ignore_index=True)
        dist_full.to_csv(TABLE_DIR / "trip_distribution_summaries.csv", index=False)
        dist_full.to_parquet(TABLE_DIR / "trip_distribution_summaries.parquet",
                             index=False, compression="snappy")
        log(f"Wrote trip_distribution_summaries: "
            f"{len(dist_full)} rows x {dist_full.shape[1]} columns")

    diag = compute_diagnostics(full)
    diag.to_csv(TABLE_DIR / "trip_sample_diagnostics.csv", index=False)
    print_summary(full, t_start)


def compute_diagnostics(full: pd.DataFrame) -> pd.DataFrame:
    """Per-month, per-operator coverage and weight stats."""
    full = full.copy()
    full["month"] = full["pickup_datetime"].dt.to_period("M").dt.to_timestamp()

    def ess(w):
        return float((w.sum() ** 2) / (w ** 2).sum())

    diag = full.groupby(["month", "operator"]).agg(
        n_sampled=("sampling_weight", "size"),
        weight_sum=("sampling_weight", "sum"),
        weight_mean=("sampling_weight", "mean"),
        weight_max=("sampling_weight", "max"),
        ess=("sampling_weight", ess),
    ).reset_index()
    return diag


def print_summary(full: pd.DataFrame, t_start: float) -> None:
    print()
    print("=" * 68)
    print("SUMMARY: stratified sample extraction")
    print("=" * 68)
    print(f"Total sampled rows:     {len(full):,}")
    print(f"  Uber:                 {(full['operator'] == 'Uber').sum():,}")
    print(f"  Lyft:                 {(full['operator'] == 'Lyft').sum():,}")
    n_months = full['pickup_datetime'].dt.to_period('M').nunique()
    print(f"Months contributing:    {n_months}")
    print(f"Date range:             {full['pickup_datetime'].min()} -> "
          f"{full['pickup_datetime'].max()}")
    print()
    print("Effective sample size (after weighting):")
    for op in ["Uber", "Lyft"]:
        sub = full[full["operator"] == op]
        if len(sub) == 0:
            continue
        w = sub["sampling_weight"]
        ess = (w.sum() ** 2) / (w ** 2).sum()
        ess_pct = ess / len(sub) * 100
        print(f"  {op}: ESS = {ess:,.0f} ({ess_pct:.1f}% of nominal {len(sub):,})")
    print()
    print("Sample mean vs population-implied mean (weighted) for key metrics:")
    for col in ["margin_proxy", "pay_per_hour", "fare_per_mile", "trip_miles"]:
        if col in full.columns:
            sub = full.dropna(subset=[col])
            uw = sub[col].mean()
            wt = (sub[col] * sub["sampling_weight"]).sum() / sub["sampling_weight"].sum()
            print(f"  {col:>20s}: unweighted={uw:+.3f}  weighted={wt:+.3f}")
    print()
    print("Strata coverage:")
    cov = full.groupby(["operator", "zone_class", "week_of_month", "hour_bucket"]).size()
    print(f"  total non-empty strata:    {len(cov):,}")
    print(f"  with >=10 sampled rows:    {(cov >= 10).mean() * 100:.1f}%")
    print(f"  with >=100 sampled rows:   {(cov >= 100).mean() * 100:.1f}%")
    print(f"  smallest:                  {cov.min()}")
    print(f"  median:                    {int(cov.median())}")
    print(f"  largest:                   {cov.max()}")
    print("=" * 68)
    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} minutes")
    print()
    print("To use this sample in pandas:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('outputs/tables/trip_sample_full.parquet')")
    print(f"  # For unbiased population estimates, weight by 'sampling_weight':")
    print(f"  weighted_mean = (df['margin_proxy'] * df['sampling_weight']).sum() / df['sampling_weight'].sum()")


if __name__ == "__main__":
    main()
