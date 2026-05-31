"""
run_all_trip_queries.py

Runs the trip-level analyses in priority order:

  1. trip_margin_distribution    (user-prioritised)
  2. trip_pay_rates              (high value: floor adherence)
  3. trip_cbd_event_study        (high value: identification)
  4. trip_response_percentiles   (medium: equity, tails)
  5. trip_tip_distribution       (medium: decomposition)
  6. trip_surge_intensity        (medium: pricing dispersion)
  7. trip_pool_share             (medium: product trajectory)

Each script is independent. If one fails the others still run. Each
prints its own summary on stdout. A consolidated summary is written
to outputs/tables/trip_run_summary.txt.

Usage:
    python run_all_trip_queries.py           # run all
    python run_all_trip_queries.py 1 2 3     # run scripts 1, 2, 3 only

Expected total runtime: 90 to 180 minutes against the full 40 GB panel.
"""

from __future__ import annotations

import importlib
import io
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE.parent / "outputs"
TABLE_DIR = OUT_DIR / "tables"
SUMMARY_PATH = TABLE_DIR / "trip_run_summary.txt"

SCRIPTS = [
    ("trip_margin_distribution",  "high"),
    ("trip_pay_rates",            "high"),
    ("trip_cbd_event_study",      "high"),
    ("trip_response_percentiles", "medium"),
    ("trip_tip_distribution",     "medium"),
    ("trip_surge_intensity",      "medium"),
    ("trip_pool_share",           "medium"),
]


def main(argv):
    selected = SCRIPTS
    if len(argv) > 1:
        idx = [int(x) - 1 for x in argv[1:]]
        selected = [SCRIPTS[i] for i in idx]

    sys.path.insert(0, str(HERE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    log.write(f"Trip analysis run, started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write("=" * 68 + "\n\n")

    overall_t0 = time.time()
    for idx, (mod_name, priority) in enumerate(selected, 1):
        print(f"\n{'#' * 68}")
        print(f"# [{idx}/{len(selected)}] {mod_name} (priority: {priority})")
        print(f"{'#' * 68}\n")
        log.write(f"\n[{idx}/{len(selected)}] {mod_name} ({priority})\n")
        t0 = time.time()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                mod = importlib.import_module(mod_name)
                mod.main()
            output = buf.getvalue()
            print(output)
            log.write(output)
            log.write(f"\n[OK] {mod_name} finished in {(time.time() - t0) / 60:.1f} min\n")
        except Exception as e:
            print(f"[ERROR] {mod_name}: {e}")
            print(traceback.format_exc())
            log.write(f"[ERROR] {mod_name}: {e}\n")
            log.write(traceback.format_exc())
            log.write("\n")

    total_min = (time.time() - overall_t0) / 60
    log.write(f"\n{'=' * 68}\n")
    log.write(f"Total runtime: {total_min:.1f} minutes\n")
    SUMMARY_PATH.write_text(log.getvalue())
    print(f"\nConsolidated summary written to {SUMMARY_PATH}")
    print(f"Total runtime: {total_min:.1f} minutes")


if __name__ == "__main__":
    main(sys.argv)
