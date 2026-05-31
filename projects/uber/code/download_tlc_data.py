#!/usr/bin/env python3.10
"""
download_tlc_data.py
====================

Fetch every available month of NYC TLC trip-record parquet files from January
2018 through the latest published month. Designed to be re-run safely: any
file already present at the destination (and non-trivially sized) is skipped.

Source: NYC TLC public CloudFront bucket.
  https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{YYYY}-{MM}.parquet

File format split:
  - Through January 2019: fhv_tripdata (Uber via dispatching_base_num).
  - February 2019 onward: fhvhv_tripdata (Uber via hvfhs_license_num = HV0003).
  TLC introduced FHVHV as a separate schema in Feb 2019.

Notes:
  - Files vary 200 MB to 800 MB each. Full coverage is roughly 40-50 GB.
  - TLC publishes with a ~2 month delay, so the latest few months may 404. The
    script treats 404 as "not yet published" and continues to the next file.
  - Uses stdlib only (urllib + os + pathlib). No pip install required.
  - Compatible with Python 3.10+.

Run from anywhere:
  python3.10 /Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber/code/download_tlc_data.py

Optional flags:
  --years 2018 2019 2020   limit to specific years
  --dry-run                show the queue without downloading
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

# ─── Destination ─────────────────────────────────────────────────────────
DATA_DIR = Path('/Users/leoss/Library/CloudStorage/'
                'OneDrive-LondonSchoolofEconomics/Uber-data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLOUDFRONT_BASE = 'https://d37ci6vzurychx.cloudfront.net/trip-data'

# Files below this byte count are treated as partial / corrupt and re-fetched.
MIN_VALID_BYTES = 10_000_000

# Coverage window. Stop at the current month minus the publishing lag; the
# script handles 404s on still-unpublished months gracefully anyway, but
# bounding the queue avoids dozens of pointless requests far into the future.
START_YEAR = 2018
END_YEAR = date.today().year
END_MONTH = date.today().month


def filename_for(year: int, month: int) -> str:
    """Return the TLC filename for a given (year, month) pickup of Uber data.

    The FHVHV schema launched in February 2019. Anything before that lives in
    the older FHV file, where Uber is identified by dispatching_base_num.
    """
    if year < 2019 or (year == 2019 and month == 1):
        return f'fhv_tripdata_{year}-{month:02d}.parquet'
    return f'fhvhv_tripdata_{year}-{month:02d}.parquet'


def all_targets(years: list[int] | None = None) -> list[tuple[int, int, str]]:
    """Enumerate (year, month, filename) tuples for the coverage window."""
    out = []
    for y in range(START_YEAR, END_YEAR + 1):
        if years and y not in years:
            continue
        for m in range(1, 13):
            # Don't queue months from the future.
            if y == END_YEAR and m > END_MONTH:
                break
            out.append((y, m, filename_for(y, m)))
    return out


def fmt_size(b: float) -> str:
    """Human-readable bytes."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def download(filename: str) -> str:
    """
    Returns one of: 'ok', 'skipped', 'not_yet_published', 'failed'.
    """
    url = f"{CLOUDFRONT_BASE}/{filename}"
    out = DATA_DIR / filename

    if out.exists() and out.stat().st_size > MIN_VALID_BYTES:
        print(f"[SKIP] {filename}  ({fmt_size(out.stat().st_size)} already present)")
        return 'skipped'

    print(f"[GET ] {filename}")
    print(f"       {url}")

    last_pct = [-1]
    start = time.time()

    def progress(block_n, block_size, total):
        if total <= 0:
            return
        downloaded = block_n * block_size
        pct = int(100 * downloaded / total)
        # Print at 20% increments to keep the log readable when downloading
        # many months in a row.
        if pct != last_pct[0] and pct % 20 == 0:
            elapsed = time.time() - start
            rate = downloaded / elapsed if elapsed > 0 else 0
            print(f"       {pct:>3d}%  {fmt_size(downloaded)} / {fmt_size(total)}  "
                  f"({fmt_size(rate)}/s)")
            last_pct[0] = pct

    try:
        urllib.request.urlretrieve(url, str(out), reporthook=progress)
        elapsed = time.time() - start
        size = out.stat().st_size
        print(f"       OK  {fmt_size(size)} in {elapsed:.1f}s "
              f"({fmt_size(size / elapsed)}/s)")
        return 'ok'
    except KeyboardInterrupt:
        print(f"       INTERRUPTED. Removing partial file.")
        if out.exists():
            out.unlink()
        raise
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"       Not yet published (HTTP 404). Skipping.")
            return 'not_yet_published'
        print(f"       HTTP error {e.code}: {e}")
        if out.exists():
            out.unlink()
        return 'failed'
    except Exception as e:
        print(f"       FAILED: {e}")
        if out.exists():
            out.unlink()
        return 'failed'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--years', nargs='+', type=int, default=None,
                    help='Restrict to specific years (e.g. --years 2020 2021).')
    ap.add_argument('--dry-run', action='store_true',
                    help='List the queue without downloading.')
    args = ap.parse_args()

    targets = all_targets(args.years)

    print(f"Destination:  {DATA_DIR}")
    print(f"Queue:        {len(targets)} months "
          f"({targets[0][2]} ... {targets[-1][2]})")
    print("-" * 72)

    if args.dry_run:
        for y, m, fn in targets:
            local = DATA_DIR / fn
            tag = "present" if local.exists() and local.stat().st_size > MIN_VALID_BYTES else "queued"
            print(f"  [{tag:>7}] {fn}")
        return 0

    counts = {'ok': 0, 'skipped': 0, 'not_yet_published': 0, 'failed': 0}
    for i, (_y, _m, fn) in enumerate(targets, 1):
        print(f"\n[{i:>3}/{len(targets)}]")
        status = download(fn)
        counts[status] = counts.get(status, 0) + 1

    print()
    print("=" * 72)
    print(f"Downloaded:        {counts['ok']}")
    print(f"Already present:   {counts['skipped']}")
    print(f"Not yet published: {counts['not_yet_published']}")
    print(f"Failed:            {counts['failed']}")
    print("=" * 72)

    # Inventory.
    parquets = sorted(DATA_DIR.glob('*.parquet'))
    total = sum(p.stat().st_size for p in parquets)
    print(f"\n{len(parquets)} parquet files in {DATA_DIR.name}/  ({fmt_size(total)} total)")

    return 0 if counts['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
