#!/usr/bin/env python3
"""
Run the data cleaning pipeline.

Reads CSV files from data/processed/, applies the appropriate cleaner for
each dataset, and writes cleaned outputs to the same directory with a
``_cleaned`` suffix.

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --processed-dir /custom/path
    python scripts/clean_data.py --seasons 2022,2023,2024
    python scripts/clean_data.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Resolve project root (scripts/ → project root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_chatbot.data.cleaning import run_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean processed NFL data CSVs and write *_cleaned.csv outputs."
    )
    parser.add_argument(
        "--processed-dir",
        default=str(ROOT / "data" / "processed"),
        help="Directory containing processed CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--seasons",
        default="",
        help="Comma-separated seasons to keep, e.g. 2022,2023,2024 (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        logging.error("Processed directory not found: %s", processed_dir)
        return 1

    seasons: list[int] | None = None
    if args.seasons.strip():
        try:
            seasons = [int(s.strip()) for s in args.seasons.split(",")]
        except ValueError:
            logging.error("Invalid --seasons value: %r", args.seasons)
            return 1

    logging.info("Cleaning pipeline — source: %s", processed_dir)
    if seasons:
        logging.info("Season filter: %s", seasons)

    reports = run_pipeline(processed_dir, seasons=seasons)

    if not reports:
        logging.warning("No matching CSV files found in %s", processed_dir)
        return 0

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"{'Dataset':<20} {'In':>7} {'Out':>7} {'Dupes':>7} {'Bad teams':>10}")
    print("─" * 60)
    for name, r in sorted(reports.items()):
        print(
            f"{r.table:<20} {r.rows_in:>7} {r.rows_out:>7} "
            f"{r.dupes_dropped:>7} {r.bad_teams_dropped:>10}"
        )
        for w in r.warnings:
            print(f"  {'WARNING:':<12} {w}")
    print("─" * 60)
    print()

    errors = [r for r in reports.values() if r.rows_out == 0 and r.rows_in == 0]
    if errors:
        logging.error("%d dataset(s) failed to clean", len(errors))
        return 1

    logging.info("Cleaning complete — %d dataset(s) processed", len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
