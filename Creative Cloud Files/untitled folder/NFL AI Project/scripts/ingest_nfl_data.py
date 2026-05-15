"""
Download and load five seasons of NFL data into SQLite.

Usage:
    python scripts/ingest_nfl_data.py
    python scripts/ingest_nfl_data.py --seasons 2023 2024
    python scripts/ingest_nfl_data.py --datasets teams schedules
    make ingest

Datasets available: teams, schedules, player_stats, rosters
If --datasets is omitted, all four are ingested.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as a standalone script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_chatbot.config import get_config
from nfl_chatbot.data.database import get_engine, init_db, table_row_counts
from nfl_chatbot.data.ingest_nflverse import IngestResult, NflverseIngester

_ALL_DATASETS = ["teams", "schedules", "player_stats", "rosters"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest NFL data from nflverse into SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YYYY",
        help="Seasons to ingest (default: value from config.yaml)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=_ALL_DATASETS,
        default=None,
        metavar="DATASET",
        help=f"Datasets to run (default: all). Choices: {_ALL_DATASETS}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download raw files even if they already exist on disk",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def _print_summary(results: dict[str, IngestResult]) -> None:
    print("\n── Ingestion summary ────────────────────────────────────")
    header = f"  {'Dataset':<20} {'Downloaded':>12} {'Inserted':>10}  Status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, r in results.items():
        status = "OK" if r.ok else f"FAILED — {r.error[:50]}"
        print(f"  {name:<20} {r.rows_downloaded:>12,d} {r.rows_inserted:>10,d}  {status}")
    print()


def _print_db_counts(engine: object) -> None:
    counts = table_row_counts(engine)  # type: ignore[arg-type]
    non_empty = {k: v for k, v in counts.items() if v > 0}
    if not non_empty:
        return
    print("── Database row counts ──────────────────────────────────")
    for table, count in sorted(non_empty.items()):
        print(f"  {count:>10,d}  {table}")
    print()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = get_config()
    seasons: list[int] = args.seasons or cfg.seasons
    datasets: list[str] = args.datasets or _ALL_DATASETS

    print("NFL Data Ingestion")
    print(f"  Seasons  : {seasons}")
    print(f"  Datasets : {datasets}")
    print(f"  Database : {cfg.database_url}")
    print(f"  Raw dir  : {cfg.raw_data_dir}")
    print()

    # Initialise database (no-op if tables already exist)
    engine = get_engine(cfg.database_url)
    init_db(engine)

    # Delete cached raw files if --force
    if args.force:
        for path in cfg.raw_data_dir.glob("*.parquet"):
            path.unlink()
            logging.getLogger(__name__).info("Removed cached file: %s", path.name)
        for path in cfg.raw_data_dir.glob("*.csv"):
            path.unlink()
            logging.getLogger(__name__).info("Removed cached file: %s", path.name)

    ingester = NflverseIngester(
        seasons=seasons,
        raw_dir=cfg.raw_data_dir,
        processed_dir=cfg.processed_data_dir,
        engine=engine,
        rate_limit=cfg.app.scraping.rate_limit_seconds,
    )

    t0 = time.perf_counter()

    # Run only the requested datasets
    results: dict[str, IngestResult] = {}
    for name in datasets:
        method = {
            "teams": ingester.ingest_teams,
            "schedules": ingester.ingest_schedules,
            "player_stats": ingester.ingest_player_stats,
            "rosters": ingester.ingest_rosters,
        }[name]
        results[name] = ingester._run(name, method)  # noqa: SLF001

    elapsed = time.perf_counter() - t0

    _print_summary(results)
    _print_db_counts(engine)

    ok = sum(1 for r in results.values() if r.ok)
    print(f"Completed {ok}/{len(results)} datasets in {elapsed:.1f}s")

    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
