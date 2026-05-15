"""
Ingest betting odds into SQLite.

Usage:
    # Auto-detect from config (uses CSV path or API key if set)
    python scripts/ingest_odds.py

    # Explicit CSV path
    python scripts/ingest_odds.py --csv data/external/nfl_betting.csv

    # Override seasons
    python scripts/ingest_odds.py --csv path/to/file.csv --seasons 2023 2024

    # Force use of The Odds API
    python scripts/ingest_odds.py --api

    make ingest-odds
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_chatbot.config import get_config
from nfl_chatbot.data.database import get_engine, init_db
from nfl_chatbot.data.ingest_odds import ingest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest NFL betting odds from a CSV file or The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a Kaggle-style or normalised NFL betting CSV",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Force ingestion from The Odds API (requires THE_ODDS_API_KEY in .env)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YYYY",
        help="Seasons to ingest (default: from config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = get_config()
    seasons = args.seasons or cfg.seasons

    # Resolve CSV path: CLI arg → config external dir default
    csv_path: Path | None = args.csv
    if csv_path is None and not args.api:
        candidate = cfg.external_data_dir / "nfl_betting.csv"
        if candidate.exists():
            csv_path = candidate
            logging.getLogger(__name__).info(
                "Found odds CSV at default location: %s", candidate
            )

    # Resolve API key
    api_key: str | None = cfg.the_odds_api_key or None
    if args.api and not api_key:
        print(
            "ERROR: --api flag requires THE_ODDS_API_KEY to be set in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    print("NFL Betting Odds Ingestion")
    print(f"  Seasons  : {seasons}")
    print(f"  CSV path : {csv_path or '(none)'}")
    print(f"  API key  : {'set' if api_key else 'not set'}")
    print(f"  Database : {cfg.database_url}")
    print()

    engine = get_engine(cfg.database_url)
    init_db(engine)

    result = ingest(
        seasons=seasons,
        csv_path=csv_path,
        api_key=api_key if (args.api or csv_path is None) else None,
        engine=engine,
        raw_dir=cfg.raw_data_dir,
        processed_dir=cfg.processed_data_dir,
        rate_limit=cfg.app.scraping.rate_limit_seconds,
    )

    print("\n── Ingestion result ─────────────────────────────────")
    print(f"  Source         : {result.source}")
    print(f"  Rows loaded    : {result.rows:,}")
    print(f"  Seasons found  : {result.seasons_found}")

    if result.warnings:
        print("\n── Warnings ─────────────────────────────────────────")
        for w in result.warnings:
            print(f"  ⚠  {w}")

    if not result.ok:
        print(
            "\n  Betting features will be unavailable until odds data is provided.\n"
            "  The rest of the project will continue to work normally.\n"
        )
        sys.exit(0)  # Non-fatal — project still runs without odds


if __name__ == "__main__":
    main()
