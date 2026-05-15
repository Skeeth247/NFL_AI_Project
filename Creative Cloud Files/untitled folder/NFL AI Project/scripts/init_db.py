"""
Initialize the NFL AI Chatbot database.

Creates all tables in the configured SQLite database (or PostgreSQL if
DATABASE_URL is overridden in .env). Safe to re-run — existing tables and
data are never dropped.

Usage:
    python scripts/init_db.py
    make init-db
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the src package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_chatbot.config import get_config
from nfl_chatbot.data.database import get_engine, init_db, table_row_counts
from nfl_chatbot.data.schema import ALL_TABLES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def run_init_db() -> None:
    cfg = get_config()

    logger.info("Database URL : %s", cfg.database_url)
    logger.info("Seasons      : %s", cfg.seasons)

    engine = get_engine(cfg.database_url)
    created = init_db(engine)

    # Report which expected tables are present
    print("\n── Table status ─────────────────────────────────────")
    for name in ALL_TABLES:
        status = "OK" if name in created else "MISSING"
        print(f"  {status:8s}  {name}")

    counts = table_row_counts(engine)
    print("\n── Row counts ───────────────────────────────────────")
    for name, count in counts.items():
        print(f"  {count:>8,d}  {name}")

    print(f"\nDatabase initialised successfully at:\n  {cfg.db_path}\n")


if __name__ == "__main__":
    run_init_db()
