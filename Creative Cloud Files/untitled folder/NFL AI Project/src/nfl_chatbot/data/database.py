"""
Database connection management and utility functions.

Provides a sync SQLAlchemy engine (used by data pipeline scripts and tests)
and helper functions for common read/write patterns.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from sqlalchemy import UniqueConstraint, create_engine, inspect, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from nfl_chatbot.data.schema import Base

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 500  # rows per INSERT batch to stay under SQLite's parameter limit


# ── Engine ─────────────────────────────────────────────────────────────────────


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """
    Return a synchronous SQLAlchemy Engine.

    Strips the async driver prefix so the same DATABASE_URL from .env works
    for both the async API layer and the sync data pipeline.
    Automatically creates the parent directory for file-based SQLite databases.
    """
    if url is None:
        from nfl_chatbot.config import get_config
        url = get_config().database_url

    # Convert async URL to sync (aiosqlite → standard sqlite driver)
    sync_url = url.replace("sqlite+aiosqlite://", "sqlite://")

    _ensure_db_dir(sync_url)
    return create_engine(sync_url, echo=echo)


def _ensure_db_dir(url: str) -> None:
    """Create parent directory for a file-based SQLite URL if it does not exist."""
    if not url.startswith("sqlite:///") or url == "sqlite:///:memory:":
        return
    db_file = Path(url.replace("sqlite:///", ""))
    if not db_file.is_absolute():
        # Resolve relative to project root (three levels above this file)
        db_file = Path(__file__).resolve().parents[3] / db_file
    db_file.parent.mkdir(parents=True, exist_ok=True)


# ── Initialisation ─────────────────────────────────────────────────────────────


def init_db(engine: Engine | None = None) -> list[str]:
    """
    Create all tables defined in the schema if they do not already exist.

    Returns the list of table names that now exist in the database.
    """
    if engine is None:
        engine = get_engine()

    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.info("Database ready. Tables: %s", tables)
    return tables


# ── Session ────────────────────────────────────────────────────────────────────


@contextmanager
def get_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy Session and commit on clean exit, rollback on error.

    Usage::

        with get_session() as session:
            session.add(obj)
    """
    if engine is None:
        engine = get_engine()

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Bulk write ─────────────────────────────────────────────────────────────────


def upsert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> int:
    """
    Insert or update all rows in *df* into *table_name*.

    Columns in *df* that do not exist in the target table are silently dropped.
    pandas NaN / NaT values are converted to SQL NULL before insertion.
    Conflicts on the primary key are resolved by updating all non-PK columns.

    Returns the number of rows written.
    """
    if df.empty:
        return 0

    if engine is None:
        engine = get_engine()

    table = Base.metadata.tables.get(table_name)
    if table is None:
        raise ValueError(
            f"Unknown table '{table_name}'. "
            f"Call init_db() first and use one of: {list(Base.metadata.tables)}"
        )

    # Keep only columns that exist in the target schema
    valid_cols = {col.name for col in table.columns}
    df = df[[c for c in df.columns if c in valid_cols]].copy()

    # Convert NaN / NaT → None so SQLAlchemy passes NULL to SQLite
    df = df.where(df.notna(), other=None)

    pk_cols = {col.name for col in table.primary_key.columns}

    # For autoincrement-PK tables, use the first UniqueConstraint as the conflict target
    # so that re-ingesting the same game/player/week updates rather than duplicates.
    unique_constraints = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    if unique_constraints:
        conflict_cols = [col.name for col in unique_constraints[0].columns]
    else:
        conflict_cols = list(pk_cols)

    non_pk_cols = [col.name for col in table.columns if col.name not in set(conflict_cols)]

    records = df.to_dict(orient="records")
    total = 0

    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            batch = records[i : i + chunk_size]
            stmt = sqlite_insert(table).values(batch)
            if non_pk_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=conflict_cols,
                    set_={col: getattr(stmt.excluded, col) for col in non_pk_cols},
                )
            else:
                stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)
            total += len(batch)

    logger.debug("upsert_dataframe: wrote %d rows to '%s'", total, table_name)
    return total


# ── Read ───────────────────────────────────────────────────────────────────────


def read_table(
    table_name: str,
    engine: Engine | None = None,
    filters: dict[str, Any] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read *table_name* into a DataFrame.

    Args:
        table_name: Name of the target table.
        engine:     SQLAlchemy Engine. Uses get_engine() default if None.
        filters:    Dict of ``{column: value}`` equality filters (AND-combined).
        columns:    Subset of columns to select. Selects all if None.

    Returns:
        A pandas DataFrame; empty DataFrame (with correct columns) if no rows match.
    """
    if engine is None:
        engine = get_engine()

    table = Base.metadata.tables.get(table_name)
    if table is None:
        raise ValueError(f"Unknown table '{table_name}'.")

    if columns:
        query = select(*[table.c[c] for c in columns])
    else:
        query = select(table)

    if filters:
        for col_name, value in filters.items():
            query = query.where(table.c[col_name] == value)

    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# ── Convenience ────────────────────────────────────────────────────────────────


def table_row_counts(engine: Engine | None = None) -> dict[str, int]:
    """Return a dict mapping each table name to its current row count."""
    if engine is None:
        engine = get_engine()

    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for table_name in Base.metadata.tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))  # noqa: S608
            counts[table_name] = result.scalar_one()
    return counts
