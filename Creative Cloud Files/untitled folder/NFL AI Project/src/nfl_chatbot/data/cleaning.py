"""
Data cleaning pipeline for NFL AI Chatbot.

Pure-DataFrame transforms: each clean_* function accepts a raw/ingested
DataFrame, applies normalisation, and returns (cleaned_df, CleaningReport).
No file I/O is performed here — see scripts/clean_data.py for the runner.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_chatbot.data.ingest_odds import TEAM_NAME_TO_ABBR, build_game_id, normalize_team

logger = logging.getLogger(__name__)

# ── Canonical status maps ──────────────────────────────────────────────────────

_STATUS_ALIASES: dict[str, str | None] = {
    "out": "Out",
    "doubtful": "Doubtful",
    "questionable": "Questionable",
    "q": "Questionable",
    "probable": "Probable",
    "p": "Probable",
    "ir": "IR",
    "injured reserve": "IR",
    "pup": "PUP",
    "physically unable to perform": "PUP",
    "dnr": "DNR",
    "did not report": "DNR",
    "": None,
    "-": None,
    "—": None,
    "n/a": None,
    "none": None,
}

_GAME_TYPE_ALIASES: dict[str, str] = {
    "reg": "REG",
    "regular": "REG",
    "regular season": "REG",
    "wc": "WC",
    "wildcard": "WC",
    "wild card": "WC",
    "wild-card": "WC",
    "div": "DIV",
    "divisional": "DIV",
    "con": "CON",
    "conf": "CON",
    "conference": "CON",
    "championship": "CON",
    "sb": "SB",
    "superbowl": "SB",
    "super bowl": "SB",
    "super-bowl": "SB",
}

# ── Practice participation codes ───────────────────────────────────────────────

_PRACTICE_FULL = frozenset({"fp", "full", "full participation"})
_PRACTICE_LIMITED = frozenset({"lp", "limited", "limited participation"})
_PRACTICE_DNP = frozenset({"dnp", "did not participate", "no practice"})
_EMPTY_VALUES = frozenset({"", "-", "—", "n/a", "none"})

# ── Required columns per dataset ───────────────────────────────────────────────

_REQUIRED: dict[str, list[str]] = {
    "games": ["season", "week", "home_team", "away_team"],
    "team_stats": ["game_id", "team", "season", "week"],
    "player_stats": ["season", "week", "team"],
    "injuries": ["season", "week", "team", "player_name"],
    "odds": ["season", "week", "home_team", "away_team"],
}

# Count stats filled with 0 when absent; float/rate stats stay NA.
_COUNT_COLS: frozenset[str] = frozenset({
    "completions", "attempts", "passing_tds", "interceptions",
    "carries", "rushing_tds", "targets", "receptions", "receiving_tds",
    "turnovers", "penalties", "penalty_yards", "first_downs", "sacks_allowed",
    "home_score", "away_score", "points_scored", "points_allowed",
})


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class CleaningReport:
    """Summary of what a cleaning step changed."""

    table: str
    rows_in: int
    rows_out: int
    dupes_dropped: int = 0
    bad_teams_dropped: int = 0
    missing_filled: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def __str__(self) -> str:
        lines = [
            f"[{self.table}] {self.rows_in} → {self.rows_out} rows",
        ]
        if self.dupes_dropped:
            lines.append(f"  duplicates dropped: {self.dupes_dropped}")
        if self.bad_teams_dropped:
            lines.append(f"  bad teams dropped:  {self.bad_teams_dropped}")
        for col, n in self.missing_filled.items():
            lines.append(f"  filled {col}: {n} values")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ── Public helpers ─────────────────────────────────────────────────────────────


def standardize_team_abbreviations(series: pd.Series) -> pd.Series:
    """
    Map a Series of team names / abbreviations to canonical nflverse abbreviations.

    Accepts full names ("Kansas City Chiefs"), city-only forms ("Kansas City"),
    historical names ("Oakland Raiders"), and existing abbreviations ("KC").
    Values that cannot be resolved become pd.NA.
    """
    return series.apply(normalize_team)


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
    table_name: str = "",
) -> None:
    """Raise ValueError when any column in *required* is absent from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        ctx = f" for {table_name!r}" if table_name else ""
        raise ValueError(
            f"Missing required columns{ctx}: {missing}. "
            f"Available: {list(df.columns)}"
        )


def handle_missing_values(
    df: pd.DataFrame,
    strategies: dict[str, str],
) -> pd.DataFrame:
    """
    Fill or drop NaN values per column according to *strategies*.

    Strategy values:
      "zero"         — fill with 0
      "median"       — fill with column median
      "mean"         — fill with column mean
      "mode"         — fill with most-frequent value
      "forward_fill" — forward-fill (assumes time-ordered rows)
      "drop"         — drop rows where this column is NaN

    Columns absent from *df* are silently skipped.
    Returns a new DataFrame; input is never modified.
    Raises ValueError for unknown strategy strings.
    """
    _VALID = {"zero", "median", "mean", "mode", "forward_fill", "drop"}
    bad = {s for s in strategies.values() if s not in _VALID}
    if bad:
        raise ValueError(f"Unknown strategies: {bad}. Valid options: {_VALID}")

    df = df.copy()
    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null == 0:
            continue

        if strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        elif strategy == "forward_fill":
            df[col] = df[col].ffill()
        elif strategy == "drop":
            df = df.dropna(subset=[col])

        logger.debug("handle_missing: %s — filled %d NaN via '%s'", col, n_null, strategy)

    return df.reset_index(drop=True)


# ── Per-table cleaners ─────────────────────────────────────────────────────────


def clean_games(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a games / schedules DataFrame.

    - Standardizes home_team / away_team to nflverse abbreviations.
    - Converts gameday to datetime.date.
    - Normalizes game_type to REG / WC / DIV / CON / SB.
    - Builds a stable game_id ({season}_{week:02d}_{away}_{home}) when
      the column is absent or has NaN entries.
    - Removes duplicate game_ids (keeps first occurrence).
    """
    df = df.copy()
    report = CleaningReport(table="games", rows_in=len(df), rows_out=0)
    validate_required_columns(df, _REQUIRED["games"], "games")

    for col in ("home_team", "away_team"):
        df[col] = standardize_team_abbreviations(df[col])

    bad = df["home_team"].isna() | df["away_team"].isna()
    if bad.any():
        n = int(bad.sum())
        report.bad_teams_dropped += n
        report.warnings.append(f"Dropped {n} rows with unresolvable team abbreviations")
        df = df[~bad].copy()

    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce").dt.date

    if "game_type" in df.columns:
        original = df["game_type"].copy()
        df["game_type"] = (
            df["game_type"].str.strip().str.lower()
            .map(_GAME_TYPE_ALIASES)
            .fillna(original.str.strip().str.upper())
        )

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Build game_id: create if absent, fill NaN entries if partially present.
    built = df.apply(
        lambda r: _safe_game_id(r["season"], r["week"], r["home_team"], r["away_team"]),
        axis=1,
    )
    if "game_id" not in df.columns:
        df["game_id"] = built
    else:
        df["game_id"] = df["game_id"].where(df["game_id"].notna(), other=built)

    df = df.dropna(subset=["game_id"])

    before = len(df)
    df = df.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    report.dupes_dropped = before - len(df)

    report.rows_out = len(df)
    logger.info("clean_games: %d → %d rows (%d dupes)", report.rows_in, report.rows_out, report.dupes_dropped)
    return df, report


def clean_team_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a team_game_stats DataFrame.

    - Standardizes team abbreviation.
    - Fills count-stat NaNs with 0; leaves rate-stat NaNs as NA.
    - Removes duplicate (game_id, team) pairs.
    """
    df = df.copy()
    report = CleaningReport(table="team_stats", rows_in=len(df), rows_out=0)
    validate_required_columns(df, _REQUIRED["team_stats"], "team_stats")

    df["team"] = standardize_team_abbreviations(df["team"])
    bad = df["team"].isna()
    if bad.any():
        n = int(bad.sum())
        report.bad_teams_dropped += n
        report.warnings.append(f"Dropped {n} rows with unresolvable team abbreviations")
        df = df[~bad].copy()

    count_cols = {c: "zero" for c in _COUNT_COLS if c in df.columns}
    if count_cols:
        for col in count_cols:
            n_null = int(df[col].isna().sum())
            if n_null:
                report.missing_filled[col] = n_null
        df = handle_missing_values(df, count_cols)

    before = len(df)
    df = df.drop_duplicates(subset=["game_id", "team"], keep="first").reset_index(drop=True)
    report.dupes_dropped = before - len(df)

    report.rows_out = len(df)
    logger.info("clean_team_stats: %d → %d rows", report.rows_in, report.rows_out)
    return df, report


def clean_player_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a player_game_stats or player_season_stats DataFrame.

    - Validates that at least player_id or player_name is present.
    - Standardizes team abbreviation.
    - Drops rows missing both player_id and player_name.
    - Fills count-stat NaNs with 0.
    - Removes duplicate (game_id, player_id) rows when both columns exist.
    """
    df = df.copy()
    report = CleaningReport(table="player_stats", rows_in=len(df), rows_out=0)
    validate_required_columns(df, _REQUIRED["player_stats"], "player_stats")

    if "player_id" not in df.columns and "player_name" not in df.columns:
        raise ValueError("player_stats must contain at least player_id or player_name")

    df["team"] = standardize_team_abbreviations(df["team"])
    bad = df["team"].isna()
    if bad.any():
        n = int(bad.sum())
        report.bad_teams_dropped += n
        df = df[~bad].copy()

    has_id = df.get("player_id", pd.Series(dtype=object, index=df.index)).notna()
    has_name = df.get("player_name", pd.Series(dtype=object, index=df.index)).notna()
    empty = ~(has_id | has_name)
    if empty.any():
        n = int(empty.sum())
        report.warnings.append(f"Dropped {n} rows with no player_id or player_name")
        df = df[~empty].copy()

    count_cols = {c: "zero" for c in _COUNT_COLS if c in df.columns}
    if count_cols:
        for col in count_cols:
            n_null = int(df[col].isna().sum())
            if n_null:
                report.missing_filled[col] = n_null
        df = handle_missing_values(df, count_cols)

    dedup_cols = [c for c in ["game_id", "player_id"] if c in df.columns]
    if dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        report.dupes_dropped = before - len(df)

    report.rows_out = len(df)
    logger.info("clean_player_stats: %d → %d rows", report.rows_in, report.rows_out)
    return df, report


def clean_injuries(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean an injuries DataFrame (from scraping or nflverse).

    - Standardizes team abbreviation.
    - Normalizes game_status → Out / Doubtful / Questionable / Probable / IR / PUP / None.
    - Normalizes practice columns → FP / LP / DNP / None.
    - Removes duplicate (season, week, player_id) rows; falls back to
      (season, week, player_name, team) when player_id is absent.
    """
    df = df.copy()
    report = CleaningReport(table="injuries", rows_in=len(df), rows_out=0)
    validate_required_columns(df, _REQUIRED["injuries"], "injuries")

    df["team"] = standardize_team_abbreviations(df["team"])
    bad = df["team"].isna()
    if bad.any():
        n = int(bad.sum())
        report.bad_teams_dropped += n
        report.warnings.append(f"Dropped {n} rows with unresolvable team abbreviations")
        df = df[~bad].copy()

    if "game_status" in df.columns:
        df["game_status"] = df["game_status"].apply(_normalise_game_status)

    for col in ("practice_status", "practice_wed", "practice_thu", "practice_fri"):
        if col in df.columns:
            df[col] = df[col].apply(_normalise_practice)

    dedup_cols: list[str]
    if "player_id" in df.columns:
        dedup_cols = ["season", "week", "player_id"]
    else:
        dedup_cols = ["season", "week", "player_name", "team"]

    before = len(df)
    # keep="last" so the most-recent update wins (injury status changes across week)
    df = df.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
    report.dupes_dropped = before - len(df)

    report.rows_out = len(df)
    logger.info("clean_injuries: %d → %d rows", report.rows_in, report.rows_out)
    return df, report


def clean_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a betting odds DataFrame.

    - Standardizes home_team / away_team.
    - Coerces all numeric odds columns.
    - Rebuilds game_id from season / week / home_team / away_team for stability.
    - Removes duplicate game_ids.
    """
    df = df.copy()
    report = CleaningReport(table="odds", rows_in=len(df), rows_out=0)
    validate_required_columns(df, _REQUIRED["odds"], "odds")

    for col in ("home_team", "away_team"):
        df[col] = standardize_team_abbreviations(df[col])

    bad = df["home_team"].isna() | df["away_team"].isna()
    if bad.any():
        n = int(bad.sum())
        report.bad_teams_dropped += n
        report.warnings.append(f"Dropped {n} rows with unresolvable team abbreviations")
        df = df[~bad].copy()

    for col in (
        "spread_line", "spread_home", "spread_away", "total_line",
        "home_ml", "away_ml", "moneyline_home", "moneyline_away",
        "closing_spread", "closing_total",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Always rebuild game_id from components so it matches nflverse format exactly.
    df["game_id"] = df.apply(
        lambda r: _safe_game_id(r["season"], r["week"], r["home_team"], r["away_team"]),
        axis=1,
    )
    df = df.dropna(subset=["game_id"])

    before = len(df)
    df = df.drop_duplicates(subset=["game_id"], keep="first").reset_index(drop=True)
    report.dupes_dropped = before - len(df)

    report.rows_out = len(df)
    logger.info("clean_odds: %d → %d rows", report.rows_in, report.rows_out)
    return df, report


# ── Pipeline ───────────────────────────────────────────────────────────────────


def run_pipeline(
    processed_dir: Path,
    *,
    seasons: list[int] | None = None,
) -> dict[str, CleaningReport]:
    """
    Discover CSV files in *processed_dir*, run the matching cleaner, and write
    cleaned outputs to the same directory with a ``_cleaned`` suffix.

    File-name prefix routing (case-insensitive):
        schedules_* / games_*  → clean_games
        team_stats_*           → clean_team_stats
        player_stats_*         → clean_player_stats
        injuries_*             → clean_injuries
        odds_*                 → clean_odds

    Returns {dataset_name: CleaningReport} for every file processed.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    _CLEANERS: dict[str, Any] = {
        "games": clean_games,
        "schedules": clean_games,
        "team_stats": clean_team_stats,
        "player_stats": clean_player_stats,
        "injuries": clean_injuries,
        "odds": clean_odds,
    }

    reports: dict[str, CleaningReport] = {}

    for csv_path in sorted(processed_dir.glob("*.csv")):
        if "_cleaned" in csv_path.stem:
            continue

        prefix = _csv_prefix(csv_path.stem)
        cleaner = _CLEANERS.get(prefix)
        if cleaner is None:
            logger.debug("No cleaner for %s — skipping", csv_path.name)
            continue

        try:
            raw = pd.read_csv(csv_path, low_memory=False)

            # Optional season filter
            if seasons and "season" in raw.columns:
                raw = raw[raw["season"].isin(seasons)].reset_index(drop=True)

            cleaned, report = cleaner(raw)
            out_path = processed_dir / f"{csv_path.stem}_cleaned.csv"
            cleaned.to_csv(out_path, index=False)
            reports[prefix] = report
            logger.info("%-20s → %-40s (%d rows)", csv_path.name, out_path.name, report.rows_out)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to clean %s: %s", csv_path.name, exc, exc_info=True)
            reports[prefix] = CleaningReport(
                table=prefix, rows_in=0, rows_out=0, warnings=[str(exc)]
            )

    return reports


# ── Internal helpers ───────────────────────────────────────────────────────────


def _safe_game_id(season: Any, week: Any, home: Any, away: Any) -> str | None:
    try:
        return build_game_id(int(season), int(week), str(home), str(away))
    except (ValueError, TypeError):
        return None


def _normalise_game_status(value: Any) -> str | None:
    if pd.isna(value):
        return None
    v = str(value).strip()
    lower = v.lower()
    if lower in _STATUS_ALIASES:
        return _STATUS_ALIASES[lower]
    return v  # preserve unknown values as-is


def _normalise_practice(value: Any) -> str | None:
    if pd.isna(value):
        return None
    v = str(value).strip()
    lower = v.lower()
    if lower in _EMPTY_VALUES:
        return None
    if lower in _PRACTICE_FULL:
        return "FP"
    if lower in _PRACTICE_LIMITED:
        return "LP"
    if lower in _PRACTICE_DNP:
        return "DNP"
    return v.upper()


def _csv_prefix(stem: str) -> str:
    """Extract dataset name from a stem like 'schedules_2020_2024'."""
    return re.sub(r"_\d{4}.*$", "", stem).lower()
