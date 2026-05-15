"""
Betting odds ingestion module.

Supports two data sources:
  1. Local CSV  — Kaggle-style or pre-normalised NFL betting datasets.
  2. The Odds API — optional, requires THE_ODDS_API_KEY in .env.

If neither source is available the project still runs; betting features
are simply skipped and the user receives a clear WARNING-level message.

Normalised output columns (all others are dropped):
  game_id, season, week, home_team, away_team,
  spread_line, total_line, moneyline_home, moneyline_away,
  closing_spread, closing_total
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from nfl_chatbot.data.database import upsert_dataframe
from nfl_chatbot.data.ingest_nflverse import clean_column_names

logger = logging.getLogger(__name__)

# ── Team name → nflverse abbreviation ─────────────────────────────────────────
# Covers historical relocations so Kaggle datasets going back to 2020 match.

TEAM_NAME_TO_ABBR: dict[str, str] = {
    # Full names
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
    "washington football team": "WAS",
    "washington redskins": "WAS",
    # Historical/relocated franchises
    "oakland raiders": "LV",
    "san diego chargers": "LAC",
    "st. louis rams": "LA",
    "st louis rams": "LA",
    # City-only short forms found in some datasets
    "arizona": "ARI",
    "atlanta": "ATL",
    "baltimore": "BAL",
    "buffalo": "BUF",
    "carolina": "CAR",
    "chicago": "CHI",
    "cincinnati": "CIN",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "green bay": "GB",
    "houston": "HOU",
    "indianapolis": "IND",
    "jacksonville": "JAX",
    "kansas city": "KC",
    "las vegas": "LV",
    "la chargers": "LAC",
    "la rams": "LA",
    "miami": "MIA",
    "minnesota": "MIN",
    "new england": "NE",
    "new orleans": "NO",
    "ny giants": "NYG",
    "ny jets": "NYJ",
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "san francisco": "SF",
    "seattle": "SEA",
    "tampa bay": "TB",
    "tennessee": "TEN",
    "washington": "WAS",
    "oakland": "LV",
    "san diego": "LAC",
}

# Maps string playoff week labels (common in Kaggle datasets) to week numbers
_PLAYOFF_WEEK_MAP: dict[str, int] = {
    "wildcard": 19,
    "wild card": 19,
    "wild-card": 19,
    "divisional": 20,
    "division": 20,
    "conference": 21,
    "championship": 21,
    "superbowl": 22,
    "super bowl": 22,
    "super-bowl": 22,
}

# Normalised output schema (all non-nullable must be present)
_OUTPUT_COLS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "spread_line",
    "total_line",
    "moneyline_home",
    "moneyline_away",
    "closing_spread",
    "closing_total",
    "source",
]

_NUMERIC_COLS = [
    "spread_line",
    "total_line",
    "moneyline_home",
    "moneyline_away",
    "closing_spread",
    "closing_total",
]


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class OddsIngestResult:
    source: str                          # "csv" | "api" | "none"
    rows: int = 0
    seasons_found: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ok(self) -> bool:
        return self.source != "none"

    @property
    def empty(self) -> bool:
        return self.rows == 0


# ── Public utilities ───────────────────────────────────────────────────────────


def normalize_team(name: str | Any) -> str | None:
    """
    Convert a team name or abbreviation to a nflverse 2-3 letter abbreviation.

    Returns None if the name cannot be resolved (caller should log a warning).
    """
    if pd.isna(name) or not str(name).strip():
        return None
    key = str(name).strip().lower()
    # Already an abbreviation (2-3 upper-case letters)
    if key.upper() in {v for v in TEAM_NAME_TO_ABBR.values()}:
        return key.upper()
    return TEAM_NAME_TO_ABBR.get(key)


def normalize_week(raw: Any) -> int | None:
    """
    Coerce a week value to an integer.

    Accepts integers, numeric strings ("1"), and playoff labels ("Wildcard").
    Returns None when the value cannot be resolved.
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    if s.isdigit():
        return int(s)
    # Handle float-as-string ("1.0")
    try:
        return int(float(s))
    except ValueError:
        pass
    return _PLAYOFF_WEEK_MAP.get(s)


def build_game_id(season: int, week: int, home: str, away: str) -> str:
    """
    Build a nflverse-style game_id: {season}_{week:02d}_{away}_{home}.

    Raises ValueError if home or away cannot be normalised.
    """
    h = normalize_team(home)
    a = normalize_team(away)
    if h is None:
        raise ValueError(f"Unknown home team: {home!r}")
    if a is None:
        raise ValueError(f"Unknown away team: {away!r}")
    return f"{season}_{week:02d}_{a}_{h}"


# ── CSV ingestion ──────────────────────────────────────────────────────────────


def load_from_csv(path: Path, seasons: list[int]) -> pd.DataFrame:
    """
    Load odds from a local CSV file and return a normalised DataFrame.

    Supports two layouts auto-detected from column names:
      - *Normalised*: columns already named game_id / season / week /
        home_team / away_team / spread_line / …
      - *Kaggle-style*: columns like schedule_season / schedule_week /
        team_home / team_away / spread_favorite / over_under_line / …

    Raises FileNotFoundError when *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Odds CSV not found: {path}")

    raw = pd.read_csv(path, low_memory=False)
    raw = clean_column_names(raw)

    fmt = _detect_csv_format(raw)
    logger.info("Detected CSV format: %s", fmt)

    if fmt == "normalised":
        df = _parse_normalised(raw)
    elif fmt == "kaggle":
        df = _parse_kaggle(raw)
    else:
        raise ValueError(
            f"Unrecognised CSV format. Found columns: {list(raw.columns[:10])}. "
            "Expected either normalised (season/week/home_team/away_team/spread_line) "
            "or Kaggle-style (schedule_season/schedule_week/team_home/team_away/spread_favorite)."
        )

    # Season filter
    df = df[df["season"].isin(seasons)].reset_index(drop=True)
    logger.info("CSV loaded: %d odds rows for seasons %s", len(df), seasons)
    return df


def _detect_csv_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"season", "week", "home_team", "away_team"}.issubset(cols):
        return "normalised"
    if {"schedule_season", "schedule_week", "team_home", "team_away"}.issubset(cols):
        return "kaggle"
    return "unknown"


def _parse_normalised(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a CSV that already uses our target column names."""
    df = df.copy()
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = df["week"].apply(normalize_week)

    # Normalise team abbreviations
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)
    df = df.dropna(subset=["season", "week", "home_team", "away_team"]).copy()

    # Build or re-build game_id to guarantee it matches nflverse format
    if "game_id" not in df.columns:
        df["game_id"] = df.apply(
            lambda r: _safe_game_id(r["season"], r["week"], r["home_team"], r["away_team"]),
            axis=1,
        )

    df["source"] = "csv"
    return _finalise(df)


def _parse_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the common Kaggle-style NFL scores & betting dataset."""
    out = pd.DataFrame()

    out["season"] = pd.to_numeric(df["schedule_season"], errors="coerce").astype("Int64")
    out["week"] = df["schedule_week"].apply(normalize_week)
    out["home_team"] = df["team_home"].apply(normalize_team)
    out["away_team"] = df["team_away"].apply(normalize_team)

    out = out.dropna(subset=["season", "week", "home_team", "away_team"])

    # Spread: in Kaggle datasets, spread_favorite is stored from the HOME team's perspective —
    # negative when home is the favourite (giving points), positive when home is the underdog
    # (receiving points). No sign conversion is needed; use the value directly.
    if "spread_favorite" in df.columns:
        out["spread_line"] = pd.to_numeric(df["spread_favorite"], errors="coerce")
    else:
        out["spread_line"] = pd.NA

    out["total_line"] = (
        pd.to_numeric(df.get("over_under_line", pd.Series(dtype=float)), errors="coerce")
    )

    # Kaggle datasets rarely include moneyline or closing lines
    for col in ("moneyline_home", "moneyline_away", "closing_spread", "closing_total"):
        src = df.get(col, pd.Series(dtype=float))
        out[col] = pd.to_numeric(src, errors="coerce")

    out["game_id"] = out.apply(
        lambda r: _safe_game_id(r["season"], r["week"], r["home_team"], r["away_team"]),
        axis=1,
    )

    out["source"] = "kaggle"
    return _finalise(out)


def _safe_game_id(season: Any, week: Any, home: Any, away: Any) -> str | None:
    try:
        return build_game_id(int(season), int(week), str(home), str(away))
    except (ValueError, TypeError):
        return None


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all output columns exist, drop rows with no usable game_id."""
    for col in _OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(subset=["game_id"])
    df = df[_OUTPUT_COLS].drop_duplicates(subset=["game_id"])
    return df.reset_index(drop=True)


# ── The Odds API ingestion ─────────────────────────────────────────────────────

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_NFL_SPORT = "americanfootball_nfl"
_MARKETS = "h2h,spreads,totals"
_REGIONS = "us"


def load_from_api(
    seasons: list[int],
    api_key: str,
    rate_limit: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch NFL odds from The Odds API for each requested season.

    NOTE: Historical event odds require a paid tier. The free tier covers
    upcoming games only. This function fetches what is available and warns
    when the response is empty for a season.

    Returns a normalised DataFrame in the same format as load_from_csv().
    """
    frames: list[pd.DataFrame] = []

    for season in seasons:
        try:
            rows = _fetch_season_from_api(season, api_key, rate_limit)
            if rows:
                frames.append(pd.DataFrame(rows))
                logger.info("  Odds API: %d rows for %d", len(rows), season)
            else:
                logger.warning(
                    "  Odds API returned no data for season %d "
                    "(historical odds may require a paid tier).",
                    season,
                )
        except requests.RequestException as exc:
            logger.error("  Odds API request failed for season %d: %s", season, exc)

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = pd.concat(frames, ignore_index=True)
    return _finalise(df)


def _fetch_season_from_api(
    season: int, api_key: str, rate_limit: float
) -> list[dict]:
    """Fetch all NFL events and their odds for one season via The Odds API."""
    # Season roughly spans Sep–Feb; use a wide window and filter later.
    url = f"{_ODDS_API_BASE}/sports/{_NFL_SPORT}/odds"
    params: dict[str, str] = {
        "apiKey": api_key,
        "regions": _REGIONS,
        "markets": _MARKETS,
        "dateFormat": "iso",
        "oddsFormat": "american",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(rate_limit)

    events = resp.json()
    rows: list[dict] = []

    for event in events:
        row = _parse_api_event(event, season)
        if row:
            rows.append(row)

    return rows


def _nfl_week_from_iso(commence_time: str, season: int) -> int | None:
    """Approximate NFL week from an ISO-8601 commence_time string."""
    try:
        # Parse date portion only (avoid tz complexity)
        game_date = date.fromisoformat(commence_time[:10])
        # First Thursday of September is the earliest possible week-1 kickoff
        sept_1 = date(season, 9, 1)
        days_ahead = (3 - sept_1.weekday()) % 7  # Thursday = weekday 3
        first_thursday = sept_1 + timedelta(days=days_ahead)
        # Week 1 window opens on the Wednesday before that Thursday
        week1_start = first_thursday - timedelta(days=1)
        if game_date < week1_start:
            return None
        week = (game_date - week1_start).days // 7 + 1
        return min(max(week, 1), 22)
    except Exception:
        return None


def _parse_api_event(event: dict, season: int) -> dict | None:
    home = normalize_team(event.get("home_team", ""))
    away = normalize_team(event.get("away_team", ""))
    if not home or not away:
        return None

    # Derive week from commence_time
    commence_time: str = event.get("commence_time", "")
    week: int | None = _nfl_week_from_iso(commence_time, season) if commence_time else None

    game_id = _safe_game_id(season, week or 0, home, away)

    spread_home: float | None = None
    total: float | None = None
    ml_home: int | None = None
    ml_away: int | None = None

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            if key == "spreads":
                home_outcome = outcomes.get(event.get("home_team", ""))
                if home_outcome:
                    spread_home = home_outcome.get("point")
            elif key == "totals":
                over = outcomes.get("Over")
                if over:
                    total = over.get("point")
            elif key == "h2h":
                home_outcome = outcomes.get(event.get("home_team", ""))
                away_outcome = outcomes.get(event.get("away_team", ""))
                if home_outcome:
                    ml_home = home_outcome.get("price")
                if away_outcome:
                    ml_away = away_outcome.get("price")
        break  # Use first bookmaker only

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "home_team": home,
        "away_team": away,
        "spread_line": spread_home,
        "total_line": total,
        "moneyline_home": ml_home,
        "moneyline_away": ml_away,
        "closing_spread": spread_home,
        "closing_total": total,
        "source": "api",
    }


# ── Orchestrator ───────────────────────────────────────────────────────────────


def ingest(
    seasons: list[int],
    csv_path: Path | None = None,
    api_key: str | None = None,
    engine: Any | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    rate_limit: float = 1.0,
) -> OddsIngestResult:
    """
    Load odds from the best available source, save to disk and SQLite.

    Source priority: CSV → The Odds API → none (with warning).

    Args:
        seasons:      List of seasons to ingest.
        csv_path:     Path to a local Kaggle-style or normalised CSV.
        api_key:      The Odds API key (optional).
        engine:       SQLAlchemy engine. If None, SQLite insertion is skipped.
        raw_dir:      Directory to write raw parquet output.
        processed_dir:Directory to write processed CSV output.
        rate_limit:   Seconds to sleep between API requests.

    Returns:
        OddsIngestResult describing what was loaded.
    """
    result = OddsIngestResult(source="none")

    # ── Try CSV ────────────────────────────────────────────────────────────────
    if csv_path is not None:
        try:
            df = load_from_csv(csv_path, seasons)
            result = OddsIngestResult(
                source="csv",
                rows=len(df),
                seasons_found=sorted(df["season"].dropna().unique().tolist()),
                df=df,
            )
        except FileNotFoundError:
            msg = (
                f"Odds CSV not found at {csv_path}. "
                "Provide a Kaggle-style NFL betting CSV to enable betting features."
            )
            logger.warning(msg)
            result.warnings.append(msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to parse odds CSV ({exc}). Trying API fallback."
            logger.warning(msg)
            result.warnings.append(msg)

    # ── Try API (if CSV failed or was not specified) ────────────────────────────
    if not result.ok and api_key:
        try:
            df = load_from_api(seasons, api_key, rate_limit)
            if not df.empty:
                result = OddsIngestResult(
                    source="api",
                    rows=len(df),
                    seasons_found=sorted(df["season"].dropna().unique().tolist()),
                    df=df,
                )
            else:
                msg = (
                    "The Odds API returned no data. "
                    "Historical odds may require a paid API tier."
                )
                logger.warning(msg)
                result.warnings.append(msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"The Odds API request failed: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)

    # ── No source available ────────────────────────────────────────────────────
    if not result.ok:
        _warn_no_odds()
        return result

    df = result.df

    # ── Persist ────────────────────────────────────────────────────────────────
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "odds.parquet"
        df.to_parquet(raw_path, index=False)
        logger.debug("Saved raw odds → %s", raw_path)

    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{min(seasons)}_{max(seasons)}"
        csv_out = processed_dir / f"odds_{tag}.csv"
        df.to_csv(csv_out, index=False)
        logger.debug("Saved processed odds → %s", csv_out)

    if engine is not None:
        # Rename to match odds table schema column names before upserting
        db_df = df.rename(columns={
            "spread_line": "spread_home",
            "moneyline_home": "home_ml",
            "moneyline_away": "away_ml",
        })
        n = upsert_dataframe(db_df, "odds", engine)
        logger.info("Inserted %d odds rows into SQLite", n)

    logger.info(
        "Odds ingestion complete — source=%s rows=%d seasons=%s",
        result.source, result.rows, result.seasons_found,
    )
    return result


def _warn_no_odds() -> None:
    msg = (
        "\n"
        "  ╔══════════════════════════════════════════════════════════════╗\n"
        "  ║  NO ODDS DATA FOUND — betting features will be disabled.    ║\n"
        "  ║                                                              ║\n"
        "  ║  To enable betting features, provide one of:                ║\n"
        "  ║    1. A local Kaggle-style CSV:                             ║\n"
        "  ║         python scripts/ingest_odds.py --csv path/to/file    ║\n"
        "  ║    2. An API key in .env:                                   ║\n"
        "  ║         THE_ODDS_API_KEY=your_key_here                      ║\n"
        "  ║                                                              ║\n"
        "  ║  The project will continue without betting features.        ║\n"
        "  ╚══════════════════════════════════════════════════════════════╝\n"
    )
    warnings.warn(msg, UserWarning, stacklevel=3)
    logger.warning("No odds data available. Betting features will be skipped.")
