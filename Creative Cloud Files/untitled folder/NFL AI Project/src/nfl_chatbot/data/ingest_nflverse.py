"""
Nflverse data ingestion module.

Attempts to load data via nflreadpy first. If nflreadpy is not installed or
a call fails, falls back to downloading parquet/CSV files directly from the
nflverse GitHub releases. All data is saved to data/raw/ (parquet) and
data/processed/ (CSV) then inserted into SQLite.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from sqlalchemy.engine import Engine

from nfl_chatbot.data.database import upsert_dataframe

logger = logging.getLogger(__name__)

# ── nflverse public URLs ───────────────────────────────────────────────────────

_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

_STATIC_URLS: dict[str, str] = {
    "schedules": f"{_BASE}/schedules/schedules.parquet",
    "teams": f"{_BASE}/nflverse_teams/teams_colors_logos.csv",
}


def _player_stats_url(season: int) -> str:
    return f"{_BASE}/player_stats/player_stats_{season}.parquet"


def _roster_url(season: int) -> str:
    return f"{_BASE}/weekly_rosters/roster_{season}.parquet"


def _team_stats_url(season: int) -> str:
    return f"{_BASE}/team_stats/team_stats_{season}.parquet"


# ── Column mappings (nflverse → our schema) ───────────────────────────────────

_TEAMS_MAP: dict[str, str] = {
    "team_abbr": "team_abbr",
    "team_name": "team_name",
    "team_nick": "team_nick",
    "team_conf": "team_conf",
    "team_division": "team_division",
    "team_color": "team_color",
    "team_logo_wikipedia": "team_logo_url",
}

_SCHEDULES_MAP: dict[str, str] = {
    "game_id": "game_id",
    "season": "season",
    "game_type": "game_type",
    "week": "week",
    "gameday": "gameday",
    "gametime": "gametime",
    "away_team": "away_team",
    "home_team": "home_team",
    "away_score": "away_score",
    "home_score": "home_score",
    "overtime": "overtime",
    "div_game": "div_game",
    "roof": "roof",
    "surface": "surface",
    "stadium": "stadium",
    "spread_line": "spread_line",
    "total_line": "total_line",
}

_TEAM_STATS_MAP: dict[str, str] = {
    "recent_team": "team",
    "season": "season",
    "week": "week",
    "game_id": "game_id",
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "first_downs": "first_downs",
    "penalties": "penalties",
    "penalty_yards": "penalty_yards",
    "turnovers": "turnovers",
    "sacks": "sacks_allowed",
    "third_down_pct": "third_down_pct",
    "red_zone_pct": "red_zone_pct",
    "time_of_possession": "time_of_possession",
}

_PLAYER_STATS_MAP: dict[str, str] = {
    "player_id": "player_id",
    "player_display_name": "player_name",
    "recent_team": "team",
    "season": "season",
    "week": "week",
    "season_type": "season_type",
    "position": "position",
    "completions": "completions",
    "attempts": "attempts",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "interceptions",
    "carries": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "fantasy_points": "fantasy_points",
    "fantasy_points_ppr": "fantasy_points_ppr",
}

# Numeric columns to coerce in player stats
_PLAYER_NUMERIC = [
    "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "fantasy_points", "fantasy_points_ppr",
]

# Target column sets for each DB table
_GAME_STATS_COLS = [
    "game_id", "player_id", "player_name", "team",
    "season", "week", "position",
    "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "fantasy_points", "fantasy_points_ppr",
]

_SEASON_SUM_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds",
    "interceptions", "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "fantasy_points",
]


# ── Result container ───────────────────────────────────────────────────────────


@dataclass
class IngestResult:
    dataset: str
    rows_downloaded: int = 0
    rows_inserted: int = 0
    raw_path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Module-level utilities ─────────────────────────────────────────────────────


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and replace whitespace/hyphens with underscores."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def _try_nflreadpy(func_name: str, **kwargs: object) -> pd.DataFrame | None:
    """
    Attempt to call nflreadpy.<func_name>(**kwargs).

    Returns None (instead of raising) if nflreadpy is not installed, the
    function does not exist, or the call raises any exception. The caller
    should fall back to a direct URL download.
    """
    try:
        import nflreadpy as nfl  # type: ignore[import-untyped]

        func: Callable | None = getattr(nfl, func_name, None)
        if func is None:
            return None
        result = func(**kwargs)
        if isinstance(result, pd.DataFrame) and not result.empty:
            logger.debug("nflreadpy.%s OK — %d rows", func_name, len(result))
            return result
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("nflreadpy.%s unavailable: %s", func_name, exc)
        return None


def _download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = 60,
    max_retries: int = 3,
    rate_limit: float = 1.5,
    user_agent: str = "NFL-AI-Research-Bot/1.0 (educational project)",
) -> Path:
    """
    Download *url* to *dest* with exponential-backoff retries.

    Raises RuntimeError after all retries are exhausted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("  GET %s (attempt %d/%d)", url, attempt, max_retries)
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": user_agent},
                stream=True,
            )
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)
            time.sleep(rate_limit)
            return dest
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts: {exc}"
                ) from exc
            wait = 2.0 ** attempt
            logger.warning(
                "  Attempt %d failed: %s — retrying in %.0fs", attempt, exc, wait
            )
            time.sleep(wait)

    raise RuntimeError("Unreachable")  # satisfies type checker


def _read_file(path: Path) -> pd.DataFrame:
    """Read parquet or CSV from *path* into a DataFrame."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _apply_map(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns in *col_map* that exist in *df*, leave others untouched."""
    present = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=present)


# ── Ingester ───────────────────────────────────────────────────────────────────


class NflverseIngester:
    """
    Downloads (or loads) NFL data from nflverse and inserts it into SQLite.

    Load priority for each dataset:
      1. nflreadpy — if installed and returns non-empty data
      2. Cached file on disk — skip download if raw file already exists
      3. Direct download from nflverse GitHub releases
    """

    def __init__(
        self,
        seasons: list[int],
        raw_dir: Path,
        processed_dir: Path,
        engine: Engine,
        rate_limit: float = 1.5,
    ) -> None:
        self.seasons = seasons
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.engine = engine
        self.rate_limit = rate_limit

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest_all(self) -> dict[str, IngestResult]:
        """Run all ingestion steps. Returns a result dict keyed by dataset name."""
        logger.info(
            "Starting nflverse ingestion — seasons %s", self.seasons
        )

        results: dict[str, IngestResult] = {}
        # Order matters: teams and games must exist before per-game stats (FK refs)
        for name, method in [
            ("teams", self.ingest_teams),
            ("schedules", self.ingest_schedules),
            ("team_stats", self.ingest_team_stats),
            ("player_stats", self.ingest_player_stats),
            ("rosters", self.ingest_rosters),
        ]:
            results[name] = self._run(name, method)

        ok = sum(1 for r in results.values() if r.ok)
        logger.info("Ingestion complete: %d/%d datasets succeeded", ok, len(results))
        return results

    def ingest_teams(self) -> IngestResult:
        logger.info("── Teams")

        df = _try_nflreadpy("load_teams") or _try_nflreadpy("import_teams")

        if df is None:
            dest = self.raw_dir / "teams_colors_logos.csv"
            if not dest.exists():
                _download_file(
                    _STATIC_URLS["teams"], dest, rate_limit=self.rate_limit
                )
            df = _read_file(dest)

        df = clean_column_names(df)
        df = self._clean_teams(df)

        raw_path = self._save_raw(df, "teams")
        self._save_processed(df, "teams")
        n = upsert_dataframe(df, "teams", self.engine)

        logger.info("  → %d team rows written to SQLite", n)
        return IngestResult("teams", rows_downloaded=len(df), rows_inserted=n, raw_path=raw_path)

    def ingest_schedules(self) -> IngestResult:
        logger.info("── Schedules (seasons %s)", self.seasons)

        df = (
            _try_nflreadpy("load_schedules", seasons=self.seasons)
            or _try_nflreadpy("import_schedules", seasons=self.seasons)
        )

        if df is None:
            dest = self.raw_dir / "schedules.parquet"
            if not dest.exists():
                _download_file(
                    _STATIC_URLS["schedules"], dest, rate_limit=self.rate_limit
                )
            df = _read_file(dest)

        df = clean_column_names(df)
        df = self._clean_schedules(df)
        df = df[df["season"].isin(self.seasons)].reset_index(drop=True)

        raw_path = self._save_raw(df, "schedules")
        self._save_processed(df, "schedules")
        n = upsert_dataframe(df, "games", self.engine)

        logger.info("  → %d game rows written to SQLite", n)
        return IngestResult("schedules", rows_downloaded=len(df), rows_inserted=n, raw_path=raw_path)

    def ingest_team_stats(self) -> IngestResult:
        logger.info("── Team game stats (seasons %s)", self.seasons)

        frames: list[pd.DataFrame] = []
        for season in self.seasons:
            dest = self.raw_dir / f"team_stats_{season}.parquet"
            try:
                if not dest.exists():
                    _download_file(
                        _team_stats_url(season), dest, rate_limit=self.rate_limit
                    )
                frames.append(_read_file(dest))
                logger.info("  loaded team_stats %d (%d rows)", season, len(frames[-1]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("  team_stats %d skipped: %s", season, exc)

        if not frames:
            return IngestResult("team_stats", error="No team stats seasons downloaded successfully")

        df = pd.concat(frames, ignore_index=True)
        df = clean_column_names(df)
        df = _apply_map(df, _TEAM_STATS_MAP)
        df = df[df["season"].isin(self.seasons)].reset_index(drop=True)

        # Derive is_home from game_id (format: {season}_{week}_{away}_{home})
        if "game_id" in df.columns and "team" in df.columns:
            home_abbr = df["game_id"].str.split("_").str[-1]
            df["is_home"] = df["team"] == home_abbr

        # Join games table to populate points_scored / points_allowed
        try:
            from nfl_chatbot.data.database import read_table
            games_df = read_table("games", self.engine)[
                ["game_id", "home_team", "away_team", "home_score", "away_score"]
            ]
            df = df.merge(games_df, on="game_id", how="left")
            home_mask = df["team"] == df["home_team"]
            df["points_scored"] = df["home_score"].where(home_mask, df["away_score"])
            df["points_allowed"] = df["away_score"].where(home_mask, df["home_score"])
            df = df.drop(columns=["home_team", "away_team", "home_score", "away_score"], errors="ignore")
        except Exception as exc:  # noqa: BLE001
            logger.debug("  Could not join scores for team_stats: %s", exc)

        raw_path = self._save_raw(df, "team_stats")
        self._save_processed(df, "team_stats")
        n = upsert_dataframe(df, "team_game_stats", self.engine)

        logger.info("  → %d team-game rows written to SQLite", n)
        return IngestResult("team_stats", rows_downloaded=len(df), rows_inserted=n, raw_path=raw_path)

    def ingest_player_stats(self) -> IngestResult:
        logger.info("── Player stats (seasons %s)", self.seasons)

        df = (
            _try_nflreadpy("load_player_stats", seasons=self.seasons)
            or _try_nflreadpy("import_player_stats", seasons=self.seasons)
        )

        if df is None:
            frames: list[pd.DataFrame] = []
            for season in self.seasons:
                dest = self.raw_dir / f"player_stats_{season}.parquet"
                try:
                    if not dest.exists():
                        _download_file(
                            _player_stats_url(season),
                            dest,
                            rate_limit=self.rate_limit,
                        )
                    frames.append(_read_file(dest))
                    logger.info("  loaded player_stats %d (%d rows)", season, len(frames[-1]))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("  player_stats %d skipped: %s", season, exc)

            if not frames:
                return IngestResult("player_stats", error="No seasons downloaded successfully")
            df = pd.concat(frames, ignore_index=True)

        df = clean_column_names(df)
        df = self._clean_player_stats(df)
        df = df[df["season"].isin(self.seasons)].reset_index(drop=True)

        # Regular season + postseason only
        if "season_type" in df.columns:
            df = df[df["season_type"].isin(["REG", "POST"])].copy()

        raw_path = self._save_raw(df, "player_stats")
        self._save_processed(df, "player_stats")

        # Load schedules from DB to resolve proper nflverse game_ids when absent
        try:
            from nfl_chatbot.data.database import read_table
            schedules_df = read_table("games", self.engine)
        except Exception:
            schedules_df = None

        game_level = self._build_game_stats(df, schedules_df=schedules_df)
        n_game = upsert_dataframe(game_level, "player_game_stats", self.engine)

        season_level = self._aggregate_season_stats(df)
        n_season = upsert_dataframe(season_level, "player_season_stats", self.engine)

        logger.info(
            "  → %d player-game rows + %d player-season rows written to SQLite",
            n_game, n_season,
        )
        return IngestResult(
            "player_stats",
            rows_downloaded=len(df),
            rows_inserted=n_game + n_season,
            raw_path=raw_path,
        )

    def ingest_rosters(self) -> IngestResult:
        logger.info("── Weekly rosters (seasons %s)", self.seasons)

        df = _try_nflreadpy("load_rosters", seasons=self.seasons)
        if df is None:
            df = _try_nflreadpy("load_weekly_rosters", seasons=self.seasons)

        if df is None:
            frames: list[pd.DataFrame] = []
            for season in self.seasons:
                dest = self.raw_dir / f"roster_{season}.parquet"
                try:
                    if not dest.exists():
                        _download_file(
                            _roster_url(season), dest, rate_limit=self.rate_limit
                        )
                    frames.append(_read_file(dest))
                    logger.info("  loaded roster %d (%d rows)", season, len(frames[-1]))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("  roster %d skipped: %s", season, exc)

            if not frames:
                return IngestResult("rosters", error="No rosters downloaded successfully")
            df = pd.concat(frames, ignore_index=True)

        df = clean_column_names(df)
        if "season" in df.columns:
            df = df[df["season"].isin(self.seasons)].reset_index(drop=True)

        raw_path = self._save_raw(df, "rosters")
        self._save_processed(df, "rosters")

        # Rosters are reference data saved to disk; no dedicated DB table
        logger.info("  → %d roster rows saved to disk (no DB table)", len(df))
        return IngestResult(
            "rosters", rows_downloaded=len(df), rows_inserted=0, raw_path=raw_path
        )

    # ── Cleaning helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_teams(df: pd.DataFrame) -> pd.DataFrame:
        df = _apply_map(df, _TEAMS_MAP)

        # Accept alternate logo column names
        if "team_logo_url" not in df.columns:
            for alt in ("team_logo_espn", "team_logo_wikipedia", "team_wordmark"):
                if alt in df.columns:
                    df["team_logo_url"] = df[alt]
                    break

        keep = [c for c in _TEAMS_MAP.values() if c in df.columns]
        return df[keep].drop_duplicates(subset=["team_abbr"]).reset_index(drop=True)

    @staticmethod
    def _clean_schedules(df: pd.DataFrame) -> pd.DataFrame:
        df = _apply_map(df, _SCHEDULES_MAP)

        # Derive home_win from result (= home_score − away_score in nflverse)
        if "result" in df.columns:
            df["home_win"] = df["result"].apply(
                lambda x: True if (pd.notna(x) and float(x) > 0)
                else (False if (pd.notna(x) and float(x) < 0) else None)
            )
        elif {"home_score", "away_score"}.issubset(df.columns):
            scored = df["home_score"].notna() & df["away_score"].notna()
            df["home_win"] = pd.NA
            df.loc[scored, "home_win"] = (
                df.loc[scored, "home_score"] > df.loc[scored, "away_score"]
            )

        # Derive neutral_site from location column
        if "location" in df.columns:
            df["neutral_site"] = (
                df["location"].str.strip().str.lower() == "neutral"
            )
            df = df.drop(columns=["location"])

        # Normalise overtime to nullable bool
        if "overtime" in df.columns:
            df["overtime"] = df["overtime"].astype("boolean")

        target = [
            "game_id", "season", "week", "game_type",
            "home_team", "away_team",
            "home_score", "away_score", "home_win", "overtime",
            "neutral_site", "div_game",
            "roof", "surface", "stadium",
            "gameday", "gametime",
            "spread_line", "total_line",
        ]
        keep = [c for c in target if c in df.columns]
        return df[keep].drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    @staticmethod
    def _clean_player_stats(df: pd.DataFrame) -> pd.DataFrame:
        df = _apply_map(df, _PLAYER_STATS_MAP)

        # Accept alternate player_id column names
        if "player_id" not in df.columns:
            for alt in ("gsis_id", "pfr_id", "esb_id"):
                if alt in df.columns:
                    df["player_id"] = df[alt]
                    break

        # Coerce numeric columns
        for col in _PLAYER_NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.reset_index(drop=True)

    # ── Derived tables ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_game_stats(
        df: pd.DataFrame,
        schedules_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Produce one row per (player, game) for player_game_stats."""
        out = df.copy()

        if "game_id" not in out.columns:
            if schedules_df is not None and not schedules_df.empty:
                # Join schedules to resolve proper nflverse game_ids
                sched = schedules_df[
                    ["game_id", "season", "week", "home_team", "away_team"]
                ].drop_duplicates()
                home_side = sched[["game_id", "season", "week", "home_team"]].rename(
                    columns={"home_team": "team"}
                )
                away_side = sched[["game_id", "season", "week", "away_team"]].rename(
                    columns={"away_team": "team"}
                )
                team_game = pd.concat([home_side, away_side]).drop_duplicates()
                merge_cols = [c for c in ["season", "week", "team"] if c in out.columns]
                out = out.merge(team_game, on=merge_cols, how="left")

            # Fallback for any rows still without a game_id
            if "game_id" not in out.columns or out["game_id"].isna().any():
                team = out.get("team", pd.Series(["UNK"] * len(out), index=out.index))
                synthetic = (
                    out["season"].astype(str)
                    + "_"
                    + out["week"].astype(str).str.zfill(2)
                    + "_"
                    + team.astype(str)
                )
                if "game_id" not in out.columns:
                    out["game_id"] = synthetic
                else:
                    out["game_id"] = out["game_id"].fillna(synthetic)

        keep = [c for c in _GAME_STATS_COLS if c in out.columns]
        dedup_on = (
            ["game_id", "player_id"]
            if "player_id" in out.columns
            else ["game_id"]
        )
        return out[keep].drop_duplicates(subset=dedup_on).reset_index(drop=True)

    @staticmethod
    def _aggregate_season_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate weekly player stats to season totals for player_season_stats."""
        group_cols = [
            c for c in ["player_id", "player_name", "team", "season", "position"]
            if c in df.columns
        ]
        sum_cols = [c for c in _SEASON_SUM_COLS if c in df.columns]

        season_df = (
            df.groupby(group_cols, as_index=False)[sum_cols]
            .sum(min_count=1)
        )

        # Games played = number of weeks with a non-null fantasy_points entry
        if "player_id" in df.columns and "season" in df.columns:
            games_played = (
                df[df["fantasy_points"].notna()]
                .groupby(["player_id", "season"])
                .size()
                .reset_index(name="games_played")
            )
            season_df = season_df.merge(
                games_played, on=["player_id", "season"], how="left"
            )

        # Derived rate stats (computed from season totals, not averaged)
        if {"completions", "attempts"}.issubset(season_df.columns):
            season_df["completion_pct"] = (
                season_df["completions"]
                / season_df["attempts"].replace(0, pd.NA)
            ).round(3)

        if {"rushing_yards", "carries"}.issubset(season_df.columns):
            season_df["yards_per_carry"] = (
                season_df["rushing_yards"]
                / season_df["carries"].replace(0, pd.NA)
            ).round(2)

        if {"receiving_yards", "receptions"}.issubset(season_df.columns):
            season_df["yards_per_reception"] = (
                season_df["receiving_yards"]
                / season_df["receptions"].replace(0, pd.NA)
            ).round(2)

        if "fantasy_points" in season_df.columns and "games_played" in season_df.columns:
            season_df["fantasy_points_per_game"] = (
                season_df["fantasy_points"]
                / season_df["games_played"].replace(0, pd.NA)
            ).round(2)

        # Rename to match player_season_stats schema
        if "fantasy_points" in season_df.columns:
            season_df = season_df.rename(
                columns={"fantasy_points": "fantasy_points_total"}
            )

        dedup_on = (
            ["player_id", "season"]
            if "player_id" in season_df.columns
            else None
        )
        return season_df.drop_duplicates(subset=dedup_on).reset_index(drop=True)

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _save_raw(self, df: pd.DataFrame, name: str) -> Path:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.debug("  raw  → %s (%d rows)", path.name, len(df))
        return path

    def _save_processed(self, df: pd.DataFrame, name: str) -> Path:
        tag = f"{min(self.seasons)}_{max(self.seasons)}"
        path = self.processed_dir / f"{name}_{tag}.csv"
        df.to_csv(path, index=False)
        logger.debug("  proc → %s (%d rows)", path.name, len(df))
        return path

    # ── Error-tolerant runner ──────────────────────────────────────────────────

    def _run(self, name: str, method: Callable[[], IngestResult]) -> IngestResult:
        try:
            return method()
        except Exception as exc:  # noqa: BLE001
            logger.error("%s ingestion failed: %s", name, exc, exc_info=True)
            return IngestResult(name, error=str(exc))
