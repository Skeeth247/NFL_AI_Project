"""
Team-level feature engineering for NFL game predictions.

Anti-leakage guarantee
----------------------
Every rolling feature is computed with ``shift(1)`` before ``rolling(N)``.
This means the value assigned to game G is the mean of games [G-N, G-1] —
the current game is never part of its own window. Season-to-date features
use ``shift(1).expanding()`` within each (team, season) group, so the
first game of a season starts with NaN (filled later for modelling).

Output
------
One row per game (game_id), with ``home_<feature>`` / ``away_<feature>``
columns for team-specific stats and unprefixed columns for game-level
context (spread, rest days, H2H, Elo, etc.).

The modeling table contains 75+ engineered features plus target columns
(``home_win``, ``home_covered``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Injury designations
_OUT_STATUSES = frozenset({"Out", "IR", "PUP", "DNR"})
_DOUBTFUL_STATUSES = frozenset({"Doubtful"})
_QUESTIONABLE_STATUSES = frozenset({"Questionable"})

# Skill positions used for starter injury proxy
_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "OT", "OG", "C", "OL"})

# Roof types considered a dome (weather-neutral)
_DOME_ROOFS = frozenset({"dome", "closed", "retractable"})

# Playoff game types
_PLAYOFF_TYPES = frozenset({"WC", "DIV", "CON", "SB"})


# ── Config dataclass ───────────────────────────────────────────────────────────


@dataclass
class EloConfig:
    k_factor: float = 20.0
    initial_rating: float = 1500.0
    home_advantage: float = 48.0
    regression_to_mean: float = 0.33


# ── Rolling helper ─────────────────────────────────────────────────────────────


def _rolling_shift(series: pd.Series, window: int, agg: str = "mean") -> pd.Series:
    """
    Apply shift(1) then rolling(window) aggregation to *series*.

    The series must already be sorted chronologically within its group
    before this function is called. shift(1) ensures the current game
    is never in its own rolling window.
    """
    shifted = series.shift(1)
    if agg == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    if agg == "sum":
        return shifted.rolling(window, min_periods=1).sum()
    raise ValueError(f"Unknown agg: {agg!r}")


def _season_expanding(series: pd.Series) -> pd.Series:
    """
    Season-to-date expanding mean with shift(1) for leakage prevention.

    Groups must be (team, season) so the window resets each new season.
    """
    return series.shift(1).expanding(min_periods=1).mean()


# ── FeatureBuilder ─────────────────────────────────────────────────────────────


class FeatureBuilder:
    """
    Build the full modeling table from database tables.

    All feature computation is done in-memory on DataFrames loaded from the
    database. If a table is empty (e.g. team_game_stats hasn't been ingested
    yet), the builder falls back to game-level data (scores only) so the
    pipeline always produces output.
    """

    def __init__(
        self,
        engine: Engine,
        window: int = 4,
        elo: EloConfig | None = None,
        h2h_lookback: int = 5,
    ) -> None:
        self.engine = engine
        self.window = window
        self.elo_cfg = elo or EloConfig()
        self.h2h_lookback = h2h_lookback

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """
        Build and return the full modeling table.

        Returns a DataFrame with one row per game containing all engineered
        features plus target columns (home_win, home_covered).
        """
        logger.info("FeatureBuilder.build() — window=%d", self.window)

        games = self._load_games()
        if games.empty:
            logger.warning("No games found in database — returning empty table")
            return pd.DataFrame()

        stats = self._load_team_stats(games)
        injuries = self._load_injuries()

        logger.info("Loaded: %d games, %d team-game-stat rows, %d injury rows",
                    len(games), len(stats), len(injuries))

        # ── Per-team time series ───────────────────────────────────────────────
        team_ts = self._build_team_time_series(stats, games)
        team_ts = team_ts.sort_values(["team", "season", "week"]).reset_index(drop=True)

        rolling_df = self._compute_rolling_features(team_ts)
        seas_df = self._compute_season_features(team_ts)

        team_features = rolling_df.merge(seas_df, on=["game_id", "team"], how="left")

        # ── Game-level base table ──────────────────────────────────────────────
        table = games[[
            "game_id", "season", "week", "game_type",
            "home_team", "away_team",
            "home_score", "away_score", "home_win",
            "neutral_site", "div_game", "gameday",
            "spread_line", "total_line", "roof",
        ]].copy()

        # ── Attach home / away team features ──────────────────────────────────
        feat_cols = [c for c in team_features.columns if c not in ("game_id", "team")]

        home_link = games[["game_id", "home_team"]].rename(columns={"home_team": "team"})
        home_feats = (
            team_features.merge(home_link, on=["game_id", "team"])
            [["game_id"] + feat_cols]
            .rename(columns={c: f"home_{c}" for c in feat_cols})
        )

        away_link = games[["game_id", "away_team"]].rename(columns={"away_team": "team"})
        away_feats = (
            team_features.merge(away_link, on=["game_id", "team"])
            [["game_id"] + feat_cols]
            .rename(columns={c: f"away_{c}" for c in feat_cols})
        )

        table = table.merge(home_feats, on="game_id", how="left")
        table = table.merge(away_feats, on="game_id", how="left")

        # ── Schedule features ──────────────────────────────────────────────────
        table = self._attach_schedule_features(table, games)

        # ── Betting market features ────────────────────────────────────────────
        table = self._attach_betting_features(table)

        # ── Injury features ────────────────────────────────────────────────────
        table = self._attach_injury_features(table, injuries)

        # ── Head-to-head features ──────────────────────────────────────────────
        table = self._attach_h2h_features(table, games)

        # ── Elo ratings ────────────────────────────────────────────────────────
        elo_df = self._compute_elo(games)
        table = table.merge(elo_df, on="game_id", how="left")

        # ── Target: home_covered ──────────────────────────────────────────────
        table = self._attach_home_covered(table)

        # ── Drop raw score columns (target leakage guard) ─────────────────────
        table = table.drop(columns=["home_score", "away_score", "roof"], errors="ignore")

        logger.info("Modeling table complete: %d rows × %d columns", len(table), len(table.columns))
        return table.reset_index(drop=True)

    # ── Data loaders ───────────────────────────────────────────────────────────

    def _load_games(self) -> pd.DataFrame:
        try:
            from nfl_chatbot.data.database import read_table
            df = read_table("games", self.engine)
            if "gameday" in df.columns:
                df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
            return df
        except Exception as exc:
            logger.error("Failed to load games: %s", exc)
            return pd.DataFrame()

    def _load_team_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        try:
            from nfl_chatbot.data.database import read_table
            df = read_table("team_game_stats", self.engine)
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("team_game_stats unavailable (%s); using scores only", exc)

        # Fallback: build minimal stats from games table
        return self._stats_from_games(games)

    def _load_injuries(self) -> pd.DataFrame:
        try:
            from nfl_chatbot.data.database import read_table
            return read_table("injuries", self.engine)
        except Exception as exc:
            logger.warning("injuries table unavailable: %s", exc)
            return pd.DataFrame()

    # ── Team time-series construction ──────────────────────────────────────────

    @staticmethod
    def _stats_from_games(games: pd.DataFrame) -> pd.DataFrame:
        """Minimal per-team stats derived from scores when team_game_stats is absent."""
        home = games[["game_id", "season", "week", "home_team", "home_score", "away_score"]].copy()
        home = home.rename(columns={"home_team": "team", "home_score": "pts_scored", "away_score": "pts_allowed"})
        home["is_home"] = True

        away = games[["game_id", "season", "week", "away_team", "away_score", "home_score"]].copy()
        away = away.rename(columns={"away_team": "team", "away_score": "pts_scored", "home_score": "pts_allowed"})
        away["is_home"] = False

        ts = pd.concat([home, away], ignore_index=True)
        ts["won"] = (ts["pts_scored"] > ts["pts_allowed"]).astype(float)
        ts["pt_diff"] = ts["pts_scored"] - ts["pts_allowed"]
        return ts

    def _build_team_time_series(
        self,
        stats: pd.DataFrame,
        games: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enrich per-team stats with opponent stats (for defensive features).

        For each (game, team) row, the opponent's offensive stats become this
        team's defensive outcomes: yards_allowed, pts_allowed, takeaways, sacks.

        Returns one row per (game_id, team).
        """
        if stats.empty:
            return self._stats_from_games(games)

        # Columns to pull from the opponent's row
        opp_src_cols = [
            c for c in ("pts_scored", "total_yards", "turnovers", "sacks_allowed")
            if c in stats.columns
        ]
        opp_rename = {c: f"opp_{c}" for c in opp_src_cols}
        opp_rename["team"] = "opp_team"

        opp = stats[["game_id", "team"] + opp_src_cols].rename(columns=opp_rename)

        # Join each row with the opponent's row in the same game
        merged = stats.merge(opp, on="game_id", how="left")
        merged = merged[merged["team"] != merged["opp_team"]].copy()

        # Defensive outcomes
        if "opp_total_yards" in merged.columns:
            merged["yards_allowed"] = merged["opp_total_yards"]
        if "opp_turnovers" in merged.columns:
            merged["takeaways"] = merged["opp_turnovers"]
        if "opp_sacks_allowed" in merged.columns:
            merged["def_sacks"] = merged["opp_sacks_allowed"]
        if "opp_pts_scored" in merged.columns:
            # Prefer existing pts_allowed column; fall back to opp_pts_scored
            if "points_allowed" in merged.columns:
                merged["pts_allowed"] = (
                    merged["points_allowed"].fillna(merged["opp_pts_scored"])
                )
            else:
                merged["pts_allowed"] = merged["opp_pts_scored"]

        # Rename for consistency
        col_renames: dict[str, str] = {}
        if "points_scored" in merged.columns and "pts_scored" not in merged.columns:
            col_renames["points_scored"] = "pts_scored"
        if "total_yards" in merged.columns and "yards_gained" not in merged.columns:
            col_renames["total_yards"] = "yards_gained"
        if "passing_yards" in merged.columns and "pass_yds" not in merged.columns:
            col_renames["passing_yards"] = "pass_yds"
        if "rushing_yards" in merged.columns and "rush_yds" not in merged.columns:
            col_renames["rushing_yards"] = "rush_yds"
        if col_renames:
            merged = merged.rename(columns=col_renames)

        # Derived
        if "pts_scored" in merged.columns and "pts_allowed" in merged.columns:
            merged["pt_diff"] = merged["pts_scored"].fillna(0) - merged["pts_allowed"].fillna(0)
            merged["won"] = (merged["pts_scored"] > merged["pts_allowed"]).astype(float)

        if "takeaways" in merged.columns and "turnovers" in merged.columns:
            merged["turnover_diff"] = merged["takeaways"].fillna(0) - merged["turnovers"].fillna(0)

        # Yards per first-down proxy for efficiency
        if "yards_gained" in merged.columns and "first_downs" in merged.columns:
            merged["yds_per_first_down"] = (
                merged["yards_gained"] / (merged["first_downs"].replace(0, np.nan))
            )

        return merged.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # ── Rolling features ───────────────────────────────────────────────────────

    def _compute_rolling_features(self, ts: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling-window features per team.

        All features use shift(1).rolling(window) so the current game is
        never included in its own feature value.
        """
        w = self.window
        suffix = f"_avg{w}"
        sum_suffix = f"_last{w}"

        # Columns to roll: (source_col, output_suffix, agg)
        roll_spec: list[tuple[str, str, str]] = [
            # Form
            ("pts_scored",       f"pts_scored{suffix}",   "mean"),
            ("pts_allowed",      f"pts_allowed{suffix}",  "mean"),
            ("pt_diff",          f"pt_diff{suffix}",      "mean"),
            ("won",              f"wins{sum_suffix}",     "sum"),
            ("won",              f"win_pct{sum_suffix}",  "mean"),
            # Yards
            ("yards_gained",     f"yards_gained{suffix}", "mean"),
            ("yards_allowed",    f"yards_allowed{suffix}","mean"),
            ("yds_per_first_down", f"yds_per_play{suffix}", "mean"),
            # Offensive
            ("pass_yds",         f"pass_yds{suffix}",     "mean"),
            ("rush_yds",         f"rush_yds{suffix}",     "mean"),
            ("turnovers",        f"turnovers{suffix}",    "mean"),
            ("third_down_pct",   f"third_down_pct{suffix}","mean"),
            ("red_zone_pct",     f"red_zone_pct{suffix}", "mean"),
            ("first_downs",      f"first_downs{suffix}",  "mean"),
            ("penalties",        f"penalties{suffix}",    "mean"),
            # Defensive
            ("def_sacks",        f"def_sacks{suffix}",    "mean"),
            ("sacks_allowed",    f"sacks_allowed{suffix}","mean"),
            ("takeaways",        f"takeaways{suffix}",    "mean"),
            ("turnover_diff",    f"turnover_diff{suffix}", "mean"),
        ]

        result = ts[["game_id", "team"]].copy()

        ts_sorted = ts.sort_values(["team", "season", "week"])

        for src_col, out_col, agg in roll_spec:
            if src_col not in ts_sorted.columns:
                result[out_col] = np.nan
                continue
            result[out_col] = (
                ts_sorted.groupby("team", sort=False)[src_col]
                .transform(lambda g, _w=w, _a=agg: _rolling_shift(g, _w, _a))
                .values
            )

        return result

    def _compute_season_features(self, ts: pd.DataFrame) -> pd.DataFrame:
        """Season-to-date averages (expanding window, resets each new season)."""
        seas_spec: list[tuple[str, str]] = [
            ("pts_scored",  "pts_scored_seas"),
            ("pts_allowed", "pts_allowed_seas"),
            ("won",         "win_pct_seas"),
        ]

        result = ts[["game_id", "team"]].copy()
        ts_sorted = ts.sort_values(["team", "season", "week"])

        for src_col, out_col in seas_spec:
            if src_col not in ts_sorted.columns:
                result[out_col] = np.nan
                continue
            result[out_col] = (
                ts_sorted.groupby(["team", "season"], sort=False)[src_col]
                .transform(_season_expanding)
                .values
            )

        return result

    # ── Schedule features ──────────────────────────────────────────────────────

    def _attach_schedule_features(
        self,
        table: pd.DataFrame,
        games: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add rest days, short-week flags, div_game, playoff_game, neutral_site,
        and is_dome to *table*.

        rest_days is the number of days since each team's previous game.
        Defaults to 7 when no prior game exists (start of dataset).
        """
        # Build (team, game_date) time series
        home_games = games[["game_id", "home_team", "gameday"]].rename(
            columns={"home_team": "team"}
        )
        away_games = games[["game_id", "away_team", "gameday"]].rename(
            columns={"away_team": "team"}
        )
        all_team_games = pd.concat([home_games, away_games], ignore_index=True)
        all_team_games = all_team_games.sort_values(["team", "gameday"])
        all_team_games["prev_gameday"] = (
            all_team_games.groupby("team")["gameday"].shift(1)
        )
        all_team_games["rest_days"] = (
            (all_team_games["gameday"] - all_team_games["prev_gameday"])
            .dt.days
            .fillna(7)
            .astype(int)
        )

        # Home rest
        home_rest = (
            all_team_games.merge(
                games[["game_id", "home_team"]].rename(columns={"home_team": "team"}),
                on=["game_id", "team"],
            )[["game_id", "rest_days"]]
            .rename(columns={"rest_days": "home_rest_days"})
        )
        # Away rest
        away_rest = (
            all_team_games.merge(
                games[["game_id", "away_team"]].rename(columns={"away_team": "team"}),
                on=["game_id", "team"],
            )[["game_id", "rest_days"]]
            .rename(columns={"rest_days": "away_rest_days"})
        )

        table = table.merge(home_rest, on="game_id", how="left")
        table = table.merge(away_rest, on="game_id", how="left")

        table["home_rest_days"] = table["home_rest_days"].fillna(7).astype(int)
        table["away_rest_days"] = table["away_rest_days"].fillna(7).astype(int)
        table["rest_diff"] = table["home_rest_days"] - table["away_rest_days"]
        table["home_short_week"] = (table["home_rest_days"] < 7).astype(int)
        table["away_short_week"] = (table["away_rest_days"] < 7).astype(int)

        # div_game / playoff_game / neutral_site already in table from games join
        if "div_game" not in table.columns:
            table["div_game"] = np.nan
        else:
            table["div_game"] = table["div_game"].fillna(False).astype(int)

        table["playoff_game"] = (
            table.get("game_type", pd.Series(dtype=str))
            .isin(_PLAYOFF_TYPES)
            .astype(int)
        )

        if "neutral_site" in table.columns:
            table["neutral_site"] = table["neutral_site"].fillna(False).astype(int)
        else:
            table["neutral_site"] = 0

        # is_dome: True when roof is closed/dome/retractable
        if "roof" in table.columns:
            table["is_dome"] = (
                table["roof"].str.lower().isin(_DOME_ROOFS)
            ).astype(int)
        else:
            table["is_dome"] = 0

        return table

    # ── Betting features ───────────────────────────────────────────────────────

    @staticmethod
    def _attach_betting_features(table: pd.DataFrame) -> pd.DataFrame:
        """
        Derive implied totals and favorite/underdog flags from spread_line and total_line.

        spread_line < 0 → home team is favored (home gives points).
        Implied totals:
            home_implied_total = (total_line − spread_line) / 2
            away_implied_total = (total_line + spread_line) / 2
        implied_home_win_prob from moneyline equivalent of spread (simplified):
            uses Elo-style logistic: 1 / (1 + 10^(spread / 20))
        """
        s = table["spread_line"]
        t = table["total_line"]

        table["home_implied_total"] = (t - s) / 2
        table["away_implied_total"] = (t + s) / 2
        table["home_is_favorite"] = (s < 0).astype(float)

        # Implied win probability from spread (logistic on spread/20)
        # When spread = 0: 50%. When home is -7: ~59%.
        table["implied_home_win_prob"] = 1 / (1 + 10 ** (s.fillna(0) / 20))

        # Ensure spread_line / total_line present (may already be from games table)
        if "spread_line" not in table.columns:
            table["spread_line"] = np.nan
        if "total_line" not in table.columns:
            table["total_line"] = np.nan

        return table

    # ── Injury features ────────────────────────────────────────────────────────

    def _attach_injury_features(
        self,
        table: pd.DataFrame,
        injuries: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Count injured players per team per game week and attach to *table*.

        Injury report is pre-game by definition (weekly report before the game).
        We join on (season, week, team) which ensures no post-game data is used.
        """
        if injuries.empty:
            for prefix in ("home", "away"):
                for col in ("injured_total", "questionable_count", "doubtful_count",
                            "out_count", "starter_proxy"):
                    table[f"{prefix}_{col}"] = 0
            return table

        inj = injuries.copy()

        # Aggregate counts per (season, week, team)
        def _agg_injuries(group: pd.DataFrame) -> pd.Series:
            total = len(group)
            q = group["game_status"].isin(_QUESTIONABLE_STATUSES).sum()
            d = group["game_status"].isin(_DOUBTFUL_STATUSES).sum()
            out = group["game_status"].isin(_OUT_STATUSES).sum()
            starter = 0
            if "position" in group.columns:
                starter = group[
                    group["position"].str.upper().isin(_SKILL_POSITIONS)
                    & group["game_status"].isin(_OUT_STATUSES | _DOUBTFUL_STATUSES)
                ].shape[0] if "game_status" in group.columns else 0
            return pd.Series({
                "injured_total": total,
                "questionable_count": int(q),
                "doubtful_count": int(d),
                "out_count": int(out),
                "starter_proxy": int(starter),
            })

        inj_counts = (
            inj.groupby(["season", "week", "team"])
            .apply(_agg_injuries)
            .reset_index()
        )

        # Merge home team injuries
        home_inj = (
            table[["game_id", "season", "week", "home_team"]]
            .merge(
                inj_counts.rename(columns={"team": "home_team"}),
                on=["season", "week", "home_team"],
                how="left",
            )
        )
        inj_cols = ["injured_total", "questionable_count", "doubtful_count", "out_count", "starter_proxy"]
        home_inj = (
            home_inj[["game_id"] + inj_cols]
            .rename(columns={c: f"home_{c}" for c in inj_cols})
            .fillna(0)
        )

        # Merge away team injuries
        away_inj = (
            table[["game_id", "season", "week", "away_team"]]
            .merge(
                inj_counts.rename(columns={"team": "away_team"}),
                on=["season", "week", "away_team"],
                how="left",
            )
        )
        away_inj = (
            away_inj[["game_id"] + inj_cols]
            .rename(columns={c: f"away_{c}" for c in inj_cols})
            .fillna(0)
        )

        table = table.merge(home_inj, on="game_id", how="left")
        table = table.merge(away_inj, on="game_id", how="left")

        for prefix in ("home", "away"):
            for col in inj_cols:
                full = f"{prefix}_{col}"
                if full in table.columns:
                    table[full] = table[full].fillna(0).astype(int)

        return table

    # ── Head-to-head features ──────────────────────────────────────────────────

    def _attach_h2h_features(
        self,
        table: pd.DataFrame,
        games: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each game G between home team H and away team A, compute:
          - h2h_home_win_rate      : H win % in the last N H2H matchups before G
          - h2h_home_pt_diff       : avg point differential for H in last N H2H
          - h2h_all_time_home_win_rate : H win % in all H2H matchups before G

        Only games with known outcomes and ``gameday < game G's gameday`` are used.
        """
        completed = games[
            games["home_win"].notna()
            & games["home_score"].notna()
            & games["away_score"].notna()
        ].copy()
        completed["gameday"] = pd.to_datetime(completed["gameday"], errors="coerce")
        completed["matchup_key"] = completed.apply(
            lambda r: "_".join(sorted([str(r["home_team"]), str(r["away_team"])])),
            axis=1,
        )

        rows: list[dict] = []
        for _, game in table.iterrows():
            home = str(game["home_team"])
            away = str(game["away_team"])
            key = "_".join(sorted([home, away]))
            game_date = pd.to_datetime(game.get("gameday"))

            prior: pd.DataFrame
            if pd.notna(game_date):
                prior = completed[
                    (completed["matchup_key"] == key)
                    & (completed["gameday"] < game_date)
                ]
            else:
                prior = completed[completed["matchup_key"] == key]

            all_prior = prior
            recent = prior.tail(self.h2h_lookback)

            def _home_win(row: pd.Series) -> int:
                if row["home_team"] == home:
                    return int(bool(row["home_win"]))
                return int(not bool(row["home_win"]))

            def _home_pt_diff(row: pd.Series) -> float:
                diff = float(row["home_score"]) - float(row["away_score"])
                return diff if row["home_team"] == home else -diff

            if recent.empty:
                h2h_win_rate = 0.5
                h2h_pt_diff = 0.0
            else:
                wins = recent.apply(_home_win, axis=1).sum()
                h2h_win_rate = wins / len(recent)
                h2h_pt_diff = recent.apply(_home_pt_diff, axis=1).mean()

            if all_prior.empty:
                all_time_rate = 0.5
            else:
                wins_all = all_prior.apply(_home_win, axis=1).sum()
                all_time_rate = wins_all / len(all_prior)

            rows.append({
                "game_id": game["game_id"],
                "h2h_home_win_rate": round(h2h_win_rate, 4),
                "h2h_home_pt_diff": round(h2h_pt_diff, 2),
                "h2h_all_time_home_win_rate": round(all_time_rate, 4),
            })

        h2h_df = pd.DataFrame(rows)
        return table.merge(h2h_df, on="game_id", how="left")

    # ── Elo ratings ────────────────────────────────────────────────────────────

    def _compute_elo(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pre-game Elo ratings using the FiveThirtyEight NFL Elo model.

        Ratings are updated AFTER each game; the stored value is the pre-game
        rating. At the start of each new season, ratings regress toward the
        mean by ``regression_to_mean`` fraction.

        Returns a DataFrame with columns: game_id, home_elo, away_elo, elo_diff.
        """
        cfg = self.elo_cfg
        elo: dict[str, float] = {}
        last_season: dict[str, int] = {}

        rows: list[dict] = []

        for _, game in games.sort_values(["season", "week"]).iterrows():
            home = str(game["home_team"])
            away = str(game["away_team"])
            season = int(game["season"])

            # Initialise / regress at season start
            for team in (home, away):
                if team not in elo:
                    elo[team] = cfg.initial_rating
                    last_season[team] = season
                elif last_season[team] != season:
                    elo[team] = (
                        elo[team] * (1 - cfg.regression_to_mean)
                        + cfg.initial_rating * cfg.regression_to_mean
                    )
                    last_season[team] = season

            home_elo = elo[home]
            away_elo = elo[away]
            # Add home-field advantage to the effective difference
            elo_diff = home_elo - away_elo + cfg.home_advantage

            rows.append({
                "game_id": game["game_id"],
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
                "elo_diff": round(elo_diff, 1),
            })

            # Update post-game (only when result is known)
            if pd.notna(game.get("home_win")):
                expected = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
                actual = 1.0 if game["home_win"] else 0.0
                delta = cfg.k_factor * (actual - expected)
                elo[home] = elo[home] + delta
                elo[away] = elo[away] - delta

        return pd.DataFrame(rows)

    # ── Target: home_covered ──────────────────────────────────────────────────

    @staticmethod
    def _attach_home_covered(table: pd.DataFrame) -> pd.DataFrame:
        """
        Derive home_covered from scores and spread_line.

        home_covered = 1 when (home_score + spread_line) > away_score.
        spread_line is negative when the home team is favored.
        Returns NULL when scores or spread are unavailable.
        """
        if "home_score" not in table.columns or "away_score" not in table.columns:
            table["home_covered"] = np.nan
            return table

        has_scores = table["home_score"].notna() & table["away_score"].notna()
        has_spread = table["spread_line"].notna()
        mask = has_scores & has_spread

        table["home_covered"] = np.nan
        table.loc[mask, "home_covered"] = (
            (table.loc[mask, "home_score"] + table.loc[mask, "spread_line"])
            > table.loc[mask, "away_score"]
        ).astype(float)
        return table


# ── Convenience function ───────────────────────────────────────────────────────


def build_modeling_table(
    engine: Engine,
    window: int = 4,
    elo_cfg: EloConfig | None = None,
    h2h_lookback: int = 5,
) -> pd.DataFrame:
    """Build and return the modeling table. Convenience wrapper around FeatureBuilder."""
    builder = FeatureBuilder(engine, window=window, elo=elo_cfg, h2h_lookback=h2h_lookback)
    return builder.build()


# ── Feature column registry ────────────────────────────────────────────────────


def feature_columns(window: int = 4) -> list[str]:
    """
    Return the ordered list of model feature column names for a given window.

    Excludes identifier columns (game_id, season, week, home_team, away_team,
    gameday, game_type) and target columns (home_win, home_covered).
    """
    w = window
    avg = f"_avg{w}"
    last = f"_last{w}"

    # Per-team rolling features (each × home + away = ×2)
    per_team_rolling = [
        f"pts_scored{avg}",
        f"pts_allowed{avg}",
        f"pt_diff{avg}",
        f"wins{last}",
        f"win_pct{last}",
        f"yards_gained{avg}",
        f"yards_allowed{avg}",
        f"yds_per_play{avg}",
        f"pass_yds{avg}",
        f"rush_yds{avg}",
        f"turnovers{avg}",
        f"third_down_pct{avg}",
        f"red_zone_pct{avg}",
        f"first_downs{avg}",
        f"penalties{avg}",
        f"def_sacks{avg}",
        f"sacks_allowed{avg}",
        f"takeaways{avg}",
        f"turnover_diff{avg}",
    ]

    # Per-team season features (×2)
    per_team_season = [
        "pts_scored_seas",
        "pts_allowed_seas",
        "win_pct_seas",
    ]

    home_cols = [f"home_{c}" for c in per_team_rolling + per_team_season]
    away_cols = [f"away_{c}" for c in per_team_rolling + per_team_season]

    # Schedule (game-level)
    schedule_cols = [
        "home_rest_days", "away_rest_days", "rest_diff",
        "home_short_week", "away_short_week",
        "div_game", "playoff_game", "neutral_site", "is_dome",
    ]

    # Betting (game-level)
    betting_cols = [
        "spread_line", "total_line",
        "home_implied_total", "away_implied_total",
        "home_is_favorite", "implied_home_win_prob",
    ]

    # Injury (×2)
    injury_per_team = [
        "injured_total", "questionable_count", "doubtful_count",
        "out_count", "starter_proxy",
    ]
    injury_cols = (
        [f"home_{c}" for c in injury_per_team]
        + [f"away_{c}" for c in injury_per_team]
    )

    # H2H
    h2h_cols = [
        "h2h_home_win_rate",
        "h2h_home_pt_diff",
        "h2h_all_time_home_win_rate",
    ]

    # Elo
    elo_cols = ["home_elo", "away_elo", "elo_diff"]

    return (
        home_cols
        + away_cols
        + schedule_cols
        + betting_cols
        + injury_cols
        + h2h_cols
        + elo_cols
    )
