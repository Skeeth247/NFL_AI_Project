"""
Chatbot tool functions for the NFL AI Chatbot.

Each tool wraps one or more existing modules (models, features, evaluation)
and returns a flat, JSON-compatible dict that the chatbot response layer can
render directly.  Tools never make up answers — they either return real data
or explain clearly why data is unavailable.

All returns include these top-level keys:
  tool   : str   — the tool name
  status : str   — "ok" | "degraded" | "error"
  error  : str | None — human-readable error message when status != "ok"

Usage (class interface)
-----------------------
    tools = ChatbotTools()
    result = tools.predict_matchup("KC", "BUF", season=2024, week=12)

Usage (module-level convenience)
---------------------------------
    from nfl_chatbot.chatbot.tools import predict_matchup
    result = predict_matchup("Chiefs", "Bills")

Lazy loading
------------
Heavy resources (model, engine, modeling table) are loaded on first use and
cached inside the ChatbotTools instance.  If a resource is unavailable the
tool returns status="degraded" or status="error" with a clear message.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Team name → canonical abbreviation ────────────────────────────────────────
# Mirrors intents.py but kept local to avoid coupling to private names.
_TEAM_MAP: dict[str, str] = {
    "buffalo bills": "BUF", "buffalo": "BUF", "bills": "BUF", "buf": "BUF",
    "miami dolphins": "MIA", "miami": "MIA", "dolphins": "MIA", "mia": "MIA",
    "new england patriots": "NE", "new england": "NE", "patriots": "NE", "pats": "NE", "ne": "NE",
    "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    "baltimore ravens": "BAL", "baltimore": "BAL", "ravens": "BAL", "bal": "BAL",
    "cleveland browns": "CLE", "cleveland": "CLE", "browns": "CLE", "cle": "CLE",
    "cincinnati bengals": "CIN", "cincinnati": "CIN", "bengals": "CIN", "cin": "CIN",
    "pittsburgh steelers": "PIT", "pittsburgh": "PIT", "steelers": "PIT", "pit": "PIT",
    "houston texans": "HOU", "houston": "HOU", "texans": "HOU", "hou": "HOU",
    "indianapolis colts": "IND", "indianapolis": "IND", "colts": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jacksonville": "JAX", "jaguars": "JAX", "jags": "JAX", "jax": "JAX",
    "tennessee titans": "TEN", "tennessee": "TEN", "titans": "TEN", "ten": "TEN",
    "denver broncos": "DEN", "denver": "DEN", "broncos": "DEN", "den": "DEN",
    "kansas city chiefs": "KC", "kansas city": "KC", "chiefs": "KC", "kc": "KC",
    "las vegas raiders": "LV", "las vegas": "LV", "raiders": "LV", "lv": "LV",
    "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "dallas cowboys": "DAL", "dallas": "DAL", "cowboys": "DAL", "dal": "DAL",
    "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "philadelphia eagles": "PHI", "philadelphia": "PHI", "eagles": "PHI", "phi": "PHI",
    "washington commanders": "WAS", "washington": "WAS", "commanders": "WAS", "was": "WAS",
    "chicago bears": "CHI", "chicago": "CHI", "bears": "CHI", "chi": "CHI",
    "detroit lions": "DET", "detroit": "DET", "lions": "DET", "det": "DET",
    "green bay packers": "GB", "green bay": "GB", "packers": "GB", "gb": "GB",
    "minnesota vikings": "MIN", "minnesota": "MIN", "vikings": "MIN", "min": "MIN",
    "atlanta falcons": "ATL", "atlanta": "ATL", "falcons": "ATL", "atl": "ATL",
    "carolina panthers": "CAR", "carolina": "CAR", "panthers": "CAR", "car": "CAR",
    "new orleans saints": "NO", "new orleans": "NO", "saints": "NO", "no": "NO",
    "tampa bay buccaneers": "TB", "tampa bay": "TB", "buccaneers": "TB", "bucs": "TB", "tb": "TB",
    "arizona cardinals": "ARI", "arizona": "ARI", "cardinals": "ARI", "cards": "ARI", "ari": "ARI",
    "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR", "lar": "LAR",
    "seattle seahawks": "SEA", "seattle": "SEA", "seahawks": "SEA", "hawks": "SEA", "sea": "SEA",
    "san francisco 49ers": "SF", "san francisco": "SF", "49ers": "SF", "niners": "SF", "sf": "SF",
}
_TEAM_MAP_SORTED = sorted(_TEAM_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)


def _canon_team(name: str) -> str | None:
    """Return the nflverse abbreviation for *name*, or None if unrecognised."""
    lower = name.strip().lower()
    for alias, abbr in _TEAM_MAP_SORTED:
        if alias == lower:
            return abbr
    # substring fallback (handles "the Chiefs" etc.)
    for alias, abbr in _TEAM_MAP_SORTED:
        if alias in lower:
            return abbr
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(value: Any) -> Any:
    """Convert NaN / inf to None for JSON compatibility."""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _safe_dict(d: dict) -> dict:
    """Recursively replace NaN/inf with None in a dict."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _safe_dict(v)
        elif isinstance(v, list):
            out[k] = [_safe_dict(i) if isinstance(i, dict) else _safe(i) for i in v]
        else:
            out[k] = _safe(v)
    return out


def _ok(tool: str, **kwargs) -> dict:
    return {"tool": tool, "status": "ok", "error": None, **kwargs}


def _degraded(tool: str, msg: str, **kwargs) -> dict:
    return {"tool": tool, "status": "degraded", "error": msg, **kwargs}


def _error(tool: str, msg: str, **kwargs) -> dict:
    return {"tool": tool, "status": "error", "error": msg, **kwargs}


# ══════════════════════════════════════════════════════════════════════════════
# ChatbotTools class
# ══════════════════════════════════════════════════════════════════════════════

class ChatbotTools:
    """
    Container for all chatbot tool functions.

    Parameters
    ----------
    engine : SQLAlchemy Engine | None
        Database connection.  If None, tools that need the DB fall back to
        the CSV modeling table or return degraded results.
    model_dir : str | Path | None
        Directory containing trained model artifacts (*.joblib, *.json, *.csv).
        Defaults to ``models/`` relative to the project root.
    data_dir : str | Path | None
        Directory containing processed data files (modeling_table.csv, etc.).
        Defaults to ``data/processed/``.

    All heavy resources are loaded lazily and cached.
    """

    def __init__(
        self,
        engine=None,
        model_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        self._engine = engine
        self._model_dir = Path(model_dir) if model_dir else self._default_model_dir()
        self._data_dir = Path(data_dir) if data_dir else self._default_data_dir()

        # Lazy caches
        self._predictor = None
        self._trends = None
        self._modeling_df: pd.DataFrame | None = None
        self._last_prediction: dict | None = None   # stored by predict_matchup

    # ── Resource defaults ──────────────────────────────────────────────────────

    @staticmethod
    def _default_model_dir() -> Path:
        try:
            from nfl_chatbot.config import get_config
            return Path(get_config().app.models.output_dir)
        except Exception:
            return Path("models")

    @staticmethod
    def _default_data_dir() -> Path:
        try:
            from nfl_chatbot.config import get_config
            return Path(get_config().app.data.processed_dir)
        except Exception:
            return Path("data") / "processed"

    # ── Lazy loaders ───────────────────────────────────────────────────────────

    def _load_predictor(self):
        """Load and cache the win-prediction model."""
        if self._predictor is not None:
            return self._predictor
        try:
            from nfl_chatbot.models.predict import WinPredictor
            self._predictor = WinPredictor.from_files(
                model_path=self._model_dir / "best_model.joblib",
                feature_columns_path=self._model_dir / "feature_columns.json",
                metrics_path=self._model_dir / "model_metrics.json",
            )
        except Exception as exc:
            logger.debug("Could not load WinPredictor: %s", exc)
        return self._predictor

    def _load_modeling_df(self) -> pd.DataFrame | None:
        """Load and cache the modeling table CSV."""
        if self._modeling_df is not None:
            return self._modeling_df
        csv = self._data_dir / "modeling_table.csv"
        if csv.exists():
            try:
                self._modeling_df = pd.read_csv(csv)
                logger.debug("Loaded modeling table: %d rows from %s", len(self._modeling_df), csv)
            except Exception as exc:
                logger.debug("Could not read modeling_table.csv: %s", exc)
        return self._modeling_df

    def _load_trends(self):
        """Load and cache a BettingTrends instance from the modeling table."""
        if self._trends is not None:
            return self._trends
        df = self._load_modeling_df()
        if df is not None and not df.empty:
            try:
                from nfl_chatbot.evaluation.betting_trends import BettingTrends
                self._trends = BettingTrends(df)
            except Exception as exc:
                logger.debug("Could not build BettingTrends: %s", exc)
        return self._trends

    def _load_db_engine(self):
        """Return an existing or lazily-created DB engine."""
        if self._engine is not None:
            return self._engine
        try:
            from nfl_chatbot.data.database import get_engine
            self._engine = get_engine()
        except Exception as exc:
            logger.debug("Could not create DB engine: %s", exc)
        return self._engine

    # ── Feature lookup helpers ─────────────────────────────────────────────────

    def _team_features_from_df(
        self, df: pd.DataFrame, home: str, away: str
    ) -> dict[str, Any]:
        """
        Find the most recent row in *df* that involves *home* and *away*, and
        return its feature values as a flat dict.  Falls back to per-team latest
        rows merged if no direct matchup exists.
        """
        if df is None or df.empty:
            return {}

        # Try exact matchup first
        matchup = df[
            (df.get("home_team", pd.Series(dtype=str)) == home) &
            (df.get("away_team", pd.Series(dtype=str)) == away)
        ]
        if not matchup.empty:
            row = matchup.sort_values(["season", "week"], ascending=False).iloc[0]
            return row.to_dict()

        # Fallback: merge most-recent home-team row with most-recent away-team row
        home_rows = df[
            (df.get("home_team", pd.Series(dtype=str)) == home) |
            (df.get("away_team", pd.Series(dtype=str)) == home)
        ]
        away_rows = df[
            (df.get("home_team", pd.Series(dtype=str)) == away) |
            (df.get("away_team", pd.Series(dtype=str)) == away)
        ]

        combined: dict[str, Any] = {}
        for prefix, rows, team in [("home_", home_rows, home), ("away_", away_rows, away)]:
            if rows.empty:
                continue
            latest = rows.sort_values(["season", "week"], ascending=False).iloc[0]
            # Remap columns so home features are always prefixed "home_"
            for col, val in latest.items():
                if col.startswith("home_") and latest.get("home_team") == team:
                    combined[col] = val
                elif col.startswith("away_") and latest.get("away_team") == team:
                    # Rename away_ → home_ when this team was the away team in that row
                    combined[col.replace("away_", "home_", 1)] = val
                elif col.startswith("away_") and latest.get("home_team") == team:
                    combined[col] = val
                elif col.startswith("home_") and latest.get("away_team") == team:
                    combined[col.replace("home_", "away_", 1)] = val
                elif col not in combined:
                    combined[col] = val
        return combined

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 1: predict_matchup
    # ══════════════════════════════════════════════════════════════════════════

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        season: int | None = None,
        week: int | None = None,
    ) -> dict[str, Any]:
        """
        Predict the winner of a matchup between *home_team* and *away_team*.

        Returns win probability, confidence, and a plain-English explanation.
        Stores the result in memory so ``explain_last_prediction`` can retrieve it.
        """
        home = _canon_team(home_team) or home_team.upper()
        away = _canon_team(away_team) or away_team.upper()

        predictor = self._load_predictor()
        if predictor is None:
            return _error(
                "predict_matchup",
                f"No trained model found in {self._model_dir}. "
                "Run `python scripts/train_model.py` first.",
                home_team=home,
                away_team=away,
            )

        # ── Build feature vector ───────────────────────────────────────────────
        df = self._load_modeling_df()
        feature_values = self._team_features_from_df(df, home, away) if df is not None else {}

        # Align to the model's expected feature columns
        aligned: dict[str, Any] = {
            col: feature_values.get(col, np.nan)
            for col in predictor.feature_columns
        }

        features_available = bool(feature_values)

        # ── Predict ────────────────────────────────────────────────────────────
        try:
            prediction = predictor.predict(aligned)
        except Exception as exc:
            return _error("predict_matchup", f"Prediction failed: {exc}",
                          home_team=home, away_team=away)

        # ── Explain ────────────────────────────────────────────────────────────
        game_input = {"home_team": home, "away_team": away, "season": season, "week": week}
        try:
            from nfl_chatbot.models.explain import explain_prediction
            explanation = explain_prediction(
                game_input=game_input,
                prediction=prediction,
                feature_values=aligned,
                importance_path=self._model_dir / "feature_importance.csv",
                metrics_path=self._model_dir / "model_metrics.json",
            )
        except Exception as exc:
            logger.warning("Explanation failed: %s", exc)
            explanation = {"headline": prediction.get("home_win_prob", "N/A"), "error": str(exc)}

        result = _ok(
            "predict_matchup",
            home_team=home,
            away_team=away,
            season=season,
            week=week,
            prediction=_safe_dict(prediction),
            explanation=_safe_dict(explanation),
            features_available=features_available,
        )

        if not features_available:
            result["status"] = "degraded"
            result["error"] = (
                f"No historical data found for {home} or {away}. "
                "Prediction uses league-average feature values — treat with caution."
            )

        # Cache for explain_last_prediction
        self._last_prediction = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 2: compare_players
    # ══════════════════════════════════════════════════════════════════════════

    def compare_players(
        self,
        player_a: str,
        player_b: str,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """
        Compare two players side by side.

        Uses ``player_features.compare_players`` which queries the database.
        Falls back to a not-available response when the DB is empty.
        """
        engine = self._load_db_engine()

        if engine is None:
            return _degraded(
                "compare_players",
                "No database connection available. Ingest player data first.",
                player_a=player_a,
                player_b=player_b,
            )

        try:
            from nfl_chatbot.features.player_features import (
                compare_players as _cp,
            )
            comparison = _cp(
                player_a=player_a,
                player_b=player_b,
                seasons=seasons,
                engine=engine,
            )
        except Exception as exc:
            logger.warning("compare_players failed: %s", exc)
            return _error(
                "compare_players",
                f"Could not compare players: {exc}",
                player_a=player_a,
                player_b=player_b,
            )

        return _ok(
            "compare_players",
            player_a=player_a,
            player_b=player_b,
            seasons=seasons,
            comparison=_safe_dict(comparison),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 3: answer_betting_trend
    # ══════════════════════════════════════════════════════════════════════════

    def answer_betting_trend(self, question: str) -> dict[str, Any]:
        """
        Answer a betting-trend question using historical game data.

        Routes to the most relevant BettingTrends method based on keywords
        in the question, then returns the result alongside a plain summary.
        """
        trends = self._load_trends()

        if trends is None:
            return _degraded(
                "answer_betting_trend",
                "No modeling table found. Run `python scripts/build_features.py` first.",
                question=question,
                trend_type=None,
                result={},
            )

        q = question.lower()

        # ── Route to the most relevant function ────────────────────────────────
        trend_type, result = self._route_trend(q, trends)

        # Strip non-serialisable objects (DataFrames → list-of-dicts)
        serialised = self._serialise_trend(result)

        summary = self._trend_summary(trend_type, serialised.get("summary", {}))

        return _ok(
            "answer_betting_trend",
            question=question,
            trend_type=trend_type,
            result=serialised,
            plain_summary=summary,
        )

    def _route_trend(
        self, q: str, trends
    ) -> tuple[str, dict]:
        """Return (trend_type, raw_result_dict) for the question."""

        # Team-specific queries
        team = self._extract_team_from_text(q)

        if team and re.search(r"\bover\b", q) and not re.search(r"\bcover\b", q):
            return "team_over_rate", trends.team_over_rate(team)

        if team:
            return "team_cover_rate", trends.team_cover_rate(team)

        # Trend-type routing
        if re.search(r"\broad\s+underdog\b", q):
            return "road_underdog_cover_rate", trends.road_underdog_cover_rate()

        if re.search(r"\bhome\s+favorite\b", q):
            return "home_favorite_cover_rate", trends.home_favorite_cover_rate()

        if re.search(r"\bunderdog\b", q):
            return "underdog_cover_rate", trends.underdog_cover_rate()

        if re.search(r"\bfavorite\b", q) and re.search(r"\bcover\b", q):
            return "favorite_cover_rate", trends.favorite_cover_rate()

        if re.search(r"\b(over|total|o\/u)\b", q) and not re.search(r"\bcover\b", q):
            return "over_under_hit_rate", trends.over_under_hit_rate()

        if re.search(r"\bspread\b.{0,20}\b(greater|more|larger|bigger)\s+than\b", q):
            return "spread_bucket_analysis", trends.spread_bucket_analysis()

        if re.search(r"\bspread\s+bucket\b|\bspread\s+size\b", q):
            return "spread_bucket_analysis", trends.spread_bucket_analysis()

        if re.search(r"\bmodel\b.{0,20}\b(edge|outperform|random|baseline)\b", q):
            return "model_edge_analysis", trends.model_edge_analysis()

        if re.search(r"\ball\s+teams?\b|\bmost\s+(often|frequently)\b|\branking\b", q):
            if re.search(r"\bover\b", q):
                df = trends.all_teams_over_rate()
                return "all_teams_over_rate", {
                    "summary": {"note": "Teams ranked by over/under rate"},
                    "table": df,
                }
            df = trends.all_teams_cover_rate()
            return "all_teams_cover_rate", {
                "summary": {"note": "Teams ranked by ATS cover rate"},
                "table": df,
            }

        # Default: favorite cover rate overview
        return "favorite_cover_rate", trends.favorite_cover_rate()

    @staticmethod
    def _extract_team_from_text(text: str) -> str | None:
        """Return the first team abbreviation found in *text*."""
        for alias, abbr in _TEAM_MAP_SORTED:
            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            if re.search(pattern, text):
                return abbr
        return None

    @staticmethod
    def _serialise_trend(result: dict) -> dict:
        """Convert DataFrames to list-of-dicts and scrub NaN for JSON safety."""
        out: dict = {}
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                out[k] = [
                    {col: _safe(val) for col, val in row.items()}
                    for row in v.to_dict("records")
                ]
            elif isinstance(v, dict):
                out[k] = _safe_dict(v)
            elif isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = _safe(v)
        return out

    @staticmethod
    def _trend_summary(trend_type: str, summary: dict) -> str:
        """Build a one-sentence plain summary from the BettingTrends result."""
        def pct(v):
            return f"{v:.1%}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"

        if trend_type == "team_cover_rate":
            team = summary.get("team", "this team")
            rate = pct(summary.get("overall_cover_rate"))
            return f"{team} has covered the spread {rate} of the time."

        if trend_type == "team_over_rate":
            team = summary.get("team", "this team")
            rate = pct(summary.get("overall_over_rate"))
            return f"{team}'s games have gone over the total {rate} of the time."

        if trend_type == "favorite_cover_rate":
            rate = pct(summary.get("favorites_cover_rate"))
            return f"Favorites cover the spread {rate} of the time overall."

        if trend_type == "underdog_cover_rate":
            rate = pct(summary.get("underdogs_cover_rate"))
            return f"Underdogs cover the spread {rate} of the time overall."

        if trend_type == "home_favorite_cover_rate":
            rate = pct(summary.get("home_favorite_cover_rate"))
            return f"Home favorites cover the spread {rate} of the time."

        if trend_type == "road_underdog_cover_rate":
            rate = pct(summary.get("road_underdog_cover_rate"))
            return f"Road underdogs cover the spread {rate} of the time."

        if trend_type == "over_under_hit_rate":
            rate = pct(summary.get("overall_over_rate"))
            return f"Games go over the total {rate} of the time."

        if trend_type == "spread_bucket_analysis":
            return "Cover and over rates by spread size are shown in the table."

        if trend_type == "model_edge_analysis":
            vs = summary.get("win_vs_random")
            if vs is not None:
                return (
                    f"The win model outperforms random guessing by "
                    f"{vs:+.1%} on the held-out test season."
                )
            return "Model vs baseline comparison is shown in the table."

        return summary.get("note", "Betting trend analysis complete.")

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 4: summarize_team
    # ══════════════════════════════════════════════════════════════════════════

    def summarize_team(
        self,
        team: str,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """
        Return season record, scoring averages, and recent form for a team.

        Queries the database when available; falls back to the modeling table.
        """
        abbr = _canon_team(team) or team.upper()
        season_list = (
            [seasons] if isinstance(seasons, int)
            else (list(seasons) if seasons else None)
        )

        summary = self._summarize_from_db(abbr, season_list)
        if summary is None:
            summary = self._summarize_from_df(abbr, season_list)
        if summary is None:
            return _degraded(
                "summarize_team",
                f"No data found for {abbr}. Ingest game data first.",
                team=abbr,
                seasons=season_list,
            )

        return _ok(
            "summarize_team",
            team=abbr,
            seasons=season_list or summary.get("seasons_in_data"),
            summary=_safe_dict(summary),
        )

    def _summarize_from_db(
        self, team: str, seasons: list[int] | None
    ) -> dict | None:
        engine = self._load_db_engine()
        if engine is None:
            return None
        try:
            return self._query_team_summary(engine, team, seasons)
        except Exception as exc:
            logger.debug("DB team summary failed for %s: %s", team, exc)
            return None

    @staticmethod
    def _query_team_summary(engine, team: str, seasons: list[int] | None) -> dict | None:
        """Run SQL queries to build a team summary dict."""
        from sqlalchemy import text

        season_filter = ""
        params: dict = {"team": team}
        if seasons:
            placeholders = ", ".join(f":s{i}" for i in range(len(seasons)))
            season_filter = f"AND season IN ({placeholders})"
            for i, s in enumerate(seasons):
                params[f"s{i}"] = s

        games_sql = text(f"""
            SELECT
                season,
                SUM(CASE WHEN (home_team = :team AND home_score > away_score)
                              OR (away_team = :team AND away_score > home_score)
                         THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN (home_team = :team AND home_score < away_score)
                              OR (away_team = :team AND away_score < home_score)
                         THEN 1 ELSE 0 END) AS losses,
                COUNT(*) AS games,
                AVG(CASE WHEN home_team = :team THEN home_score
                         WHEN away_team = :team THEN away_score END) AS avg_pts_scored,
                AVG(CASE WHEN home_team = :team THEN away_score
                         WHEN away_team = :team THEN home_score END) AS avg_pts_allowed
            FROM games
            WHERE (home_team = :team OR away_team = :team)
              AND home_score IS NOT NULL
              {season_filter}
            GROUP BY season
            ORDER BY season DESC
        """)

        with engine.connect() as conn:
            rows = conn.execute(games_sql, params).fetchall()

        if not rows:
            return None

        season_records = [
            {
                "season": r[0],
                "wins": r[1],
                "losses": r[2],
                "games": r[3],
                "win_pct": round(r[1] / r[3], 3) if r[3] else None,
                "avg_pts_scored": round(r[4], 1) if r[4] is not None else None,
                "avg_pts_allowed": round(r[5], 1) if r[5] is not None else None,
                "pt_diff_per_game": (
                    round(r[4] - r[5], 1)
                    if r[4] is not None and r[5] is not None
                    else None
                ),
            }
            for r in rows
        ]

        # Aggregate across all returned seasons
        total_wins = sum(r["wins"] for r in season_records)
        total_losses = sum(r["losses"] for r in season_records)
        total_games = sum(r["games"] for r in season_records)
        avg_scored = (
            round(sum(r["avg_pts_scored"] for r in season_records
                      if r["avg_pts_scored"] is not None) / len(season_records), 1)
            if any(r["avg_pts_scored"] is not None for r in season_records)
            else None
        )
        avg_allowed = (
            round(sum(r["avg_pts_allowed"] for r in season_records
                      if r["avg_pts_allowed"] is not None) / len(season_records), 1)
            if any(r["avg_pts_allowed"] is not None for r in season_records)
            else None
        )

        return {
            "team": team,
            "seasons_in_data": [r["season"] for r in season_records],
            "overall": {
                "wins": total_wins,
                "losses": total_losses,
                "games": total_games,
                "win_pct": round(total_wins / total_games, 3) if total_games else None,
                "avg_pts_scored": avg_scored,
                "avg_pts_allowed": avg_allowed,
                "avg_pt_diff": (
                    round(avg_scored - avg_allowed, 1)
                    if avg_scored is not None and avg_allowed is not None
                    else None
                ),
            },
            "by_season": season_records,
        }

    def _summarize_from_df(
        self, team: str, seasons: list[int] | None
    ) -> dict | None:
        """Build a summary from the modeling_table CSV when DB is unavailable."""
        df = self._load_modeling_df()
        if df is None or df.empty:
            return None

        mask = (df.get("home_team", pd.Series(dtype=str)) == team) | \
               (df.get("away_team", pd.Series(dtype=str)) == team)
        sub = df[mask].copy()
        if sub.empty:
            return None

        if seasons:
            sub = sub[sub["season"].isin(seasons)]
        if sub.empty:
            return None

        # Compute wins
        home = sub[sub["home_team"] == team]
        away = sub[sub["away_team"] == team]
        wins = int((home["home_win"] == 1).sum()) + int((away["home_win"] == 0).sum())
        losses = int((home["home_win"] == 0).sum()) + int((away["home_win"] == 1).sum())
        games = len(sub)

        # Scoring averages
        scored = pd.concat([
            home.get("home_score", pd.Series(dtype=float)),
            away.get("away_score", pd.Series(dtype=float)),
        ]).dropna()
        allowed = pd.concat([
            home.get("away_score", pd.Series(dtype=float)),
            away.get("home_score", pd.Series(dtype=float)),
        ]).dropna()

        return {
            "team": team,
            "seasons_in_data": sorted(sub["season"].dropna().unique().astype(int).tolist()),
            "overall": {
                "wins": wins,
                "losses": losses,
                "games": games,
                "win_pct": round(wins / games, 3) if games else None,
                "avg_pts_scored": round(float(scored.mean()), 1) if len(scored) else None,
                "avg_pts_allowed": round(float(allowed.mean()), 1) if len(allowed) else None,
                "avg_pt_diff": (
                    round(float(scored.mean()) - float(allowed.mean()), 1)
                    if len(scored) and len(allowed)
                    else None
                ),
            },
            "by_season": [],
            "source": "modeling_table_csv",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 5: explain_last_prediction
    # ══════════════════════════════════════════════════════════════════════════

    def explain_last_prediction(self) -> dict[str, Any]:
        """
        Return the explanation for the most recent ``predict_matchup`` call.

        If no prediction has been made this session, returns a clear message.
        """
        if self._last_prediction is None:
            return _degraded(
                "explain_last_prediction",
                "No prediction has been made yet. "
                "Ask me to predict a matchup first, then I can explain it.",
                has_prediction=False,
                explanation=None,
            )

        pred = self._last_prediction
        explanation = pred.get("explanation", {})
        raw = pred.get("prediction", {})

        return _ok(
            "explain_last_prediction",
            has_prediction=True,
            home_team=pred.get("home_team"),
            away_team=pred.get("away_team"),
            prediction_summary={
                "home_win_prob": raw.get("home_win_prob"),
                "away_win_prob": raw.get("away_win_prob"),
                "predicted_winner": raw.get("predicted_winner"),
                "confidence": raw.get("confidence"),
            },
            explanation=explanation,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 6: get_model_metrics
    # ══════════════════════════════════════════════════════════════════════════

    def get_model_metrics(self) -> dict[str, Any]:
        """
        Load and return saved training metrics for the win and spread models.

        Reads from ``model_metrics.json`` and ``spread_model_metrics.json``
        in the model directory.
        """
        def _load(path: Path, label: str) -> dict:
            if not path.exists():
                return {"available": False, "path": str(path)}
            try:
                doc = json.loads(path.read_text())
                best = doc.get("best_model", "")
                m = doc.get("models", {}).get(best, {})
                split = doc.get("split", {})
                return {
                    "available": True,
                    "model_name": best,
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "roc_auc": m.get("roc_auc"),
                    "brier_score": m.get("brier_score"),
                    "cv_roc_auc": m.get("cv_roc_auc"),
                    "test_season": split.get("test_season"),
                    "train_rows": split.get("train_rows"),
                    "test_rows": split.get("test_rows"),
                    "trained_at": doc.get("trained_at"),
                    "path": str(path),
                }
            except Exception as exc:
                return {"available": False, "error": str(exc), "path": str(path)}

        win = _load(self._model_dir / "model_metrics.json", "win")
        spread = _load(self._model_dir / "spread_model_metrics.json", "spread")

        if not win["available"] and not spread["available"]:
            return _error(
                "get_model_metrics",
                f"No model metrics found in {self._model_dir}. "
                "Run `python scripts/train_model.py` first.",
                win_model=win,
                spread_model=spread,
            )

        # Provide a short interpretation for each model
        def _interpret_win(m: dict) -> str:
            acc = m.get("accuracy")
            if acc is None:
                return "No accuracy data."
            pct = int(round(acc * 100))
            if acc >= 0.70:
                return f"{pct}% accuracy — strong predictive performance for NFL."
            if acc >= 0.60:
                return f"{pct}% accuracy — moderate performance, typical for NFL prediction."
            return f"{pct}% accuracy — weak; model may need more data."

        def _interpret_spread(m: dict) -> str:
            acc = m.get("accuracy")
            if acc is None:
                return "No accuracy data."
            pct = int(round(acc * 100))
            if acc >= 0.55:
                return f"{pct}% ATS accuracy — good edge above market noise."
            if acc >= 0.52:
                return f"{pct}% ATS accuracy — modest edge, consistent with a well-calibrated model."
            return f"{pct}% ATS accuracy — near-random (expected; the spread is designed to be 50/50)."

        if win.get("available"):
            win["interpretation"] = _interpret_win(win)
        if spread.get("available"):
            spread["interpretation"] = _interpret_spread(spread)

        return _ok(
            "get_model_metrics",
            win_model=win,
            spread_model=spread,
            model_dir=str(self._model_dir),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tool 7: get_available_data_summary
    # ══════════════════════════════════════════════════════════════════════════

    def get_available_data_summary(self) -> dict[str, Any]:
        """
        Report what data and model artifacts are available.

        Returns counts and season ranges for each database table, plus
        which model files exist in the model directory.
        """
        db_info = self._db_summary()
        model_info = self._model_artifact_summary()
        csv_info = self._csv_summary()

        all_unavailable = (
            not db_info.get("connected")
            and not model_info.get("any_available")
            and not csv_info.get("modeling_table_available")
        )

        status = "error" if all_unavailable else "ok"
        error_msg = (
            "No data or models found. "
            "Run the ingestion pipeline then train models before using the chatbot."
            if all_unavailable else None
        )

        return {
            "tool": "get_available_data_summary",
            "status": status,
            "error": error_msg,
            "database": db_info,
            "model_artifacts": model_info,
            "csv_files": csv_info,
            "recommendations": self._recommendations(db_info, model_info, csv_info),
        }

    def _db_summary(self) -> dict:
        engine = self._load_db_engine()
        if engine is None:
            return {"connected": False, "error": "No database connection"}

        tables = {
            "games": "SELECT COUNT(*), MIN(season), MAX(season) FROM games WHERE home_score IS NOT NULL",
            "team_game_stats": "SELECT COUNT(*), MIN(season), MAX(season) FROM team_game_stats",
            "player_game_stats": "SELECT COUNT(*), MIN(season), MAX(season) FROM player_game_stats",
            "injuries": "SELECT COUNT(*), MIN(season), MAX(season) FROM injuries",
            "odds": "SELECT COUNT(*) FROM odds",
        }

        info: dict = {"connected": True}
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                for table, sql in tables.items():
                    try:
                        row = conn.execute(text(sql)).fetchone()
                        if row and len(row) >= 3:
                            info[table] = {
                                "count": row[0],
                                "min_season": row[1],
                                "max_season": row[2],
                            }
                        elif row:
                            info[table] = {"count": row[0]}
                    except Exception:
                        info[table] = {"count": 0, "error": "table missing or empty"}
        except Exception as exc:
            info["error"] = str(exc)

        return info

    def _model_artifact_summary(self) -> dict:
        d = self._model_dir
        files = {
            "win_model": "best_model.joblib",
            "win_feature_columns": "feature_columns.json",
            "win_metrics": "model_metrics.json",
            "win_feature_importance": "feature_importance.csv",
            "spread_model": "spread_model.joblib",
            "spread_metrics": "spread_model_metrics.json",
            "spread_feature_importance": "spread_feature_importance.csv",
        }
        info: dict = {}
        for key, fname in files.items():
            p = d / fname
            info[key] = {"available": p.exists(), "path": str(p)}

        info["any_available"] = any(v["available"] for v in info.values() if isinstance(v, dict))
        return info

    def _csv_summary(self) -> dict:
        table_csv = self._data_dir / "modeling_table.csv"
        info: dict = {
            "modeling_table_available": table_csv.exists(),
            "modeling_table_path": str(table_csv),
        }
        if table_csv.exists():
            df = self._load_modeling_df()
            if df is not None:
                info["modeling_table_rows"] = len(df)
                if "season" in df.columns:
                    info["seasons"] = sorted(df["season"].dropna().unique().astype(int).tolist())
                if "home_team" in df.columns or "away_team" in df.columns:
                    teams = set(df.get("home_team", pd.Series()).dropna()) | \
                            set(df.get("away_team", pd.Series()).dropna())
                    info["teams"] = sorted(teams)
        return info

    @staticmethod
    def _recommendations(db: dict, models: dict, csvs: dict) -> list[str]:
        recs: list[str] = []
        games = db.get("games", {})
        if not db.get("connected") or games.get("count", 0) == 0:
            recs.append("Run `python scripts/ingest.py` to load game data into the database.")
        if not csvs.get("modeling_table_available"):
            recs.append("Run `python scripts/build_features.py` to build the modeling table.")
        if not models.get("win_model", {}).get("available"):
            recs.append("Run `python scripts/train_model.py --model win` to train the win model.")
        if not models.get("spread_model", {}).get("available"):
            recs.append("Run `python scripts/train_model.py --model spread` to train the ATS model.")
        if not recs:
            recs.append("All systems ready. The chatbot can answer all question types.")
        return recs


# ══════════════════════════════════════════════════════════════════════════════
# Module-level convenience functions (use a shared default ChatbotTools instance)
# ══════════════════════════════════════════════════════════════════════════════

_default: ChatbotTools | None = None


def _get_default() -> ChatbotTools:
    global _default
    if _default is None:
        _default = ChatbotTools()
    return _default


def predict_matchup(
    home_team: str,
    away_team: str,
    season: int | None = None,
    week: int | None = None,
) -> dict[str, Any]:
    """Predict who wins a game. See ChatbotTools.predict_matchup."""
    return _get_default().predict_matchup(home_team, away_team, season, week)


def compare_players(
    player_a: str,
    player_b: str,
    seasons: list[int] | int | None = None,
) -> dict[str, Any]:
    """Compare two players side by side. See ChatbotTools.compare_players."""
    return _get_default().compare_players(player_a, player_b, seasons)


def answer_betting_trend(question: str) -> dict[str, Any]:
    """Answer a betting-trend question. See ChatbotTools.answer_betting_trend."""
    return _get_default().answer_betting_trend(question)


def summarize_team(
    team: str,
    seasons: list[int] | int | None = None,
) -> dict[str, Any]:
    """Return season record and stats for a team. See ChatbotTools.summarize_team."""
    return _get_default().summarize_team(team, seasons)


def explain_last_prediction() -> dict[str, Any]:
    """Explain the most recent prediction. See ChatbotTools.explain_last_prediction."""
    return _get_default().explain_last_prediction()


def get_model_metrics() -> dict[str, Any]:
    """Return saved model training metrics. See ChatbotTools.get_model_metrics."""
    return _get_default().get_model_metrics()


def get_available_data_summary() -> dict[str, Any]:
    """Report available data and model artifacts. See ChatbotTools.get_available_data_summary."""
    return _get_default().get_available_data_summary()
