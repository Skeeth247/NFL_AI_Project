"""
Demo mode loader for the NFL AI Chatbot.

Bootstraps a fully functional in-memory environment from the small sample
datasets in data/demo/ so the Streamlit app works without any ingestion step.

Public API
----------
    from nfl_chatbot.demo.loader import load_demo_env

    env = load_demo_env()        # cached after first call
    env.engine                   # SQLAlchemy in-memory engine
    env.model_dir                # pathlib.Path to temp dir with model artifacts
    env.data_dir                 # pathlib.Path to temp dir with modeling_table.csv
    env.tools                    # ready-to-use ChatbotTools instance
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import sqlalchemy

logger = logging.getLogger(__name__)

_DEMO_DIR = Path(__file__).resolve().parents[3] / "data" / "demo"


@dataclass
class DemoEnv:
    engine: sqlalchemy.Engine
    model_dir: Path
    data_dir: Path
    tools: object          # ChatbotTools — typed as object to avoid circular import


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_games(engine: sqlalchemy.Engine) -> pd.DataFrame:
    df = pd.read_csv(_DEMO_DIR / "games.csv")
    # Keep gameday as plain ISO date string so SQLite stores it as TEXT,
    # which FeatureBuilder's pd.to_datetime(..., errors="coerce") handles cleanly.
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["location"] = "Home"
    df["created_at"] = None
    df["updated_at"] = None
    df.to_sql("games", con=engine, if_exists="replace", index=False)
    logger.info("Demo: loaded %d games", len(df))
    return df


def _load_player_stats(engine: sqlalchemy.Engine) -> None:
    df = pd.read_csv(_DEMO_DIR / "player_stats.csv")
    df["created_at"] = None
    df["updated_at"] = None
    df.to_sql("player_season_stats", con=engine, if_exists="replace", index=True, index_label="id")
    logger.info("Demo: loaded %d player season stat rows", len(df))


def _load_injuries(engine: sqlalchemy.Engine) -> None:
    df = pd.read_csv(_DEMO_DIR / "injuries.csv")
    df["created_at"] = None
    df["updated_at"] = None
    # ORM schema expects an integer PK column named "id"
    df.to_sql("injuries", con=engine, if_exists="replace", index=True, index_label="id")
    logger.info("Demo: loaded %d injury rows", len(df))


def _load_odds(engine: sqlalchemy.Engine) -> None:
    df = pd.read_csv(_DEMO_DIR / "odds.csv")
    df["created_at"] = None
    df["updated_at"] = None
    df.to_sql("odds", con=engine, if_exists="replace", index=False)
    logger.info("Demo: loaded %d odds rows", len(df))


def _build_features(engine: sqlalchemy.Engine) -> pd.DataFrame:
    """Run FeatureBuilder on demo games; fall back to a minimal hand-crafted table."""
    try:
        from nfl_chatbot.features.team_features import EloConfig, FeatureBuilder
        builder = FeatureBuilder(engine, window=3, elo=EloConfig())
        table = builder.build()
        if not table.empty:
            logger.info("Demo: FeatureBuilder produced %d rows", len(table))
            return table
    except Exception as exc:
        logger.warning("Demo: FeatureBuilder failed (%s) — using fallback table", exc)

    # Fallback: build a minimal modeling table directly from games
    games = pd.read_sql("SELECT * FROM games", con=engine)
    games = games[games["home_win"].notna()].copy()
    games["spread_line"] = games["spread_line"].fillna(0.0)
    games["total_line"] = games["total_line"].fillna(47.0)
    games["home_rest_days"] = 7
    games["away_rest_days"] = 7
    games["rest_diff"] = 0
    games["elo_diff"] = -games["spread_line"] * 5
    games["implied_home_win_prob"] = 1 / (1 + 10 ** (games["spread_line"].fillna(0) / 20))
    for col in ["home_wins_last4","away_wins_last4"]:
        games[col] = 2.0
    for col in ["home_pts_scored_avg4","away_pts_scored_avg4"]:
        games[col] = 24.0
    for col in ["home_pts_allowed_avg4","away_pts_allowed_avg4"]:
        games[col] = 22.0
    for col in ["home_yds_per_play_avg4","away_yds_per_play_avg4"]:
        games[col] = 5.5
    for col in ["home_turnover_diff_avg4","away_turnover_diff_avg4"]:
        games[col] = 0.0
    games["div_game"] = 0
    games["neutral_site"] = 0
    games["h2h_home_win_rate"] = 0.5
    games["h2h_home_pt_diff"] = 0.0
    games["h2h_all_time_home_win_rate"] = 0.5
    games["home_covered"] = (
        (games["home_score"] - games["away_score"] + games["spread_line"]) > 0
    ).astype(int)
    logger.info("Demo: fallback table has %d rows", len(games))
    return games


def _train_demo_model(table: pd.DataFrame, model_dir: Path) -> None:
    """Train a minimal win-prediction model on demo data and save artifacts."""
    from nfl_chatbot.models.train import ModelTrainer

    # Use only the test season that exists in the data so the split works
    seasons = sorted(table["season"].dropna().unique().astype(int).tolist())
    test_season = seasons[-1] if len(seasons) >= 2 else seasons[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = ModelTrainer(test_season=test_season)
        try:
            result = trainer.train(table)
            trainer.save(result, output_dir=model_dir)
            logger.info(
                "Demo: trained %s  acc=%.3f",
                result.best_model_name,
                result.metrics[result.best_model_name]["accuracy"],
            )
        except Exception as exc:
            logger.warning("Demo: model training failed (%s) — predictions unavailable", exc)


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_demo_env() -> DemoEnv:
    """
    Build a complete demo environment from the CSV fixtures in data/demo/.
    Results are cached in process memory — only runs once per interpreter session.
    """
    logger.info("Demo mode: bootstrapping environment from %s", _DEMO_DIR)

    if not _DEMO_DIR.exists():
        raise FileNotFoundError(
            f"Demo data directory not found: {_DEMO_DIR}\n"
            "Run from the project root or verify data/demo/ exists."
        )

    # In-memory SQLite — fast, zero setup, disappears when the process exits
    engine = sqlalchemy.create_engine("sqlite:///:memory:", echo=False)

    _load_games(engine)
    _load_player_stats(engine)
    _load_injuries(engine)
    _load_odds(engine)

    table = _build_features(engine)

    # Temp dirs survive for the lifetime of the process (Streamlit cache keeps
    # the DemoEnv alive, so the dirs won't be cleaned up mid-session)
    tmp = tempfile.mkdtemp(prefix="nfl_demo_")
    model_dir = Path(tmp) / "models"
    data_dir  = Path(tmp) / "data"
    model_dir.mkdir()
    data_dir.mkdir()

    csv_path = data_dir / "modeling_table.csv"
    table.to_csv(csv_path, index=False)

    _train_demo_model(table, model_dir)

    # Wire up ChatbotTools with the demo engine and model artifacts
    from nfl_chatbot.chatbot.tools import ChatbotTools
    tools = ChatbotTools(
        engine=engine,
        model_dir=str(model_dir),
        data_dir=str(data_dir),
    )

    logger.info("Demo mode: environment ready")
    return DemoEnv(engine=engine, model_dir=model_dir, data_dir=data_dir, tools=tools)


def is_demo_mode() -> bool:
    """
    Return True when the app should run in demo mode.

    Rules (first match wins):
      1. Environment variable NFL_DEMO_MODE=1  → always demo
      2. Environment variable NFL_DEMO_MODE=0  → never demo
      3. Real database has 0 games            → auto demo
    """
    import os
    flag = os.environ.get("NFL_DEMO_MODE", "").strip()
    if flag == "1":
        return True
    if flag == "0":
        return False

    # Auto-detect: check whether the real DB has any games
    try:
        from nfl_chatbot.data.database import get_engine
        real_engine = get_engine()
        with real_engine.connect() as conn:
            n = conn.execute(
                sqlalchemy.text("SELECT COUNT(*) FROM games")
            ).scalar()
            return int(n or 0) == 0
    except Exception:
        return True
