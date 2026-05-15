"""
SQLAlchemy ORM table definitions for the NFL AI Chatbot database.

All tables use SQLite by default. The schema covers raw ingested data
(games, teams, stats, injuries, odds), derived ML inputs (engineered_features),
model outputs (model_predictions), and application logs (chatbot_logs).
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── teams ──────────────────────────────────────────────────────────────────────


class Team(Base):
    """One row per NFL franchise. Uses the nflverse three-letter abbreviation as PK."""

    __tablename__ = "teams"

    team_abbr: Mapped[str] = mapped_column(String(10), primary_key=True)
    team_name: Mapped[str] = mapped_column(String(100))
    team_nick: Mapped[str] = mapped_column(String(50))
    team_conf: Mapped[str] = mapped_column(String(10))
    team_division: Mapped[str] = mapped_column(String(30))
    team_color: Mapped[Optional[str]] = mapped_column(String(20))
    team_logo_url: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── games ──────────────────────────────────────────────────────────────────────


class Game(Base):
    """One row per NFL game. game_id follows nflverse format: {season}_{week}_{away}_{home}."""

    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer, index=True)
    game_type: Mapped[str] = mapped_column(String(10))  # REG, WC, DIV, CON, SB
    home_team: Mapped[str] = mapped_column(String(10), ForeignKey("teams.team_abbr"), index=True)
    away_team: Mapped[str] = mapped_column(String(10), ForeignKey("teams.team_abbr"), index=True)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_win: Mapped[Optional[bool]] = mapped_column(Boolean)  # NULL = not yet played
    overtime: Mapped[Optional[bool]] = mapped_column(Boolean)
    neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)
    div_game: Mapped[Optional[bool]] = mapped_column(Boolean)
    roof: Mapped[Optional[str]] = mapped_column(String(20))    # dome, outdoors, retractable
    surface: Mapped[Optional[str]] = mapped_column(String(30))
    stadium: Mapped[Optional[str]] = mapped_column(String(100))
    gameday: Mapped[Optional[datetime.date]] = mapped_column(Date)
    gametime: Mapped[Optional[str]] = mapped_column(String(10))
    spread_line: Mapped[Optional[float]] = mapped_column(Float)  # negative = home favored
    total_line: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── team_game_stats ────────────────────────────────────────────────────────────


class TeamGameStats(Base):
    """Per-game team performance stats. One row per (game, team)."""

    __tablename__ = "team_game_stats"
    __table_args__ = (UniqueConstraint("game_id", "team", name="uq_team_game"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("games.game_id"), index=True)
    team: Mapped[str] = mapped_column(String(10), ForeignKey("teams.team_abbr"), index=True)
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    is_home: Mapped[bool] = mapped_column(Boolean)
    points_scored: Mapped[Optional[int]] = mapped_column(Integer)
    points_allowed: Mapped[Optional[int]] = mapped_column(Integer)
    total_yards: Mapped[Optional[float]] = mapped_column(Float)
    passing_yards: Mapped[Optional[float]] = mapped_column(Float)
    rushing_yards: Mapped[Optional[float]] = mapped_column(Float)
    turnovers: Mapped[Optional[int]] = mapped_column(Integer)
    penalties: Mapped[Optional[int]] = mapped_column(Integer)
    penalty_yards: Mapped[Optional[int]] = mapped_column(Integer)
    time_of_possession: Mapped[Optional[float]] = mapped_column(Float)  # minutes
    third_down_pct: Mapped[Optional[float]] = mapped_column(Float)
    red_zone_pct: Mapped[Optional[float]] = mapped_column(Float)
    first_downs: Mapped[Optional[int]] = mapped_column(Integer)
    sacks_allowed: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── player_game_stats ──────────────────────────────────────────────────────────


class PlayerGameStats(Base):
    """Per-game player stats. One row per (game, player)."""

    __tablename__ = "player_game_stats"
    __table_args__ = (UniqueConstraint("game_id", "player_id", name="uq_player_game"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("games.game_id"), index=True)
    player_id: Mapped[str] = mapped_column(String(50), index=True)
    player_name: Mapped[str] = mapped_column(String(100))
    team: Mapped[str] = mapped_column(String(10), ForeignKey("teams.team_abbr"), index=True)
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    position: Mapped[Optional[str]] = mapped_column(String(10))
    # Passing
    completions: Mapped[Optional[int]] = mapped_column(Integer)
    attempts: Mapped[Optional[int]] = mapped_column(Integer)
    passing_yards: Mapped[Optional[float]] = mapped_column(Float)
    passing_tds: Mapped[Optional[int]] = mapped_column(Integer)
    interceptions: Mapped[Optional[int]] = mapped_column(Integer)
    passer_rating: Mapped[Optional[float]] = mapped_column(Float)
    # Rushing
    carries: Mapped[Optional[int]] = mapped_column(Integer)
    rushing_yards: Mapped[Optional[float]] = mapped_column(Float)
    rushing_tds: Mapped[Optional[int]] = mapped_column(Integer)
    # Receiving
    targets: Mapped[Optional[int]] = mapped_column(Integer)
    receptions: Mapped[Optional[int]] = mapped_column(Integer)
    receiving_yards: Mapped[Optional[float]] = mapped_column(Float)
    receiving_tds: Mapped[Optional[int]] = mapped_column(Integer)
    # Fantasy
    fantasy_points: Mapped[Optional[float]] = mapped_column(Float)
    fantasy_points_ppr: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── player_season_stats ────────────────────────────────────────────────────────


class PlayerSeasonStats(Base):
    """Aggregated season totals per player. One row per (player, season)."""

    __tablename__ = "player_season_stats"
    __table_args__ = (UniqueConstraint("player_id", "season", name="uq_player_season"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[str] = mapped_column(String(50), index=True)
    player_name: Mapped[str] = mapped_column(String(100))
    team: Mapped[str] = mapped_column(String(10))
    season: Mapped[int] = mapped_column(Integer, index=True)
    position: Mapped[Optional[str]] = mapped_column(String(10))
    games_played: Mapped[Optional[int]] = mapped_column(Integer)
    # Passing
    passing_yards: Mapped[Optional[float]] = mapped_column(Float)
    passing_tds: Mapped[Optional[int]] = mapped_column(Integer)
    interceptions: Mapped[Optional[int]] = mapped_column(Integer)
    completion_pct: Mapped[Optional[float]] = mapped_column(Float)
    passer_rating: Mapped[Optional[float]] = mapped_column(Float)
    # Rushing
    carries: Mapped[Optional[int]] = mapped_column(Integer)
    rushing_yards: Mapped[Optional[float]] = mapped_column(Float)
    rushing_tds: Mapped[Optional[int]] = mapped_column(Integer)
    yards_per_carry: Mapped[Optional[float]] = mapped_column(Float)
    # Receiving
    targets: Mapped[Optional[int]] = mapped_column(Integer)
    receptions: Mapped[Optional[int]] = mapped_column(Integer)
    receiving_yards: Mapped[Optional[float]] = mapped_column(Float)
    receiving_tds: Mapped[Optional[int]] = mapped_column(Integer)
    yards_per_reception: Mapped[Optional[float]] = mapped_column(Float)
    # Fantasy
    fantasy_points_total: Mapped[Optional[float]] = mapped_column(Float)
    fantasy_points_per_game: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── injuries ───────────────────────────────────────────────────────────────────


class Injury(Base):
    """Weekly injury report entries. One row per (season, week, player)."""

    __tablename__ = "injuries"
    __table_args__ = (UniqueConstraint("season", "week", "player_id", name="uq_injury_week"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer, index=True)
    team: Mapped[str] = mapped_column(String(10), ForeignKey("teams.team_abbr"), index=True)
    player_id: Mapped[str] = mapped_column(String(50), index=True)
    player_name: Mapped[str] = mapped_column(String(100))
    position: Mapped[Optional[str]] = mapped_column(String(10))
    report_primary_injury: Mapped[Optional[str]] = mapped_column(String(100))
    report_secondary_injury: Mapped[Optional[str]] = mapped_column(String(100))
    practice_status: Mapped[Optional[str]] = mapped_column(String(10))  # FP, LP, DNP
    game_status: Mapped[Optional[str]] = mapped_column(String(20))      # Out, Doubtful, Q, IR
    date_modified: Mapped[Optional[datetime.date]] = mapped_column(Date)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── odds ───────────────────────────────────────────────────────────────────────


class Odds(Base):
    """Betting lines and closing odds. One row per (game, bookmaker source)."""

    __tablename__ = "odds"
    __table_args__ = (UniqueConstraint("game_id", "source", name="uq_odds_source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("games.game_id"), index=True)
    season: Mapped[int] = mapped_column(Integer)
    week: Mapped[int] = mapped_column(Integer)
    home_team: Mapped[str] = mapped_column(String(10))
    away_team: Mapped[str] = mapped_column(String(10))
    spread_home: Mapped[Optional[float]] = mapped_column(Float)  # negative = home favored
    spread_away: Mapped[Optional[float]] = mapped_column(Float)
    total_line: Mapped[Optional[float]] = mapped_column(Float)
    home_ml: Mapped[Optional[int]] = mapped_column(Integer)      # American moneyline
    away_ml: Mapped[Optional[int]] = mapped_column(Integer)
    home_covered: Mapped[Optional[bool]] = mapped_column(Boolean)
    away_covered: Mapped[Optional[bool]] = mapped_column(Boolean)
    over_hit: Mapped[Optional[bool]] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(50))              # bookmaker name
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── engineered_features ────────────────────────────────────────────────────────


class EngineeredFeatures(Base):
    """Pre-computed ML feature row per game. One row per game_id."""

    __tablename__ = "engineered_features"

    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("games.game_id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer)
    home_team: Mapped[str] = mapped_column(String(10))
    away_team: Mapped[str] = mapped_column(String(10))
    # Elo
    home_elo: Mapped[Optional[float]] = mapped_column(Float)
    away_elo: Mapped[Optional[float]] = mapped_column(Float)
    elo_diff: Mapped[Optional[float]] = mapped_column(Float)     # home - away
    # Rolling form (last N games)
    home_wins_last4: Mapped[Optional[int]] = mapped_column(Integer)
    away_wins_last4: Mapped[Optional[int]] = mapped_column(Integer)
    home_pts_scored_avg4: Mapped[Optional[float]] = mapped_column(Float)
    away_pts_scored_avg4: Mapped[Optional[float]] = mapped_column(Float)
    home_pts_allowed_avg4: Mapped[Optional[float]] = mapped_column(Float)
    away_pts_allowed_avg4: Mapped[Optional[float]] = mapped_column(Float)
    home_yds_per_play_avg4: Mapped[Optional[float]] = mapped_column(Float)
    away_yds_per_play_avg4: Mapped[Optional[float]] = mapped_column(Float)
    home_turnover_diff_avg4: Mapped[Optional[float]] = mapped_column(Float)
    away_turnover_diff_avg4: Mapped[Optional[float]] = mapped_column(Float)
    # Rest and schedule
    home_rest_days: Mapped[Optional[int]] = mapped_column(Integer)
    away_rest_days: Mapped[Optional[int]] = mapped_column(Integer)
    rest_diff: Mapped[Optional[int]] = mapped_column(Integer)    # home - away
    # Situational
    div_game: Mapped[Optional[bool]] = mapped_column(Boolean)
    playoff_game: Mapped[Optional[bool]] = mapped_column(Boolean)
    neutral_site: Mapped[Optional[bool]] = mapped_column(Boolean)
    # Betting context
    spread_line: Mapped[Optional[float]] = mapped_column(Float)
    total_line: Mapped[Optional[float]] = mapped_column(Float)
    implied_home_win_prob: Mapped[Optional[float]] = mapped_column(Float)
    # Target labels (NULL until game is played)
    home_win: Mapped[Optional[bool]] = mapped_column(Boolean)
    home_covered: Mapped[Optional[bool]] = mapped_column(Boolean)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── model_predictions ──────────────────────────────────────────────────────────


class ModelPrediction(Base):
    """One prediction row per (game, model_name, model_version)."""

    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint("game_id", "model_name", "model_version", name="uq_prediction"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(50), ForeignKey("games.game_id"), index=True)
    model_name: Mapped[str] = mapped_column(String(50))   # win_predictor, ats_predictor, baseline
    model_version: Mapped[str] = mapped_column(String(20))
    season: Mapped[int] = mapped_column(Integer, index=True)
    week: Mapped[int] = mapped_column(Integer)
    home_team: Mapped[str] = mapped_column(String(10))
    away_team: Mapped[str] = mapped_column(String(10))
    home_win_prob: Mapped[Optional[float]] = mapped_column(Float)
    home_cover_prob: Mapped[Optional[float]] = mapped_column(Float)
    predicted_winner: Mapped[Optional[str]] = mapped_column(String(10))
    actual_winner: Mapped[Optional[str]] = mapped_column(String(10))
    correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )


# ── chatbot_logs ───────────────────────────────────────────────────────────────


class ChatbotLog(Base):
    """Audit log for every chatbot turn: user messages, assistant replies, tool calls."""

    __tablename__ = "chatbot_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(50), index=True)
    role: Mapped[str] = mapped_column(String(20))          # user, assistant, tool
    content: Mapped[Optional[str]] = mapped_column(Text)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100))
    tool_args: Mapped[Optional[dict]] = mapped_column(JSON)
    tool_result: Mapped[Optional[dict]] = mapped_column(JSON)
    llm_provider: Mapped[Optional[str]] = mapped_column(String(30))
    model_name: Mapped[Optional[str]] = mapped_column(String(60))
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, server_default=func.now(), index=True
    )


# ── registry ───────────────────────────────────────────────────────────────────

ALL_TABLES = [
    "teams",
    "games",
    "team_game_stats",
    "player_game_stats",
    "player_season_stats",
    "injuries",
    "odds",
    "engineered_features",
    "model_predictions",
    "chatbot_logs",
]
