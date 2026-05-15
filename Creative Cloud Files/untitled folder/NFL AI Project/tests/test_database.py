"""Tests for the database layer (schema + CRUD utilities)."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from nfl_chatbot.data.database import (
    get_engine,
    init_db,
    read_table,
    table_row_counts,
    upsert_dataframe,
)
from nfl_chatbot.data.schema import ALL_TABLES, Base


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def engine() -> Engine:
    """In-memory SQLite engine, fresh for every test."""
    eng = create_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture
def team_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team_abbr": "KC",
                "team_name": "Kansas City Chiefs",
                "team_nick": "Chiefs",
                "team_conf": "AFC",
                "team_division": "AFC West",
            },
            {
                "team_abbr": "SF",
                "team_name": "San Francisco 49ers",
                "team_nick": "49ers",
                "team_conf": "NFC",
                "team_division": "NFC West",
            },
        ]
    )


@pytest.fixture
def game_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2024_01_KC_SF",
                "season": 2024,
                "week": 1,
                "game_type": "REG",
                "home_team": "SF",
                "away_team": "KC",
                "home_score": 17,
                "away_score": 27,
                "home_win": False,
                "neutral_site": False,
                "gameday": datetime.date(2024, 9, 8),
            }
        ]
    )


# ── init_db ────────────────────────────────────────────────────────────────────


class TestInitDb:
    def test_returns_table_list(self, engine):
        tables = sa_inspect(engine).get_table_names()
        assert isinstance(tables, list)
        assert len(tables) > 0

    def test_all_expected_tables_created(self, engine):
        tables = sa_inspect(engine).get_table_names()
        for name in ALL_TABLES:
            assert name in tables, f"Expected table '{name}' not found"

    def test_ten_tables_total(self, engine):
        tables = sa_inspect(engine).get_table_names()
        assert len(tables) == 10

    def test_idempotent_second_call(self, engine):
        # Calling init_db twice must not raise or duplicate tables
        init_db(engine)
        tables = sa_inspect(engine).get_table_names()
        assert len(tables) == 10

    def test_teams_columns(self, engine):
        cols = {c["name"] for c in sa_inspect(engine).get_columns("teams")}
        assert {"team_abbr", "team_name", "team_conf", "team_division"}.issubset(cols)

    def test_games_columns(self, engine):
        cols = {c["name"] for c in sa_inspect(engine).get_columns("games")}
        assert {"game_id", "season", "week", "home_team", "away_team"}.issubset(cols)

    def test_engineered_features_columns(self, engine):
        cols = {c["name"] for c in sa_inspect(engine).get_columns("engineered_features")}
        assert {"game_id", "home_elo", "away_elo", "elo_diff", "home_win"}.issubset(cols)

    def test_chatbot_logs_columns(self, engine):
        cols = {c["name"] for c in sa_inspect(engine).get_columns("chatbot_logs")}
        assert {"session_id", "role", "content", "tool_name", "tool_args"}.issubset(cols)

    def test_model_predictions_columns(self, engine):
        cols = {c["name"] for c in sa_inspect(engine).get_columns("model_predictions")}
        assert {"game_id", "model_name", "home_win_prob", "predicted_winner"}.issubset(cols)


# ── upsert_dataframe ───────────────────────────────────────────────────────────


class TestUpsertDataframe:
    def test_insert_returns_row_count(self, engine, team_df):
        n = upsert_dataframe(team_df, "teams", engine)
        assert n == 2

    def test_rows_are_stored(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine)
        assert len(result) == 2

    def test_upsert_updates_existing_row(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        updated = team_df.copy()
        updated.loc[updated["team_abbr"] == "KC", "team_name"] = "KC Chiefs Updated"
        upsert_dataframe(updated, "teams", engine)

        result = read_table("teams", engine, filters={"team_abbr": "KC"})
        assert result.iloc[0]["team_name"] == "KC Chiefs Updated"

    def test_upsert_does_not_duplicate_rows(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine)
        assert len(result) == 2

    def test_empty_dataframe_returns_zero(self, engine):
        empty = pd.DataFrame(columns=["team_abbr", "team_name", "team_nick", "team_conf", "team_division"])
        n = upsert_dataframe(empty, "teams", engine)
        assert n == 0

    def test_extra_columns_silently_dropped(self, engine, team_df):
        team_df["nonexistent_col"] = "noise"
        # Should not raise
        n = upsert_dataframe(team_df, "teams", engine)
        assert n == 2

    def test_nan_values_stored_as_null(self, engine, team_df):
        team_df["team_color"] = float("nan")
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine)
        assert result["team_color"].isna().all()

    def test_unknown_table_raises_value_error(self, engine, team_df):
        with pytest.raises(ValueError, match="Unknown table"):
            upsert_dataframe(team_df, "nonexistent_table", engine)

    def test_insert_game_row(self, engine, team_df, game_df):
        upsert_dataframe(team_df, "teams", engine)
        n = upsert_dataframe(game_df, "games", engine)
        assert n == 1

    def test_upsert_game_updates_score(self, engine, team_df, game_df):
        upsert_dataframe(team_df, "teams", engine)
        upsert_dataframe(game_df, "games", engine)

        updated = game_df.copy()
        updated["home_score"] = 30
        upsert_dataframe(updated, "games", engine)

        result = read_table("games", engine, filters={"game_id": "2024_01_KC_SF"})
        assert result.iloc[0]["home_score"] == 30


# ── read_table ─────────────────────────────────────────────────────────────────


class TestReadTable:
    def test_returns_dataframe(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine)
        assert len(result) == 2

    def test_empty_table_returns_empty_dataframe(self, engine):
        result = read_table("teams", engine)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_filter_by_column(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine, filters={"team_conf": "AFC"})
        assert len(result) == 1
        assert result.iloc[0]["team_abbr"] == "KC"

    def test_filter_no_match_returns_empty(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine, filters={"team_conf": "XYZ"})
        assert len(result) == 0

    def test_filter_multiple_conditions(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine, filters={"team_conf": "NFC", "team_nick": "49ers"})
        assert len(result) == 1
        assert result.iloc[0]["team_abbr"] == "SF"

    def test_select_specific_columns(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        result = read_table("teams", engine, columns=["team_abbr", "team_name"])
        assert list(result.columns) == ["team_abbr", "team_name"]

    def test_unknown_table_raises_value_error(self, engine):
        with pytest.raises(ValueError, match="Unknown table"):
            read_table("not_a_table", engine)


# ── table_row_counts ───────────────────────────────────────────────────────────


class TestTableRowCounts:
    def test_returns_dict(self, engine):
        counts = table_row_counts(engine)
        assert isinstance(counts, dict)

    def test_all_tables_present(self, engine):
        counts = table_row_counts(engine)
        for name in ALL_TABLES:
            assert name in counts

    def test_empty_tables_count_zero(self, engine):
        counts = table_row_counts(engine)
        assert all(v == 0 for v in counts.values())

    def test_counts_reflect_inserts(self, engine, team_df):
        upsert_dataframe(team_df, "teams", engine)
        counts = table_row_counts(engine)
        assert counts["teams"] == 2


# ── get_engine ─────────────────────────────────────────────────────────────────


class TestGetEngine:
    def test_in_memory_url(self):
        eng = get_engine("sqlite:///:memory:")
        assert eng is not None
        eng.dispose()

    def test_explicit_url_used(self):
        eng = get_engine("sqlite:///:memory:")
        assert "sqlite" in str(eng.url)
        eng.dispose()

    def test_async_url_stripped(self):
        eng = get_engine("sqlite+aiosqlite:///:memory:")
        assert "aiosqlite" not in str(eng.url)
        eng.dispose()
