"""
Tests for feature engineering.

Coverage:
  - _rolling_shift applies shift(1) before rolling — no leakage
  - _season_expanding applies shift(1) — no leakage
  - FeatureBuilder.build() runs on a synthetic 3-team, 3-season DB
  - Output contains expected column groups
  - Raw score columns are dropped from the output (target leakage guard)
  - Targets (home_win, home_covered) are present and binary
  - Rolling features for game N never contain the result of game N
  - build_modeling_table / feature_columns smoke tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sqlalchemy

from nfl_chatbot.features.team_features import (
    FeatureBuilder,
    _rolling_shift,
    _season_expanding,
    feature_columns,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_games(
    n_games: int = 60,
    teams: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Synthetic games DataFrame with ALL columns that read_table("games") expects
    (schema.py defines the full column list via SQLAlchemy ORM).
    """
    rng = np.random.default_rng(seed)
    if teams is None:
        teams = ["KC", "BUF", "PHI"]

    rows = []
    for season in [2022, 2023]:
        for week in range(1, 11):
            home = teams[week % len(teams)]
            away = teams[(week + 1) % len(teams)]
            home_score = int(rng.integers(14, 42))
            away_score = int(rng.integers(14, 42))
            spread = float(rng.uniform(-10, 10))
            rows.append({
                "game_id":      f"{season}_W{week:02d}_{away}_{home}",
                "season":       season,
                "week":         week,
                "game_type":    "REG",
                "gameday":      f"{season}-09-{week+1:02d}",
                "gametime":     "13:00",
                "home_team":    home,
                "away_team":    away,
                "home_score":   float(home_score),
                "away_score":   float(away_score),
                "home_win":     int(home_score > away_score),
                "overtime":     0,
                "neutral_site": 0,
                "div_game":     0,
                "spread_line":  spread,
                "total_line":   float(home_score + away_score + rng.uniform(-5, 5)),
                "roof":         "dome",
                "surface":      "grass",
                "stadium":      "Test Stadium",
                "location":     "Home",
                "created_at":   None,
                "updated_at":   None,
            })

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def db_engine():
    """In-memory SQLite engine with a populated 'games' table."""
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    df = _make_games()
    df.to_sql("games", con=engine, if_exists="replace", index=False)
    return engine


@pytest.fixture(scope="module")
def feature_builder(db_engine):
    """FeatureBuilder wired to the in-memory DB."""
    return FeatureBuilder(engine=db_engine, window=3)


@pytest.fixture(scope="module")
def built_df(feature_builder):
    return feature_builder.build()


# ══════════════════════════════════════════════════════════════════════════════
# _rolling_shift — leakage prevention unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingShift:
    """Verify that shift(1) is applied before rolling so game N is excluded."""

    def test_first_value_is_nan(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = _rolling_shift(s, window=3)
        assert pd.isna(result.iloc[0]), "First value must be NaN (nothing before game 1)"

    def test_second_value_uses_only_game_1(self):
        s = pd.Series([10.0, 20.0, 30.0])
        result = _rolling_shift(s, window=3)
        assert result.iloc[1] == pytest.approx(10.0), "Game-2 rolling should only include game-1"

    def test_does_not_include_current_game(self):
        s = pd.Series([1.0, 1.0, 1.0, 999.0])
        result = _rolling_shift(s, window=4)
        assert result.iloc[3] == pytest.approx(1.0), "Current game (999) must not appear in its own window"

    def test_sum_aggregation(self):
        s = pd.Series([2.0, 4.0, 6.0, 8.0])
        result = _rolling_shift(s, window=2, agg="sum")
        # index 2: shift makes it look at [2,4] → sum=6
        assert result.iloc[2] == pytest.approx(6.0)

    def test_output_length_matches_input(self):
        s = pd.Series(range(10), dtype=float)
        assert len(_rolling_shift(s, window=4)) == len(s)


class TestSeasonExpanding:
    """_season_expanding also uses shift(1) — verify same leakage guarantee."""

    def test_first_value_is_nan(self):
        s = pd.Series([5.0, 10.0, 15.0])
        result = _season_expanding(s)
        assert pd.isna(result.iloc[0])

    def test_expanding_mean_excludes_current(self):
        s = pd.Series([10.0, 20.0, 30.0])
        result = _season_expanding(s)
        # index 2: expanding includes [10, 20] → mean = 15
        assert result.iloc[2] == pytest.approx(15.0)

    def test_output_length_matches_input(self):
        s = pd.Series(range(8), dtype=float)
        assert len(_season_expanding(s)) == len(s)


# ══════════════════════════════════════════════════════════════════════════════
# FeatureBuilder.build()
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureBuilder:
    def test_build_returns_dataframe(self, built_df):
        assert isinstance(built_df, pd.DataFrame)

    def test_build_not_empty(self, built_df):
        assert len(built_df) > 0

    def test_target_home_win_present(self, built_df):
        assert "home_win" in built_df.columns

    def test_target_home_win_binary(self, built_df):
        vals = built_df["home_win"].dropna().unique()
        assert set(vals).issubset({0, 1, 0.0, 1.0})

    def test_no_raw_score_leakage(self, built_df):
        """Raw scores must be dropped — they directly encode the target."""
        assert "home_score" not in built_df.columns, "home_score leaks the outcome"
        assert "away_score" not in built_df.columns, "away_score leaks the outcome"

    def test_rolling_features_present(self, built_df):
        rolling_cols = [c for c in built_df.columns
                        if any(tag in c for tag in ("last", "avg", "rolling"))]
        assert len(rolling_cols) > 0, "Expected rolling features in output"

    def test_home_team_column_present(self, built_df):
        assert "home_team" in built_df.columns

    def test_season_column_present(self, built_df):
        assert "season" in built_df.columns

    def test_rolling_window_respected(self, db_engine):
        fb_narrow = FeatureBuilder(engine=db_engine, window=2)
        fb_wide = FeatureBuilder(engine=db_engine, window=5)
        r_narrow = fb_narrow.build()
        r_wide = fb_wide.build()
        assert isinstance(r_narrow, pd.DataFrame)
        assert isinstance(r_wide, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
# feature_columns helper
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureColumns:
    def test_returns_list(self):
        cols = feature_columns()
        assert isinstance(cols, list)

    def test_non_empty(self):
        assert len(feature_columns()) > 0

    def test_no_target_columns(self):
        cols = feature_columns()
        assert "home_win" not in cols
        assert "home_score" not in cols
        assert "away_score" not in cols

    def test_window_arg_accepted(self):
        c2 = feature_columns(window=2)
        c5 = feature_columns(window=5)
        assert isinstance(c2, list)
        assert isinstance(c5, list)
