"""
Tests for src/nfl_chatbot/data/cleaning.py

Coverage:
  - standardize_team_abbreviations: full names, abbreviations, historical,
      city-only, unknown values, NaN input
  - handle_missing_values: all six strategies, unknown strategy, absent column
  - validate_required_columns: passes, raises on missing
  - clean_games: game_id generation, dedup, date conversion, bad team drop
  - clean_team_stats: team normalisation, count-col fill, dedup
  - clean_player_stats: drops no-id rows, dedup
  - clean_injuries: game_status normalisation, practice normalisation, dedup
  - clean_odds: game_id rebuild, numeric coercion, dedup
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_chatbot.data.cleaning import (
    CleaningReport,
    clean_games,
    clean_injuries,
    clean_odds,
    clean_player_stats,
    clean_team_stats,
    handle_missing_values,
    standardize_team_abbreviations,
    validate_required_columns,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _games_df(**kwargs) -> pd.DataFrame:
    base = {
        "season": [2023, 2023],
        "week": [1, 2],
        "home_team": ["KC", "SF"],
        "away_team": ["BUF", "DAL"],
        "gameday": ["2023-09-07", "2023-09-14"],
        "game_type": ["REG", "REG"],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def _injuries_df(**kwargs) -> pd.DataFrame:
    base = {
        "season": [2023, 2023],
        "week": [1, 1],
        "team": ["KC", "BUF"],
        "player_name": ["Patrick Mahomes", "Stefon Diggs"],
        "game_status": ["Questionable", "Out"],
        "practice_wed": ["LP", "DNP"],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ── standardize_team_abbreviations ────────────────────────────────────────────


class TestStandardizeTeamAbbreviations:
    def test_full_name_to_abbr(self):
        s = pd.Series(["Kansas City Chiefs"])
        assert standardize_team_abbreviations(s).iloc[0] == "KC"

    def test_already_valid_abbr_unchanged(self):
        abbrs = ["KC", "BUF", "SF", "DAL", "NE", "GB", "LA", "LV", "LAC"]
        s = pd.Series(abbrs)
        result = standardize_team_abbreviations(s)
        assert list(result) == abbrs

    def test_city_only_form(self):
        s = pd.Series(["kansas city", "Buffalo", "Green Bay"])
        result = standardize_team_abbreviations(s)
        assert list(result) == ["KC", "BUF", "GB"]

    def test_historical_oakland_raiders(self):
        s = pd.Series(["Oakland Raiders", "Oakland"])
        result = standardize_team_abbreviations(s)
        assert list(result) == ["LV", "LV"]

    def test_historical_san_diego_chargers(self):
        s = pd.Series(["San Diego Chargers", "San Diego"])
        result = standardize_team_abbreviations(s)
        assert list(result) == ["LAC", "LAC"]

    def test_historical_washington_names(self):
        s = pd.Series(["Washington Redskins", "Washington Football Team", "Washington Commanders"])
        result = standardize_team_abbreviations(s)
        assert list(result).count("WAS") == 3

    def test_unknown_value_becomes_na(self):
        s = pd.Series(["Not A Team"])
        result = standardize_team_abbreviations(s)
        assert pd.isna(result.iloc[0])

    def test_nan_input_becomes_na(self):
        s = pd.Series([float("nan"), None, pd.NA])
        result = standardize_team_abbreviations(s)
        assert result.isna().all()

    def test_mixed_series(self):
        s = pd.Series(["KC", "Kansas City Chiefs", "Unknown", None])
        result = standardize_team_abbreviations(s)
        assert result.iloc[0] == "KC"
        assert result.iloc[1] == "KC"
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])

    def test_case_insensitive(self):
        s = pd.Series(["kc", "KC", "Kc"])
        result = standardize_team_abbreviations(s)
        assert list(result) == ["KC", "KC", "KC"]

    def test_all_32_teams_have_valid_abbrs(self):
        from nfl_chatbot.data.ingest_odds import TEAM_NAME_TO_ABBR
        full_names = [k for k in TEAM_NAME_TO_ABBR if len(k.split()) >= 2 and not k[0].isupper()]
        s = pd.Series(full_names[:32])
        result = standardize_team_abbreviations(s)
        assert result.notna().all()


# ── handle_missing_values ─────────────────────────────────────────────────────


class TestHandleMissingValues:
    def _df_with_nulls(self) -> pd.DataFrame:
        return pd.DataFrame({
            "a": [1.0, None, 3.0, None],
            "b": [10.0, 20.0, None, None],
            "c": ["x", None, "x", "y"],
        })

    def test_zero_strategy(self):
        df = self._df_with_nulls()
        result = handle_missing_values(df, {"a": "zero"})
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == 0.0
        assert result["a"].iloc[3] == 0.0

    def test_median_strategy(self):
        df = self._df_with_nulls()
        result = handle_missing_values(df, {"a": "median"})
        # median of [1, 3] = 2.0
        assert result["a"].iloc[1] == pytest.approx(2.0)
        assert result["a"].isna().sum() == 0

    def test_mean_strategy(self):
        df = self._df_with_nulls()
        result = handle_missing_values(df, {"a": "mean"})
        assert result["a"].iloc[1] == pytest.approx(2.0)

    def test_mode_strategy(self):
        df = pd.DataFrame({"c": ["x", None, "x", "y"]})
        result = handle_missing_values(df, {"c": "mode"})
        assert result["c"].iloc[1] == "x"

    def test_drop_strategy(self):
        df = self._df_with_nulls()
        result = handle_missing_values(df, {"a": "drop"})
        assert len(result) == 2
        assert result["a"].isna().sum() == 0

    def test_forward_fill_strategy(self):
        df = pd.DataFrame({"v": [1.0, None, None, 4.0]})
        result = handle_missing_values(df, {"v": "forward_fill"})
        assert list(result["v"]) == [1.0, 1.0, 1.0, 4.0]

    def test_absent_column_silently_skipped(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = handle_missing_values(df, {"nonexistent": "zero"})
        assert list(result.columns) == ["a"]

    def test_no_nulls_column_untouched(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = handle_missing_values(df, {"a": "zero"})
        assert list(result["a"]) == [1.0, 2.0, 3.0]

    def test_multiple_columns_multiple_strategies(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
        result = handle_missing_values(df, {"a": "zero", "b": "zero"})
        assert result["a"].iloc[1] == 0.0
        assert result["b"].iloc[0] == 0.0

    def test_input_dataframe_not_mutated(self):
        df = pd.DataFrame({"a": [1.0, None]})
        _ = handle_missing_values(df, {"a": "zero"})
        assert df["a"].isna().any()

    def test_unknown_strategy_raises(self):
        df = pd.DataFrame({"a": [1.0, None]})
        with pytest.raises(ValueError, match="Unknown strategies"):
            handle_missing_values(df, {"a": "magic"})


# ── validate_required_columns ─────────────────────────────────────────────────


class TestValidateRequiredColumns:
    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_required_columns(df, ["a", "b"])  # no exception

    def test_empty_required_list(self):
        df = pd.DataFrame({"a": [1]})
        validate_required_columns(df, [])  # no exception

    def test_raises_on_single_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="c"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_raises_on_multiple_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_table_name_in_error_message(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="my_table"):
            validate_required_columns(df, ["b"], table_name="my_table")


# ── clean_games ───────────────────────────────────────────────────────────────


class TestCleanGames:
    def test_builds_game_id_when_absent(self):
        df = _games_df()
        cleaned, report = clean_games(df)
        assert "game_id" in cleaned.columns
        assert cleaned["game_id"].notna().all()
        assert cleaned["game_id"].iloc[0] == "2023_01_BUF_KC"

    def test_game_id_format(self):
        df = _games_df()
        cleaned, _ = clean_games(df)
        # nflverse format: {season}_{week:02d}_{away}_{home}
        assert cleaned["game_id"].iloc[1] == "2023_02_DAL_SF"

    def test_preserves_existing_game_ids(self):
        df = _games_df()
        df["game_id"] = ["2023_01_BUF_KC", "2023_02_DAL_SF"]
        cleaned, _ = clean_games(df)
        assert list(cleaned["game_id"]) == ["2023_01_BUF_KC", "2023_02_DAL_SF"]

    def test_fills_missing_game_id_entries(self):
        df = _games_df()
        df["game_id"] = [None, "2023_02_DAL_SF"]
        cleaned, _ = clean_games(df)
        assert cleaned["game_id"].iloc[0] == "2023_01_BUF_KC"
        assert cleaned["game_id"].iloc[1] == "2023_02_DAL_SF"

    def test_converts_gameday_to_date(self):
        df = _games_df()
        cleaned, _ = clean_games(df)
        import datetime
        assert isinstance(cleaned["gameday"].iloc[0], datetime.date)

    def test_normalises_game_type_lowercase(self):
        df = _games_df(game_type=["reg", "wildcard"])
        cleaned, _ = clean_games(df)
        assert cleaned["game_type"].iloc[0] == "REG"
        assert cleaned["game_type"].iloc[1] == "WC"

    def test_drops_duplicate_game_ids(self):
        df = _games_df()
        df = pd.concat([df, df], ignore_index=True)
        cleaned, report = clean_games(df)
        assert len(cleaned) == 2
        assert report.dupes_dropped == 2

    def test_drops_rows_with_unknown_team(self):
        df = _games_df(home_team=["KC", "NotATeam"])
        cleaned, report = clean_games(df)
        assert len(cleaned) == 1
        assert report.bad_teams_dropped == 1

    def test_normalises_team_full_names(self):
        df = _games_df(home_team=["Kansas City Chiefs", "San Francisco 49ers"])
        cleaned, _ = clean_games(df)
        assert cleaned["home_team"].iloc[0] == "KC"
        assert cleaned["home_team"].iloc[1] == "SF"

    def test_returns_cleaning_report(self):
        df = _games_df()
        _, report = clean_games(df)
        assert isinstance(report, CleaningReport)
        assert report.table == "games"
        assert report.rows_in == 2
        assert report.rows_out == 2

    def test_raises_on_missing_required_columns(self):
        df = pd.DataFrame({"season": [2023], "week": [1]})
        with pytest.raises(ValueError, match="home_team"):
            clean_games(df)


# ── clean_team_stats ──────────────────────────────────────────────────────────


class TestCleanTeamStats:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "game_id": ["2023_01_BUF_KC", "2023_01_BUF_KC"],
            "team": ["KC", "BUF"],
            "season": [2023, 2023],
            "week": [1, 1],
            "turnovers": [1, None],
            "penalties": [None, 5],
        })

    def test_fills_count_cols_with_zero(self):
        df = self._df()
        cleaned, _ = clean_team_stats(df)
        assert cleaned["turnovers"].isna().sum() == 0
        assert cleaned["penalties"].isna().sum() == 0

    def test_deduplicates_game_team(self):
        df = pd.concat([self._df(), self._df()], ignore_index=True)
        cleaned, report = clean_team_stats(df)
        assert len(cleaned) == 2
        assert report.dupes_dropped == 2

    def test_normalises_team_abbr(self):
        df = self._df()
        df["team"] = ["Kansas City Chiefs", "Buffalo Bills"]
        cleaned, _ = clean_team_stats(df)
        assert list(cleaned["team"]) == ["KC", "BUF"]

    def test_drops_unresolvable_team(self):
        df = self._df()
        df["team"] = ["KC", "FakeTeam"]
        cleaned, report = clean_team_stats(df)
        assert len(cleaned) == 1
        assert report.bad_teams_dropped == 1


# ── clean_player_stats ────────────────────────────────────────────────────────


class TestCleanPlayerStats:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "player_id": ["p1", "p2"],
            "player_name": ["Mahomes", "Allen"],
            "team": ["KC", "BUF"],
            "season": [2023, 2023],
            "week": [1, 1],
            "game_id": ["2023_01_BUF_KC", "2023_01_BUF_KC"],
            "passing_tds": [3, None],
        })

    def test_fills_count_cols(self):
        df = self._df()
        cleaned, _ = clean_player_stats(df)
        assert cleaned["passing_tds"].isna().sum() == 0

    def test_deduplicates_game_player(self):
        df = pd.concat([self._df(), self._df()], ignore_index=True)
        cleaned, report = clean_player_stats(df)
        assert len(cleaned) == 2
        assert report.dupes_dropped == 2

    def test_drops_rows_with_no_player_id_and_no_name(self):
        df = self._df()
        df.loc[0, "player_id"] = None
        df.loc[0, "player_name"] = None
        cleaned, report = clean_player_stats(df)
        assert len(cleaned) == 1
        assert "Dropped 1 rows" in report.warnings[0]

    def test_raises_if_no_player_identifier_column(self):
        df = pd.DataFrame({
            "team": ["KC"],
            "season": [2023],
            "week": [1],
        })
        with pytest.raises(ValueError, match="player_id or player_name"):
            clean_player_stats(df)

    def test_normalises_team_abbr(self):
        df = self._df()
        df["team"] = ["Kansas City Chiefs", "Buffalo Bills"]
        cleaned, _ = clean_player_stats(df)
        assert list(cleaned["team"]) == ["KC", "BUF"]


# ── clean_injuries ────────────────────────────────────────────────────────────


class TestCleanInjuries:
    def test_normalises_game_status_variants(self):
        statuses = ["out", "Out", "OUT", "questionable", "Q", "doubtful", "ir", "pup"]
        expected = ["Out", "Out", "Out", "Questionable", "Questionable", "Doubtful", "IR", "PUP"]
        df = pd.DataFrame({
            "season": [2023] * len(statuses),
            "week": [1] * len(statuses),
            "team": ["KC"] * len(statuses),
            "player_name": [f"P{i}" for i in range(len(statuses))],
            "game_status": statuses,
        })
        cleaned, _ = clean_injuries(df)
        assert list(cleaned["game_status"]) == expected

    def test_blank_and_dash_status_become_none(self):
        df = pd.DataFrame({
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "player_name": ["P1", "P2"],
            "game_status": ["", "-"],
        })
        cleaned, _ = clean_injuries(df)
        assert cleaned["game_status"].isna().all()

    def test_normalises_practice_status(self):
        df = _injuries_df(practice_wed=["Full Participation", "Did Not Participate"])
        cleaned, _ = clean_injuries(df)
        assert cleaned["practice_wed"].iloc[0] == "FP"
        assert cleaned["practice_wed"].iloc[1] == "DNP"

    def test_practice_lp_variants(self):
        for raw in ("LP", "lp", "Limited", "Limited Participation", "LIMITED"):
            df = pd.DataFrame({
                "season": [2023], "week": [1],
                "team": ["KC"], "player_name": ["P1"],
                "practice_wed": [raw],
            })
            cleaned, _ = clean_injuries(df)
            assert cleaned["practice_wed"].iloc[0] == "LP", f"Failed for {raw!r}"

    def test_deduplicates_by_season_week_player(self):
        df = _injuries_df()
        df = pd.concat([df, df], ignore_index=True)
        cleaned, report = clean_injuries(df)
        assert len(cleaned) == 2
        assert report.dupes_dropped == 2

    def test_keeps_last_duplicate_row(self):
        df = pd.DataFrame({
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "player_name": ["Mahomes", "Mahomes"],
            "game_status": ["Questionable", "Out"],  # status updated mid-week
        })
        cleaned, _ = clean_injuries(df)
        assert len(cleaned) == 1
        assert cleaned["game_status"].iloc[0] == "Out"

    def test_normalises_team_abbr(self):
        df = _injuries_df()
        df["team"] = ["Kansas City Chiefs", "Buffalo Bills"]
        cleaned, _ = clean_injuries(df)
        assert list(cleaned["team"]) == ["KC", "BUF"]


# ── clean_odds ────────────────────────────────────────────────────────────────


class TestCleanOdds:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "season": [2023, 2023],
            "week": [1, 2],
            "home_team": ["KC", "SF"],
            "away_team": ["BUF", "DAL"],
            "spread_line": [-3.5, "6.0"],
            "total_line": ["47.5", None],
        })

    def test_builds_game_id(self):
        df = self._df()
        cleaned, _ = clean_odds(df)
        assert "game_id" in cleaned.columns
        assert cleaned["game_id"].iloc[0] == "2023_01_BUF_KC"

    def test_coerces_numeric_columns(self):
        df = self._df()
        cleaned, _ = clean_odds(df)
        assert cleaned["spread_line"].dtype in (float, "float64")
        assert cleaned["spread_line"].iloc[1] == pytest.approx(6.0)

    def test_missing_numeric_stays_na(self):
        df = self._df()
        cleaned, _ = clean_odds(df)
        # total_line row 2 was None — should stay NaN, not 0
        assert pd.isna(cleaned["total_line"].iloc[1])

    def test_deduplicates_by_game_id(self):
        df = pd.concat([self._df(), self._df()], ignore_index=True)
        cleaned, report = clean_odds(df)
        assert len(cleaned) == 2
        assert report.dupes_dropped == 2

    def test_drops_unresolvable_team(self):
        df = self._df()
        df.loc[0, "home_team"] = "NotATeam"
        cleaned, report = clean_odds(df)
        assert len(cleaned) == 1
        assert report.bad_teams_dropped == 1

    def test_normalises_team_full_names(self):
        df = self._df()
        df["home_team"] = ["Kansas City Chiefs", "San Francisco 49ers"]
        cleaned, _ = clean_odds(df)
        assert cleaned["home_team"].iloc[0] == "KC"
        assert cleaned["home_team"].iloc[1] == "SF"

    def test_raises_on_missing_required_columns(self):
        df = pd.DataFrame({"season": [2023], "week": [1]})
        with pytest.raises(ValueError, match="home_team"):
            clean_odds(df)
