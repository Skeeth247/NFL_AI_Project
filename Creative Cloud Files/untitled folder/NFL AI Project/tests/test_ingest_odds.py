"""Tests for the betting odds ingestion module."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from nfl_chatbot.data.database import init_db, read_table
from nfl_chatbot.data.ingest_odds import (
    OddsIngestResult,
    build_game_id,
    ingest,
    load_from_csv,
    normalize_team,
    normalize_week,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"
NORMALISED_CSV = FIXTURES / "sample_odds.csv"
KAGGLE_CSV = FIXTURES / "sample_odds_kaggle.csv"

ALL_SEASONS = [2020, 2021, 2022, 2023, 2024]


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


# ── normalize_team ─────────────────────────────────────────────────────────────


class TestNormalizeTeam:
    def test_full_name(self):
        assert normalize_team("Kansas City Chiefs") == "KC"

    def test_full_name_lowercase(self):
        assert normalize_team("kansas city chiefs") == "KC"

    def test_city_only(self):
        assert normalize_team("Kansas City") == "KC"

    def test_already_abbreviation(self):
        assert normalize_team("KC") == "KC"

    def test_abbreviation_lowercase(self):
        assert normalize_team("kc") == "KC"

    def test_historical_relocation_raiders(self):
        assert normalize_team("Oakland Raiders") == "LV"

    def test_historical_relocation_chargers(self):
        assert normalize_team("San Diego Chargers") == "LAC"

    def test_historical_relocation_rams(self):
        assert normalize_team("St. Louis Rams") == "LA"

    def test_washington_variants(self):
        assert normalize_team("Washington Redskins") == "WAS"
        assert normalize_team("Washington Football Team") == "WAS"
        assert normalize_team("Washington Commanders") == "WAS"

    def test_49ers(self):
        assert normalize_team("San Francisco 49ers") == "SF"
        assert normalize_team("San Francisco") == "SF"

    def test_unknown_returns_none(self):
        assert normalize_team("Fake Team Name") is None

    def test_empty_string_returns_none(self):
        assert normalize_team("") is None

    def test_nan_returns_none(self):
        assert normalize_team(float("nan")) is None

    def test_all_32_teams_have_mapping(self):
        teams = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens",
            "Buffalo Bills", "Carolina Panthers", "Chicago Bears",
            "Cincinnati Bengals", "Cleveland Browns", "Dallas Cowboys",
            "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
            "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
            "Los Angeles Rams", "Miami Dolphins", "Minnesota Vikings",
            "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers",
            "San Francisco 49ers", "Seattle Seahawks", "Tampa Bay Buccaneers",
            "Tennessee Titans", "Washington Commanders",
        ]
        for name in teams:
            assert normalize_team(name) is not None, f"No mapping for {name!r}"


# ── normalize_week ─────────────────────────────────────────────────────────────


class TestNormalizeWeek:
    def test_integer(self):
        assert normalize_week(1) == 1

    def test_string_int(self):
        assert normalize_week("5") == 5

    def test_float_string(self):
        assert normalize_week("3.0") == 3

    def test_wildcard(self):
        assert normalize_week("Wildcard") == 19

    def test_wild_card_space(self):
        assert normalize_week("Wild Card") == 19

    def test_divisional(self):
        assert normalize_week("Divisional") == 20

    def test_conference(self):
        assert normalize_week("Conference") == 21

    def test_superbowl(self):
        assert normalize_week("Superbowl") == 22
        assert normalize_week("Super Bowl") == 22

    def test_none_returns_none(self):
        assert normalize_week(None) is None

    def test_nan_returns_none(self):
        assert normalize_week(float("nan")) is None

    def test_unrecognised_string_returns_none(self):
        assert normalize_week("bye week") is None


# ── build_game_id ──────────────────────────────────────────────────────────────


class TestBuildGameId:
    def test_format(self):
        gid = build_game_id(2024, 1, "KC", "SF")
        assert gid == "2024_01_SF_KC"

    def test_week_zero_padded(self):
        gid = build_game_id(2024, 9, "DAL", "PHI")
        assert gid == "2024_09_PHI_DAL"

    def test_week_two_digits(self):
        gid = build_game_id(2024, 14, "BUF", "MIA")
        assert gid == "2024_14_MIA_BUF"

    def test_full_team_names_normalised(self):
        gid = build_game_id(2023, 1, "Kansas City Chiefs", "Detroit Lions")
        assert gid == "2023_01_DET_KC"

    def test_unknown_home_raises(self):
        with pytest.raises(ValueError, match="home team"):
            build_game_id(2024, 1, "Unknown FC", "KC")

    def test_unknown_away_raises(self):
        with pytest.raises(ValueError, match="away team"):
            build_game_id(2024, 1, "KC", "Unknown FC")


# ── load_from_csv (normalised format) ─────────────────────────────────────────


class TestLoadFromCsvNormalised:
    def test_loads_successfully(self):
        df = load_from_csv(NORMALISED_CSV, ALL_SEASONS)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self):
        df = load_from_csv(NORMALISED_CSV, ALL_SEASONS)
        required = [
            "game_id", "season", "week", "home_team", "away_team",
            "spread_line", "total_line",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_season_filter(self):
        df = load_from_csv(NORMALISED_CSV, [2024])
        assert df["season"].unique().tolist() == [2024]

    def test_multi_season_filter(self):
        df = load_from_csv(NORMALISED_CSV, [2022, 2023])
        assert set(df["season"].unique()) == {2022, 2023}

    def test_no_duplicate_game_ids(self):
        df = load_from_csv(NORMALISED_CSV, ALL_SEASONS)
        assert df["game_id"].nunique() == len(df)

    def test_game_id_format(self):
        df = load_from_csv(NORMALISED_CSV, [2024])
        for gid in df["game_id"]:
            parts = gid.split("_")
            assert len(parts) == 4
            assert parts[0] == "2024"
            assert len(parts[1]) == 2

    def test_spread_is_numeric(self):
        df = load_from_csv(NORMALISED_CSV, ALL_SEASONS)
        assert pd.api.types.is_float_dtype(df["spread_line"].dropna())

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_csv(tmp_path / "nonexistent.csv", ALL_SEASONS)

    def test_team_abbreviations_normalised(self):
        df = load_from_csv(NORMALISED_CSV, ALL_SEASONS)
        all_teams = pd.concat([df["home_team"], df["away_team"]]).dropna().unique()
        for team in all_teams:
            assert team.isupper() and len(team) <= 3, f"Bad abbreviation: {team!r}"


# ── load_from_csv (Kaggle format) ─────────────────────────────────────────────


class TestLoadFromCsvKaggle:
    def test_loads_successfully(self):
        df = load_from_csv(KAGGLE_CSV, ALL_SEASONS)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self):
        df = load_from_csv(KAGGLE_CSV, ALL_SEASONS)
        for col in ["game_id", "season", "week", "home_team", "away_team", "spread_line"]:
            assert col in df.columns

    def test_spread_sign_home_favoured(self):
        """When home team is the favourite, spread_line must be negative."""
        df = load_from_csv(KAGGLE_CSV, [2024])
        # Row 0: KC at home, KC is favourite → spread_line should be negative
        kc_row = df[df["home_team"] == "KC"].iloc[0]
        assert kc_row["spread_line"] < 0

    def test_spread_sign_away_favoured(self):
        """When away team is the favourite, home spread_line must be positive."""
        df = load_from_csv(KAGGLE_CSV, [2024])
        # Row 3: SEA at home, LA is favourite → SEA is underdog → spread_line > 0
        sea_row = df[df["home_team"] == "SEA"].iloc[0]
        assert sea_row["spread_line"] > 0

    def test_game_id_nflverse_format(self):
        df = load_from_csv(KAGGLE_CSV, ALL_SEASONS)
        assert all("_" in gid for gid in df["game_id"])

    def test_season_filter_works(self):
        df = load_from_csv(KAGGLE_CSV, [2023])
        assert set(df["season"].unique()).issubset({2023})


# ── ingest (orchestrator) ──────────────────────────────────────────────────────


class TestIngest:
    def test_csv_source_returns_ok_result(self, engine):
        result = ingest(
            seasons=ALL_SEASONS,
            csv_path=NORMALISED_CSV,
            engine=engine,
        )
        assert result.ok
        assert result.source == "csv"
        assert result.rows > 0

    def test_seasons_found_populated(self, engine):
        result = ingest(seasons=ALL_SEASONS, csv_path=NORMALISED_CSV, engine=engine)
        assert len(result.seasons_found) > 0
        assert all(isinstance(s, (int, float)) for s in result.seasons_found)

    def test_data_inserted_into_sqlite(self, engine):
        result = ingest(seasons=ALL_SEASONS, csv_path=NORMALISED_CSV, engine=engine)
        df = read_table("odds", engine)
        assert len(df) == result.rows

    def test_no_source_returns_none_result(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ingest(
                seasons=ALL_SEASONS,
                csv_path=None,
                api_key=None,
            )
        assert not result.ok
        assert result.source == "none"
        assert result.rows == 0
        # Should have issued a UserWarning
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_missing_csv_falls_back_gracefully(self, tmp_path):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = ingest(
                seasons=ALL_SEASONS,
                csv_path=tmp_path / "nonexistent.csv",
                api_key=None,
            )
        assert not result.ok
        assert len(result.warnings) > 0

    def test_season_filtering_applied(self, engine):
        result = ingest(seasons=[2024], csv_path=NORMALISED_CSV, engine=engine)
        assert all(s == 2024 for s in result.seasons_found)

    def test_result_df_has_correct_columns(self, engine):
        result = ingest(seasons=ALL_SEASONS, csv_path=NORMALISED_CSV, engine=engine)
        required = ["game_id", "season", "week", "home_team", "away_team", "spread_line"]
        for col in required:
            assert col in result.df.columns

    def test_raw_parquet_saved(self, engine, tmp_path):
        ingest(
            seasons=ALL_SEASONS,
            csv_path=NORMALISED_CSV,
            engine=engine,
            raw_dir=tmp_path / "raw",
        )
        assert (tmp_path / "raw" / "odds.parquet").exists()

    def test_processed_csv_saved(self, engine, tmp_path):
        ingest(
            seasons=ALL_SEASONS,
            csv_path=NORMALISED_CSV,
            engine=engine,
            processed_dir=tmp_path / "processed",
        )
        csvs = list((tmp_path / "processed").glob("odds_*.csv"))
        assert len(csvs) == 1


# ── OddsIngestResult ───────────────────────────────────────────────────────────


class TestOddsIngestResult:
    def test_ok_when_source_is_csv(self):
        r = OddsIngestResult(source="csv", rows=10)
        assert r.ok is True

    def test_not_ok_when_source_is_none(self):
        r = OddsIngestResult(source="none")
        assert r.ok is False

    def test_empty_when_rows_zero(self):
        r = OddsIngestResult(source="none", rows=0)
        assert r.empty is True

    def test_not_empty_when_rows_positive(self):
        r = OddsIngestResult(source="csv", rows=5)
        assert r.empty is False
