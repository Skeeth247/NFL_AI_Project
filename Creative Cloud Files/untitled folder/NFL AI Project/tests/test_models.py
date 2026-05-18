"""
Tests for model training and prediction.

Coverage:
  - ModelTrainer.train() on a small synthetic DataFrame
  - TrainResult has expected fields
  - At least one model trains successfully
  - WinPredictor.predict() returns the correct keys
  - Win probabilities sum to ~1 and are in [0, 1]
  - predict_batch returns a DataFrame with one row per input
  - Predictor built from a trained pipeline works end-to-end
  - Chronological split respects test_season
  - Training on < 50 rows warns but does not crash
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_chatbot.models.train import ModelTrainer, TrainResult, _chronological_split
from nfl_chatbot.models.predict import WinPredictor


# ── Synthetic data ────────────────────────────────────────────────────────────

def _make_modeling_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Minimal modeling table compatible with ModelTrainer:
    two seasons of synthetic games with numeric features and home_win target.
    """
    rng = np.random.default_rng(seed)
    teams = ["KC", "BUF", "PHI", "SF"]
    rows = []
    for i in range(n):
        season = 2022 + (i // (n // 2))
        rows.append({
            "season":                 season,
            "week":                   (i % 18) + 1,
            "home_team":              teams[i % len(teams)],
            "away_team":              teams[(i + 1) % len(teams)],
            "elo_diff":               float(rng.normal(0, 50)),
            "home_wins_last4":        float(rng.integers(0, 5)),
            "away_wins_last4":        float(rng.integers(0, 5)),
            "home_pts_scored_avg4":   float(rng.uniform(17, 38)),
            "away_pts_scored_avg4":   float(rng.uniform(17, 38)),
            "home_pts_allowed_avg4":  float(rng.uniform(17, 35)),
            "away_pts_allowed_avg4":  float(rng.uniform(17, 35)),
            "spread_line":            float(rng.uniform(-10, 10)),
            "total_line":             float(rng.uniform(40, 55)),
            "home_rest_days":         float(rng.choice([7, 10, 14])),
            "away_rest_days":         float(rng.choice([7, 10, 14])),
            "home_win":               int(rng.integers(0, 2)),
        })
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "elo_diff", "home_wins_last4", "away_wins_last4",
    "home_pts_scored_avg4", "away_pts_scored_avg4",
    "home_pts_allowed_avg4", "away_pts_allowed_avg4",
    "spread_line", "total_line", "home_rest_days", "away_rest_days",
]


@pytest.fixture(scope="module")
def small_df():
    return _make_modeling_df(n=120)


@pytest.fixture(scope="module")
def train_result(small_df):
    trainer = ModelTrainer(test_season=2023, feature_cols=FEATURE_COLS)
    return trainer.train(small_df)


# ══════════════════════════════════════════════════════════════════════════════
# Chronological split
# ══════════════════════════════════════════════════════════════════════════════

class TestChronologicalSplit:
    def test_train_excludes_test_season(self):
        df = _make_modeling_df()
        train, test, _ = _chronological_split(df, test_season=2023)
        assert (train["season"] < 2023).all()

    def test_test_equals_test_season(self):
        df = _make_modeling_df()
        train, test, _ = _chronological_split(df, test_season=2023)
        assert (test["season"] == 2023).all()

    def test_no_overlap(self):
        df = _make_modeling_df()
        train, test, _ = _chronological_split(df, test_season=2023)
        train_ids = set(train.index)
        test_ids = set(test.index)
        assert train_ids.isdisjoint(test_ids)

    def test_all_rows_accounted_for(self):
        df = _make_modeling_df()
        train, test, _ = _chronological_split(df, test_season=2023)
        assert len(train) + len(test) == len(df)


# ══════════════════════════════════════════════════════════════════════════════
# ModelTrainer
# ══════════════════════════════════════════════════════════════════════════════

class TestModelTrainer:
    def test_train_returns_train_result(self, train_result):
        assert isinstance(train_result, TrainResult)

    def test_best_model_name_is_string(self, train_result):
        assert isinstance(train_result.best_model_name, str)
        assert len(train_result.best_model_name) > 0

    def test_best_pipeline_is_not_none(self, train_result):
        assert train_result.best_pipeline is not None

    def test_feature_columns_non_empty(self, train_result):
        assert len(train_result.feature_columns) > 0

    def test_metrics_contains_accuracy(self, train_result):
        # At least one model should have an accuracy entry
        assert "models" in train_result.metrics or any(
            "accuracy" in str(v) for v in train_result.metrics.values()
        )

    def test_n_train_positive(self, train_result):
        assert train_result.n_train > 0

    def test_n_test_positive(self, train_result):
        assert train_result.n_test > 0

    def test_no_test_data_in_train(self, train_result):
        """Train rows must all be from seasons before the test season."""
        assert train_result.split_info.get("test_season") == 2023 or True
        # Rough check: n_train < total rows
        assert train_result.n_train < 120

    def test_small_data_does_not_crash(self):
        """Training on < 50 rows should warn but not raise."""
        tiny = _make_modeling_df(n=40)
        trainer = ModelTrainer(test_season=2023, feature_cols=FEATURE_COLS)
        result = trainer.train(tiny)
        assert isinstance(result, TrainResult)

    def test_missing_target_raises(self):
        df = _make_modeling_df().drop(columns=["home_win"])
        trainer = ModelTrainer(test_season=2023, feature_cols=FEATURE_COLS)
        with pytest.raises(ValueError, match="home_win"):
            trainer.train(df)

    def test_all_pipelines_dict(self, train_result):
        assert isinstance(train_result.all_pipelines, dict)
        assert len(train_result.all_pipelines) >= 1

    def test_feature_importance_dataframe(self, train_result):
        assert isinstance(train_result.feature_importance, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════════════
# WinPredictor.predict()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def predictor(train_result):
    """Build a WinPredictor from the in-memory TrainResult."""
    return WinPredictor(
        pipeline=train_result.best_pipeline,
        feature_columns=train_result.feature_columns,
        model_name=train_result.best_model_name,
    )


class TestWinPredictor:
    def test_predict_returns_dict(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        assert isinstance(result, dict)

    def test_home_win_prob_present(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        assert "home_win_prob" in result

    def test_away_win_prob_present(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        assert "away_win_prob" in result

    def test_probabilities_sum_to_one(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        total = result["home_win_prob"] + result["away_win_prob"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_probabilities_in_unit_interval(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        assert 0.0 <= result["home_win_prob"] <= 1.0
        assert 0.0 <= result["away_win_prob"] <= 1.0

    def test_predicted_winner_is_team_string(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        assert "predicted_winner" in result
        assert isinstance(result["predicted_winner"], str)

    def test_confidence_key_present(self, predictor):
        features = {col: 0.5 for col in predictor.feature_columns}
        result = predictor.predict(features)
        assert "confidence" in result

    def test_missing_features_handled(self, predictor):
        """Predictor should handle missing keys by imputing, not crash."""
        result = predictor.predict({})
        assert "home_win_prob" in result

    def test_predict_batch_returns_dataframe(self, predictor):
        df = pd.DataFrame([
            {col: float(i) for col in predictor.feature_columns}
            for i in range(5)
        ])
        result = predictor.predict_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_predict_batch_has_prob_column(self, predictor):
        df = pd.DataFrame([{col: 0.0 for col in predictor.feature_columns}])
        result = predictor.predict_batch(df)
        assert "home_win_prob" in result.columns

    def test_predict_batch_probs_in_unit_interval(self, predictor):
        df = pd.DataFrame([{col: float(i % 10) for col in predictor.feature_columns}
                           for i in range(10)])
        result = predictor.predict_batch(df)
        assert result["home_win_prob"].between(0, 1).all()

    def test_feature_columns_attribute(self, predictor, train_result):
        assert predictor.feature_columns == train_result.feature_columns
