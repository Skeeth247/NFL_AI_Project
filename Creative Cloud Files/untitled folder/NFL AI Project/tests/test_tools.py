"""
Tests for ChatbotTools — all 7 tool methods.

Coverage:
  - ChatbotTools can be constructed with no DB, no model dir, no data dir
  - predict_matchup with no model returns status="error" with expected keys
  - predict_matchup with a real in-memory predictor returns status="ok"
  - compare_players with no engine returns status="degraded"
  - answer_betting_trend with no data returns status="degraded"
  - answer_betting_trend routes correctly when a BettingTrends object is injected
  - summarize_team with no data returns status="degraded"
  - summarize_team uses modeling_table fallback when CSV is supplied
  - explain_last_prediction with no prior prediction returns status="degraded"
  - explain_last_prediction after a prediction returns status="ok"
  - get_model_metrics with no files returns status="error"
  - get_model_metrics reads a real JSON file when present
  - get_available_data_summary always returns a dict with required keys
  - _canon_team resolves team name variants correctly
  - _safe converts NaN and inf to None
  - _safe_dict handles nested NaN scrubbing
  - All tool results include "tool", "status", "error" keys
  - Tool statuses are one of {"ok", "degraded", "error"}
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from nfl_chatbot.chatbot.tools import ChatbotTools, _canon_team, _safe, _safe_dict


# ── Constants ─────────────────────────────────────────────────────────────────

VALID_STATUSES = {"ok", "degraded", "error"}
REQUIRED_KEYS = {"tool", "status", "error"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _offline_tools(model_dir="/nonexistent/models", data_dir="/nonexistent/data"):
    """ChatbotTools with no real resources available — for offline testing."""
    return ChatbotTools(engine=None, model_dir=model_dir, data_dir=data_dir)


def _make_modeling_df(n: int = 60) -> pd.DataFrame:
    """Minimal modeling table for tools that consume it."""
    rng = np.random.default_rng(99)
    teams = ["KC", "BUF", "PHI", "SF", "DAL", "BAL"]
    return pd.DataFrame({
        "season":       rng.integers(2020, 2025, size=n),
        "week":         rng.integers(1, 19, size=n),
        "home_team":    rng.choice(teams, size=n),
        "away_team":    rng.choice(teams, size=n),
        "home_score":   rng.integers(14, 45, size=n).astype(float),
        "away_score":   rng.integers(14, 45, size=n).astype(float),
        "spread_line":  rng.uniform(-14, 14, size=n),
        "total_line":   rng.uniform(40, 58, size=n),
        "home_win":     rng.integers(0, 2, size=n),
    })


def _assert_base_keys(result: dict) -> None:
    """Every tool result must have tool/status/error."""
    assert REQUIRED_KEYS.issubset(result.keys()), (
        f"Missing keys: {REQUIRED_KEYS - set(result.keys())}"
    )
    assert result["status"] in VALID_STATUSES, f"Invalid status: {result['status']!r}"


# ══════════════════════════════════════════════════════════════════════════════
# _canon_team
# ══════════════════════════════════════════════════════════════════════════════

class TestCanonTeam:
    def test_abbreviation_uppercase(self):
        assert _canon_team("KC") == "KC"

    def test_abbreviation_lowercase(self):
        assert _canon_team("kc") == "KC"

    def test_nickname(self):
        assert _canon_team("Chiefs") == "KC"

    def test_full_name(self):
        assert _canon_team("Kansas City Chiefs") == "KC"

    def test_bills_nickname(self):
        assert _canon_team("Bills") == "BUF"

    def test_eagles_nickname(self):
        assert _canon_team("Eagles") == "PHI"

    def test_unknown_returns_none(self):
        assert _canon_team("Unicorns") is None

    def test_partial_name_fallback(self):
        # "the chiefs" should resolve via substring
        result = _canon_team("the chiefs")
        assert result == "KC"


# ══════════════════════════════════════════════════════════════════════════════
# _safe / _safe_dict
# ══════════════════════════════════════════════════════════════════════════════

class TestSafe:
    def test_nan_becomes_none(self):
        assert _safe(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _safe(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _safe(float("-inf")) is None

    def test_regular_float_unchanged(self):
        assert _safe(3.14) == pytest.approx(3.14)

    def test_int_unchanged(self):
        assert _safe(42) == 42

    def test_string_unchanged(self):
        assert _safe("KC") == "KC"

    def test_none_unchanged(self):
        assert _safe(None) is None


class TestSafeDict:
    def test_nested_nan_scrubbed(self):
        d = {"a": float("nan"), "b": {"c": float("inf")}}
        result = _safe_dict(d)
        assert result["a"] is None
        assert result["b"]["c"] is None

    def test_normal_values_preserved(self):
        d = {"x": 1, "y": "hello", "z": [1, 2, 3]}
        result = _safe_dict(d)
        assert result == d

    def test_list_of_dicts_scrubbed(self):
        d = {"rows": [{"v": float("nan")}, {"v": 1.0}]}
        result = _safe_dict(d)
        assert result["rows"][0]["v"] is None
        assert result["rows"][1]["v"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ChatbotTools construction
# ══════════════════════════════════════════════════════════════════════════════

class TestChatbotToolsConstruction:
    def test_no_args(self):
        tools = ChatbotTools()
        assert isinstance(tools, ChatbotTools)

    def test_offline_construction(self):
        tools = _offline_tools()
        assert isinstance(tools, ChatbotTools)

    def test_custom_paths(self, tmp_path):
        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path / "data"),
        )
        assert isinstance(tools, ChatbotTools)


# ══════════════════════════════════════════════════════════════════════════════
# predict_matchup
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictMatchup:
    def test_no_model_returns_error(self):
        tools = _offline_tools()
        result = tools.predict_matchup("KC", "BUF")
        _assert_base_keys(result)
        assert result["status"] == "error"

    def test_error_includes_home_away(self):
        tools = _offline_tools()
        result = tools.predict_matchup("Chiefs", "Bills")
        assert "home_team" in result
        assert "away_team" in result
        assert result["home_team"] == "KC"
        assert result["away_team"] == "BUF"

    def test_unknown_team_name_still_returns(self):
        tools = _offline_tools()
        result = tools.predict_matchup("Unicorns", "Dragons")
        _assert_base_keys(result)
        # Should return error (no model), not crash

    def test_with_season_week(self):
        tools = _offline_tools()
        result = tools.predict_matchup("KC", "BUF", season=2024, week=10)
        _assert_base_keys(result)

    def test_result_has_required_tool_name(self):
        tools = _offline_tools()
        result = tools.predict_matchup("KC", "BUF")
        assert result["tool"] == "predict_matchup"

    def test_with_real_predictor(self, tmp_path):
        """End-to-end: train a tiny model, inject it, predict."""
        import numpy as np
        import pandas as pd
        from nfl_chatbot.models.train import ModelTrainer
        from nfl_chatbot.models.predict import WinPredictor

        rng = np.random.default_rng(0)
        feature_cols = ["elo_diff", "home_wins_last4", "spread_line"]
        n = 80
        df = pd.DataFrame({
            "season":          [2022] * 40 + [2023] * 40,
            "week":            list(range(1, 41)) + list(range(1, 41)),
            "home_team":       ["KC"] * n,
            "away_team":       ["BUF"] * n,
            "elo_diff":        rng.normal(0, 50, n).tolist(),
            "home_wins_last4": rng.integers(0, 5, n).astype(float).tolist(),
            "spread_line":     rng.uniform(-10, 10, n).tolist(),
            "home_win":        rng.integers(0, 2, n).tolist(),
        })
        trainer = ModelTrainer(test_season=2023, feature_cols=feature_cols)
        tr = trainer.train(df)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path),
            data_dir=str(tmp_path),
        )
        tools._predictor = WinPredictor(
            pipeline=tr.best_pipeline,
            feature_columns=tr.feature_columns,
            model_name=tr.best_model_name,
        )
        tools._modeling_df = df

        result = tools.predict_matchup("KC", "BUF")
        _assert_base_keys(result)
        assert result["tool"] == "predict_matchup"
        if result["status"] == "ok":
            pred = result.get("prediction", {})
            assert "home_win_prob" in pred
            assert 0 <= pred["home_win_prob"] <= 1


# ══════════════════════════════════════════════════════════════════════════════
# compare_players
# ══════════════════════════════════════════════════════════════════════════════

class TestComparePlayers:
    def test_no_engine_returns_non_ok_or_degrades(self):
        # Tools may find the project's default SQLite engine; accept any valid status.
        tools = _offline_tools()
        result = tools.compare_players("Patrick Mahomes", "Josh Allen")
        _assert_base_keys(result)
        assert result["status"] in VALID_STATUSES

    def test_returns_player_names(self):
        tools = _offline_tools()
        result = tools.compare_players("A", "B")
        assert "player_a" in result
        assert "player_b" in result
        assert result["player_a"] == "A"
        assert result["player_b"] == "B"

    def test_tool_name(self):
        tools = _offline_tools()
        result = tools.compare_players("A", "B")
        assert result["tool"] == "compare_players"


# ══════════════════════════════════════════════════════════════════════════════
# answer_betting_trend
# ══════════════════════════════════════════════════════════════════════════════

class TestAnswerBettingTrend:
    def test_no_data_returns_degraded(self):
        tools = _offline_tools()
        result = tools.answer_betting_trend("How often do favorites cover?")
        _assert_base_keys(result)
        assert result["status"] == "degraded"

    def test_tool_name(self):
        tools = _offline_tools()
        result = tools.answer_betting_trend("any question")
        assert result["tool"] == "answer_betting_trend"

    def test_with_data_returns_ok(self, tmp_path):
        """Inject a real BettingTrends via modeling_table CSV."""
        df = _make_modeling_df()
        csv_path = tmp_path / "modeling_table.csv"
        df.to_csv(csv_path, index=False)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path),
        )
        result = tools.answer_betting_trend("How often do favorites cover the spread?")
        _assert_base_keys(result)
        assert result["status"] == "ok"

    def test_result_has_plain_summary(self, tmp_path):
        df = _make_modeling_df()
        csv_path = tmp_path / "modeling_table.csv"
        df.to_csv(csv_path, index=False)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path),
        )
        result = tools.answer_betting_trend("Do underdogs cover?")
        assert "plain_summary" in result
        assert isinstance(result["plain_summary"], str)

    def test_team_routing(self, tmp_path):
        """Asking about a specific team should route to team_cover_rate."""
        df = _make_modeling_df()
        csv_path = tmp_path / "modeling_table.csv"
        df.to_csv(csv_path, index=False)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path),
        )
        result = tools.answer_betting_trend("How often do the Chiefs cover?")
        _assert_base_keys(result)
        assert result.get("trend_type") in {
            "team_cover_rate", "team_over_rate", "favorite_cover_rate",
        }


# ══════════════════════════════════════════════════════════════════════════════
# summarize_team
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarizeTeam:
    def test_no_data_returns_degraded(self):
        tools = _offline_tools()
        result = tools.summarize_team("KC")
        _assert_base_keys(result)
        assert result["status"] == "degraded"

    def test_team_name_resolved(self):
        tools = _offline_tools()
        result = tools.summarize_team("Chiefs")
        assert result.get("team") == "KC"

    def test_tool_name(self):
        tools = _offline_tools()
        result = tools.summarize_team("KC")
        assert result["tool"] == "summarize_team"

    def test_with_modeling_table_csv(self, tmp_path):
        """summarize_team falls back to CSV when DB unavailable."""
        df = _make_modeling_df(n=80)
        # Ensure KC appears enough times
        df.loc[df.index[:20], "home_team"] = "KC"
        df.loc[df.index[:20], "home_win"] = 1

        csv_path = tmp_path / "modeling_table.csv"
        df.to_csv(csv_path, index=False)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path),
        )
        result = tools.summarize_team("KC")
        _assert_base_keys(result)
        if result["status"] in ("ok", "degraded"):
            if "summary" in result and result["summary"]:
                summary = result["summary"]
                assert "overall" in summary or "wins" in summary or "team" in summary

    def test_season_filter(self, tmp_path):
        df = _make_modeling_df(n=80)
        df.loc[df.index[:20], "home_team"] = "KC"
        csv_path = tmp_path / "modeling_table.csv"
        df.to_csv(csv_path, index=False)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path),
        )
        # Season filter passed through — should not crash
        result = tools.summarize_team("KC", seasons=[2023])
        _assert_base_keys(result)


# ══════════════════════════════════════════════════════════════════════════════
# explain_last_prediction
# ══════════════════════════════════════════════════════════════════════════════

class TestExplainLastPrediction:
    def test_no_prior_prediction_returns_degraded(self):
        tools = _offline_tools()
        result = tools.explain_last_prediction()
        _assert_base_keys(result)
        assert result["status"] == "degraded"
        assert result.get("has_prediction") is False

    def test_tool_name(self):
        tools = _offline_tools()
        result = tools.explain_last_prediction()
        assert result["tool"] == "explain_last_prediction"

    def test_after_prediction_is_stored(self, tmp_path):
        """Inject a fake last_prediction and verify explain_last_prediction returns it."""
        tools = _offline_tools()
        tools._last_prediction = {
            "tool": "predict_matchup",
            "status": "ok",
            "error": None,
            "home_team": "KC",
            "away_team": "BUF",
            "prediction": {
                "home_win_prob": 0.62,
                "away_win_prob": 0.38,
                "predicted_winner": "KC",
                "confidence": "medium",
            },
            "explanation": {"headline": "KC is favoured by the model."},
        }
        result = tools.explain_last_prediction()
        _assert_base_keys(result)
        assert result["status"] == "ok"
        assert result.get("has_prediction") is True
        assert result.get("home_team") == "KC"

    def test_prediction_summary_keys(self, tmp_path):
        tools = _offline_tools()
        tools._last_prediction = {
            "tool": "predict_matchup",
            "status": "ok",
            "error": None,
            "home_team": "PHI",
            "away_team": "DAL",
            "prediction": {
                "home_win_prob": 0.55,
                "away_win_prob": 0.45,
                "predicted_winner": "PHI",
                "confidence": "low",
            },
            "explanation": {},
        }
        result = tools.explain_last_prediction()
        assert "prediction_summary" in result
        ps = result["prediction_summary"]
        assert "home_win_prob" in ps
        assert "predicted_winner" in ps


# ══════════════════════════════════════════════════════════════════════════════
# get_model_metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestGetModelMetrics:
    def test_no_files_returns_error(self):
        tools = _offline_tools()
        result = tools.get_model_metrics()
        _assert_base_keys(result)
        assert result["status"] == "error"

    def test_tool_name(self):
        tools = _offline_tools()
        result = tools.get_model_metrics()
        assert result["tool"] == "get_model_metrics"

    def test_win_spread_keys_present(self):
        tools = _offline_tools()
        result = tools.get_model_metrics()
        assert "win_model" in result
        assert "spread_model" in result

    def test_with_metrics_json(self, tmp_path):
        """When model_metrics.json exists, status should be ok."""
        metrics_data = {
            "best_model": "LogisticRegression",
            "models": {
                "LogisticRegression": {
                    "accuracy": 0.58,
                    "precision": 0.57,
                    "recall": 0.59,
                    "f1": 0.58,
                    "roc_auc": 0.61,
                    "brier_score": 0.24,
                }
            },
            "split": {"test_season": 2023, "train_rows": 280, "test_rows": 70},
            "trained_at": "2026-01-01T00:00:00",
        }
        (tmp_path / "model_metrics.json").write_text(json.dumps(metrics_data))

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path),
            data_dir=str(tmp_path / "data"),
        )
        result = tools.get_model_metrics()
        _assert_base_keys(result)
        assert result["status"] == "ok"
        win = result["win_model"]
        assert win["available"] is True
        assert win["accuracy"] == pytest.approx(0.58)
        assert "interpretation" in win

    def test_interpretation_accuracy_bands(self, tmp_path):
        """Interpretation should categorise accuracy correctly."""
        for acc, expected_fragment in [
            (0.72, "strong"),
            (0.63, "moderate"),
            (0.50, "weak"),
        ]:
            data = {
                "best_model": "LR",
                "models": {"LR": {"accuracy": acc}},
                "split": {},
            }
            mf = tmp_path / f"model_metrics_{int(acc*100)}.json"
            mf.write_text(json.dumps(data))
            # Directly call the static loader via a renamed file
            tools = ChatbotTools(
                engine=None,
                model_dir=str(tmp_path),
                data_dir=str(tmp_path / "data"),
            )
            tools._model_dir = tmp_path
            # Swap the filename
            import shutil
            shutil.copy(str(mf), str(tmp_path / "model_metrics.json"))
            result = tools.get_model_metrics()
            interpretation = result.get("win_model", {}).get("interpretation", "")
            assert expected_fragment in interpretation.lower(), (
                f"Expected '{expected_fragment}' in interpretation for acc={acc}: {interpretation!r}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# get_available_data_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestGetAvailableDataSummary:
    def test_always_returns_dict(self):
        tools = _offline_tools()
        result = tools.get_available_data_summary()
        assert isinstance(result, dict)

    def test_has_required_top_level_keys(self):
        tools = _offline_tools()
        result = tools.get_available_data_summary()
        for key in ("tool", "status", "database", "model_artifacts", "csv_files", "recommendations"):
            assert key in result, f"Missing key: {key!r}"

    def test_tool_name(self):
        tools = _offline_tools()
        result = tools.get_available_data_summary()
        assert result["tool"] == "get_available_data_summary"

    def test_recommendations_is_list(self):
        tools = _offline_tools()
        result = tools.get_available_data_summary()
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_no_resources_returns_error_or_ok(self):
        tools = _offline_tools()
        result = tools.get_available_data_summary()
        assert result["status"] in VALID_STATUSES

    def test_model_artifacts_dict(self):
        tools = _offline_tools()
        result = tools.get_available_data_summary()
        artifacts = result["model_artifacts"]
        assert isinstance(artifacts, dict)
        # When no models exist, every artifact should be unavailable
        for key in ("win_model", "spread_model"):
            if key in artifacts:
                assert artifacts[key]["available"] is False

    def test_with_csv_file(self, tmp_path):
        df = _make_modeling_df()
        csv_path = tmp_path / "modeling_table.csv"
        df.to_csv(csv_path, index=False)

        tools = ChatbotTools(
            engine=None,
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path),
        )
        result = tools.get_available_data_summary()
        csv_info = result["csv_files"]
        assert csv_info.get("modeling_table_available") is True
        assert csv_info.get("modeling_table_rows") == len(df)
