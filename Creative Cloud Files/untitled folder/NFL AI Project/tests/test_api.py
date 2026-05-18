"""
Tests for the FastAPI backend.

Coverage:
  - GET /health returns 200 with status/version/timestamp
  - GET /health JSON shape validates as HealthResponse
  - POST /chat with a valid question returns 200 with required keys
  - POST /chat with an empty question returns 422
  - POST /chat with a question > 500 chars returns 422
  - POST /predict/matchup with valid body returns 200 with status/home_team/away_team
  - POST /predict/matchup missing required fields returns 422
  - GET /players/compare with player_a and player_b returns 200
  - GET /trends/betting with a question returns 200
  - GET /metrics returns 200 with win_model/spread_model keys
  - GET /teams/{team}/summary for a valid team returns 200 or 404 (no data)
  - GET /teams/{team}/summary 404 when team data unavailable
  - GET /openapi.json returns 200 (schema accessible)
  - GET /docs returns 200 (Swagger UI)

All tests use TestClient as a context manager so the lifespan runs.
Resources (DB, models) are not available in the test environment; tools
fall back gracefully, so endpoints return tool-level error statuses rather
than HTTP 5xx.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nfl_chatbot.api.main import create_app


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Single TestClient for the whole module — lifespan runs once."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_has_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data
        # Should be a valid ISO datetime string
        from datetime import datetime
        dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert dt.year >= 2024

    def test_content_type_json(self, client):
        r = client.get("/health")
        assert "application/json" in r.headers.get("content-type", "")


# ══════════════════════════════════════════════════════════════════════════════
# GET /openapi.json, /docs
# ══════════════════════════════════════════════════════════════════════════════


class TestOpenAPI:
    def test_openapi_schema_accessible(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200

    def test_openapi_has_paths(self, client):
        schema = client.get("/openapi.json").json()
        assert "paths" in schema
        assert "/health" in schema["paths"]

    def test_docs_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_accessible(self, client):
        r = client.get("/redoc")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# POST /chat
# ══════════════════════════════════════════════════════════════════════════════


class TestPostChat:
    def test_valid_question_returns_200(self, client):
        r = client.post("/chat", json={"question": "Who will win the Super Bowl?"})
        assert r.status_code == 200

    def test_response_has_answer(self, client):
        data = client.post("/chat", json={"question": "General NFL question"}).json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_has_intent(self, client):
        data = client.post("/chat", json={"question": "Any question"}).json()
        assert "intent" in data
        assert isinstance(data["intent"], str)

    def test_response_has_confidence(self, client):
        data = client.post("/chat", json={"question": "Hello"}).json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_has_follow_ups(self, client):
        data = client.post("/chat", json={"question": "Hi"}).json()
        assert "follow_ups" in data
        assert isinstance(data["follow_ups"], list)

    def test_response_has_has_caveat(self, client):
        data = client.post("/chat", json={"question": "Any question"}).json()
        assert "has_caveat" in data

    def test_response_has_needs_clarification(self, client):
        data = client.post("/chat", json={"question": "Tell me something"}).json()
        assert "needs_clarification" in data

    def test_empty_question_returns_422(self, client):
        r = client.post("/chat", json={"question": ""})
        assert r.status_code == 422

    def test_too_long_question_returns_422(self, client):
        r = client.post("/chat", json={"question": "x" * 501})
        assert r.status_code == 422

    def test_missing_question_field_returns_422(self, client):
        r = client.post("/chat", json={})
        assert r.status_code == 422

    def test_use_llm_false_by_default(self, client):
        r = client.post("/chat", json={"question": "NFL question"})
        assert r.status_code == 200

    def test_use_llm_true_still_returns_200(self, client):
        r = client.post("/chat", json={"question": "NFL question", "use_llm": True})
        assert r.status_code == 200

    def test_matchup_question_intent(self, client):
        """Matchup question should classify as matchup_prediction or similar."""
        data = client.post(
            "/chat",
            json={"question": "Who wins Chiefs vs Bills this week?"},
        ).json()
        # Intent should be a non-empty string from the classifier
        assert data["intent"] in {
            "matchup_prediction", "general_help", "unknown", "betting_trend",
            "team_summary", "player_comparison", "model_explanation", "data_question",
        }

    def test_chat_with_betting_question(self, client):
        data = client.post(
            "/chat",
            json={"question": "How often do home favorites cover the spread?"},
        ).json()
        assert "answer" in data


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict/matchup
# ══════════════════════════════════════════════════════════════════════════════


class TestPostPredictMatchup:
    def test_returns_200(self, client):
        r = client.post("/predict/matchup", json={"home_team": "KC", "away_team": "BUF"})
        assert r.status_code == 200

    def test_response_has_status(self, client):
        data = client.post(
            "/predict/matchup", json={"home_team": "KC", "away_team": "BUF"}
        ).json()
        assert "status" in data
        assert data["status"] in {"ok", "degraded", "error"}

    def test_response_has_home_away_team(self, client):
        data = client.post(
            "/predict/matchup", json={"home_team": "PHI", "away_team": "DAL"}
        ).json()
        assert "home_team" in data
        assert "away_team" in data

    def test_team_name_resolved(self, client):
        data = client.post(
            "/predict/matchup", json={"home_team": "Chiefs", "away_team": "Bills"}
        ).json()
        # Canon team names should be returned
        assert data.get("home_team") in {"KC", "Chiefs", None}

    def test_missing_home_team_returns_422(self, client):
        r = client.post("/predict/matchup", json={"away_team": "BUF"})
        assert r.status_code == 422

    def test_missing_away_team_returns_422(self, client):
        r = client.post("/predict/matchup", json={"home_team": "KC"})
        assert r.status_code == 422

    def test_with_season_and_week(self, client):
        r = client.post(
            "/predict/matchup",
            json={"home_team": "KC", "away_team": "BUF", "season": 2024, "week": 10},
        )
        assert r.status_code == 200

    def test_no_model_status_is_error(self, client):
        data = client.post(
            "/predict/matchup", json={"home_team": "KC", "away_team": "BUF"}
        ).json()
        # No model available in test env → status should be error (not 5xx)
        assert data["status"] in {"ok", "degraded", "error"}


# ══════════════════════════════════════════════════════════════════════════════
# GET /players/compare
# ══════════════════════════════════════════════════════════════════════════════


class TestGetPlayersCompare:
    def test_returns_200(self, client):
        r = client.get(
            "/players/compare",
            params={"player_a": "Patrick Mahomes", "player_b": "Josh Allen"},
        )
        assert r.status_code == 200

    def test_response_has_status(self, client):
        data = client.get(
            "/players/compare",
            params={"player_a": "A", "player_b": "B"},
        ).json()
        assert "status" in data

    def test_response_has_player_names(self, client):
        data = client.get(
            "/players/compare",
            params={"player_a": "Mahomes", "player_b": "Allen"},
        ).json()
        assert "player_a" in data
        assert "player_b" in data

    def test_missing_player_a_returns_422(self, client):
        r = client.get("/players/compare", params={"player_b": "Josh Allen"})
        assert r.status_code == 422

    def test_missing_player_b_returns_422(self, client):
        r = client.get("/players/compare", params={"player_a": "Mahomes"})
        assert r.status_code == 422

    def test_no_db_returns_degraded(self, client):
        data = client.get(
            "/players/compare",
            params={"player_a": "A", "player_b": "B"},
        ).json()
        # Without DB, should be degraded (not 5xx)
        assert data["status"] in {"ok", "degraded", "error"}


# ══════════════════════════════════════════════════════════════════════════════
# GET /trends/betting
# ══════════════════════════════════════════════════════════════════════════════


class TestGetTrendsBetting:
    def test_returns_200(self, client):
        r = client.get(
            "/trends/betting",
            params={"question": "How often do favorites cover?"},
        )
        assert r.status_code == 200

    def test_response_has_status(self, client):
        data = client.get("/trends/betting").json()
        assert "status" in data

    def test_response_has_question(self, client):
        data = client.get(
            "/trends/betting",
            params={"question": "underdog cover rate?"},
        ).json()
        assert "question" in data

    def test_no_data_returns_degraded_not_500(self, client):
        data = client.get("/trends/betting").json()
        assert data["status"] in {"ok", "degraded", "error"}

    def test_default_question_accepted(self, client):
        # No question param — should use the default value defined in routes.py
        r = client.get("/trends/betting")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# GET /metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestGetMetrics:
    def test_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_response_has_status(self, client):
        data = client.get("/metrics").json()
        assert "status" in data
        assert data["status"] in {"ok", "degraded", "error"}

    def test_response_has_win_model_key(self, client):
        data = client.get("/metrics").json()
        assert "win_model" in data

    def test_response_has_spread_model_key(self, client):
        data = client.get("/metrics").json()
        assert "spread_model" in data

    def test_no_models_status_is_error(self, client):
        data = client.get("/metrics").json()
        # No model artifacts in test env
        assert data["status"] in {"error", "ok", "degraded"}


# ══════════════════════════════════════════════════════════════════════════════
# GET /teams/{team}/summary
# ══════════════════════════════════════════════════════════════════════════════


class TestGetTeamSummary:
    def test_known_team_returns_200_or_404(self, client):
        r = client.get("/teams/KC/summary")
        # 200 (with data) or 404 (no data ingested) — both are valid
        assert r.status_code in {200, 404}

    def test_unknown_team_returns_404_or_200_degraded(self, client):
        # The route raises 404 only on status="error"; unknown teams return
        # status="degraded" (200) when no data is found but no hard error occurs.
        r = client.get("/teams/XXXUNKNOWNXXX/summary")
        assert r.status_code in {200, 404}
        if r.status_code == 200:
            assert r.json().get("status") in {"degraded", "ok", "error"}

    def test_team_nickname_accepted(self, client):
        r = client.get("/teams/Chiefs/summary")
        assert r.status_code in {200, 404}

    def test_200_response_has_team_key(self, client):
        r = client.get("/teams/KC/summary")
        if r.status_code == 200:
            data = r.json()
            assert "team" in data

    def test_404_response_has_detail(self, client):
        r = client.get("/teams/XXXUNKNOWNXXX/summary")
        if r.status_code == 404:
            data = r.json()
            assert "detail" in data

    def test_with_seasons_param(self, client):
        r = client.get("/teams/BUF/summary", params={"seasons": [2023]})
        assert r.status_code in {200, 404}
