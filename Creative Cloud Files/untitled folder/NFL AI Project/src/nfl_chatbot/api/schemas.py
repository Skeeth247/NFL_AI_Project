"""
Pydantic request and response models for the NFL Chatbot API.

All models use Pydantic v2 syntax.  Response envelopes share a common
``status`` / ``error`` wrapper so clients have a consistent shape to parse.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Shared envelope ───────────────────────────────────────────────────────────


class APIError(BaseModel):
    """Returned for 4xx / 5xx responses."""

    status: str = "error"
    error: str
    detail: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Always 'ok' when the server is up.")
    version: str = Field(..., description="API version string.")
    timestamp: datetime = Field(..., description="Server time (UTC).")

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": "2026-01-01T12:00:00",
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════


class ModelMetrics(BaseModel):
    available: bool
    model_name: str | None = None
    accuracy: float | None = None
    roc_auc: float | None = None
    brier_score: float | None = None
    f1: float | None = None
    test_season: int | None = None
    train_rows: int | None = None
    test_rows: int | None = None
    interpretation: str | None = None
    error: str | None = None


class MetricsResponse(BaseModel):
    status: str
    error: str | None = None
    win_model: ModelMetrics | None = None
    spread_model: ModelMetrics | None = None
    model_dir: str | None = None

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "error": None,
        "win_model": {
            "available": True,
            "model_name": "random_forest",
            "accuracy": 0.638,
            "roc_auc": 0.682,
            "interpretation": "64% accuracy — moderate performance, typical for NFL prediction.",
        },
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Chat
# ══════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural-language NFL question.",
        examples=["Who wins Chiefs vs Bills?"],
    )
    use_llm: bool = Field(
        False,
        description=(
            "If True and an LLM API key is configured, the answer is rewritten "
            "by the LLM for more natural prose.  Falls back to rule-based if no key."
        ),
    )

    model_config = {"json_schema_extra": {"example": {
        "question": "Who wins Chiefs vs Bills?",
        "use_llm": False,
    }}}


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Natural-language response.")
    intent: str = Field(..., description="Classified user intent.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent classification confidence.")
    follow_ups: list[str] = Field(default_factory=list, description="Suggested follow-up questions.")
    has_caveat: bool = Field(False, description="True if the answer includes a data-quality caveat.")
    needs_clarification: bool = Field(False, description="True if the chatbot needs more info.")
    tool_result: dict[str, Any] | None = Field(None, description="Raw structured data behind the answer.")

    model_config = {"json_schema_extra": {"example": {
        "answer": "The Eagles have a 61% win probability over the Cowboys.",
        "intent": "matchup_prediction",
        "confidence": 0.87,
        "follow_ups": ["Ask me to explain why I picked the Eagles."],
        "has_caveat": False,
        "needs_clarification": False,
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Predict matchup
# ══════════════════════════════════════════════════════════════════════════════


class MatchupRequest(BaseModel):
    home_team: str = Field(
        ...,
        description="Home team name or abbreviation (e.g. 'KC', 'Chiefs', 'Kansas City Chiefs').",
        examples=["KC"],
    )
    away_team: str = Field(
        ...,
        description="Away team name or abbreviation.",
        examples=["BUF"],
    )
    season: int | None = Field(
        None,
        ge=2000,
        le=2030,
        description="NFL season year.  Omit to use most recent data.",
        examples=[2024],
    )
    week: int | None = Field(
        None,
        ge=1,
        le=22,
        description="Week number (1–22 including playoffs).",
        examples=[12],
    )

    model_config = {"json_schema_extra": {"example": {
        "home_team": "KC",
        "away_team": "BUF",
        "season": 2024,
        "week": 12,
    }}}


class PredictionDetail(BaseModel):
    home_win_prob: float | None = None
    away_win_prob: float | None = None
    predicted_winner: str | None = None
    confidence: str | None = None


class MatchupResponse(BaseModel):
    status: str
    error: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    season: int | None = None
    week: int | None = None
    prediction: PredictionDetail | None = None
    explanation: dict[str, Any] | None = None
    features_available: bool = True

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "home_team": "KC",
        "away_team": "BUF",
        "prediction": {
            "home_win_prob": 0.58,
            "away_win_prob": 0.42,
            "predicted_winner": "KC",
            "confidence": "medium",
        },
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Players compare
# ══════════════════════════════════════════════════════════════════════════════


class PlayerCompareResponse(BaseModel):
    status: str
    error: str | None = None
    player_a: str | None = None
    player_b: str | None = None
    seasons: list[int] | None = None
    comparison: dict[str, Any] | None = None

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "player_a": "Patrick Mahomes",
        "player_b": "Josh Allen",
        "comparison": {"position": "QB"},
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Betting trends
# ══════════════════════════════════════════════════════════════════════════════


class BettingTrendResponse(BaseModel):
    status: str
    error: str | None = None
    question: str | None = None
    trend_type: str | None = None
    plain_summary: str | None = None
    result: dict[str, Any] | None = None

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "trend_type": "underdog_cover_rate",
        "plain_summary": "Underdogs cover the spread 48.2% of the time overall.",
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Team summary
# ══════════════════════════════════════════════════════════════════════════════


class TeamSummaryResponse(BaseModel):
    status: str
    error: str | None = None
    team: str | None = None
    seasons: list[int] | None = None
    summary: dict[str, Any] | None = None

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "team": "PHI",
        "summary": {
            "overall": {"wins": 11, "losses": 6, "win_pct": 0.647},
        },
    }}}
