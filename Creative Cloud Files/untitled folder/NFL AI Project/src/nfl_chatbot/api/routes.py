"""
Route handlers for the NFL Chatbot API.

All handlers use dependency injection to access the shared ChatbotTools and
LLMResponder instances initialised in the app lifespan.  Every handler returns
a typed Pydantic response model; unexpected exceptions bubble up to the global
500 handler registered in main.py.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from nfl_chatbot.api.schemas import (
    BettingTrendResponse,
    ChatRequest,
    ChatResponse,
    MatchupRequest,
    MatchupResponse,
    MetricsResponse,
    ModelMetrics,
    PlayerCompareResponse,
    TeamSummaryResponse,
)
from nfl_chatbot.chatbot.llm import LLMResponder
from nfl_chatbot.chatbot.responder import Responder
from nfl_chatbot.chatbot.tools import ChatbotTools

logger = logging.getLogger(__name__)

# ── Dependency helpers ────────────────────────────────────────────────────────


def get_tools(request: Request) -> ChatbotTools:
    return request.app.state.tools


def get_responder(request: Request) -> Responder:
    return request.app.state.responder


ToolsDep = Annotated[ChatbotTools, Depends(get_tools)]
ResponderDep = Annotated[Responder, Depends(get_responder)]


# ══════════════════════════════════════════════════════════════════════════════
# /chat
# ══════════════════════════════════════════════════════════════════════════════

chat_router = APIRouter(prefix="/chat", tags=["Chat"])


@chat_router.post(
    "",
    response_model=ChatResponse,
    summary="Ask the chatbot a natural-language NFL question",
    description=(
        "Classifies the intent, calls the appropriate data tool, and returns a "
        "plain-English answer.  Set `use_llm=true` to rewrite the answer through "
        "an LLM (requires an API key in the server environment)."
    ),
)
def post_chat(body: ChatRequest, responder: ResponderDep) -> ChatResponse:
    if body.use_llm and isinstance(responder, LLMResponder):
        resp = responder.respond(body.question)
    else:
        # Use the underlying rule-based responder directly
        base = responder if isinstance(responder, Responder) else responder._base
        resp = base.respond(body.question)

    return ChatResponse(
        answer=resp.answer,
        intent=resp.intent,
        confidence=resp.confidence,
        follow_ups=resp.follow_ups,
        has_caveat=resp.has_caveat,
        needs_clarification=resp.needs_clarification,
        tool_result=resp.tool_result,
    )


# ══════════════════════════════════════════════════════════════════════════════
# /predict
# ══════════════════════════════════════════════════════════════════════════════

predict_router = APIRouter(prefix="/predict", tags=["Predictions"])


@predict_router.post(
    "/matchup",
    response_model=MatchupResponse,
    summary="Predict the winner of an NFL matchup",
    description=(
        "Returns win probabilities, the predicted winner, and a plain-English "
        "explanation.  Requires a trained model — run "
        "`python scripts/train_model.py` first."
    ),
)
def post_predict_matchup(body: MatchupRequest, tools: ToolsDep) -> MatchupResponse:
    result = tools.predict_matchup(
        home_team=body.home_team,
        away_team=body.away_team,
        season=body.season,
        week=body.week,
    )

    prediction_raw = result.get("prediction") or {}
    from nfl_chatbot.api.schemas import PredictionDetail
    prediction = PredictionDetail(
        home_win_prob=prediction_raw.get("home_win_prob"),
        away_win_prob=prediction_raw.get("away_win_prob"),
        predicted_winner=prediction_raw.get("predicted_winner"),
        confidence=prediction_raw.get("confidence"),
    ) if prediction_raw else None

    return MatchupResponse(
        status=result["status"],
        error=result.get("error"),
        home_team=result.get("home_team"),
        away_team=result.get("away_team"),
        season=result.get("season"),
        week=result.get("week"),
        prediction=prediction,
        explanation=result.get("explanation"),
        features_available=result.get("features_available", True),
    )


# ══════════════════════════════════════════════════════════════════════════════
# /players
# ══════════════════════════════════════════════════════════════════════════════

players_router = APIRouter(prefix="/players", tags=["Players"])


@players_router.get(
    "/compare",
    response_model=PlayerCompareResponse,
    summary="Compare two NFL players side by side",
    description=(
        "Returns per-season and career aggregates with position-specific metrics "
        "and percentile ranks.  Requires player data to be ingested into the database."
    ),
)
def get_players_compare(
    tools: ToolsDep,
    player_a: Annotated[str, Query(description="First player name.", examples=["Patrick Mahomes"])],
    player_b: Annotated[str, Query(description="Second player name.", examples=["Josh Allen"])],
    seasons: Annotated[
        list[int] | None,
        Query(description="Filter to specific seasons (repeat for multiple).", examples=[2024]),
    ] = None,
) -> PlayerCompareResponse:
    result = tools.compare_players(player_a, player_b, seasons=seasons)
    return PlayerCompareResponse(
        status=result["status"],
        error=result.get("error"),
        player_a=result.get("player_a"),
        player_b=result.get("player_b"),
        seasons=result.get("seasons"),
        comparison=result.get("comparison"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# /trends
# ══════════════════════════════════════════════════════════════════════════════

trends_router = APIRouter(prefix="/trends", tags=["Betting Trends"])


@trends_router.get(
    "/betting",
    response_model=BettingTrendResponse,
    summary="Answer a betting trend question",
    description=(
        "Routes the question to the most relevant betting analysis function "
        "(cover rates, over/under, spread buckets, model edge, etc.).  "
        "All rates include Wilson 95% confidence intervals."
    ),
)
def get_trends_betting(
    tools: ToolsDep,
    question: Annotated[
        str,
        Query(
            description="Natural-language betting question.",
            examples=["How often do home favorites cover the spread?"],
        ),
    ] = "How often do favorites cover the spread?",
) -> BettingTrendResponse:
    result = tools.answer_betting_trend(question)
    return BettingTrendResponse(
        status=result["status"],
        error=result.get("error"),
        question=result.get("question"),
        trend_type=result.get("trend_type"),
        plain_summary=result.get("plain_summary"),
        result=result.get("result"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# /teams
# ══════════════════════════════════════════════════════════════════════════════

teams_router = APIRouter(prefix="/teams", tags=["Teams"])


@teams_router.get(
    "/{team}/summary",
    response_model=TeamSummaryResponse,
    summary="Get a team's season record and scoring averages",
    description=(
        "Returns win-loss record, points scored/allowed, and point differential "
        "by season.  Accepts any team name format: abbreviation ('KC'), "
        "nickname ('Chiefs'), or full name ('Kansas City Chiefs')."
    ),
)
def get_team_summary(
    team: str,
    tools: ToolsDep,
    seasons: Annotated[
        list[int] | None,
        Query(description="Filter to specific seasons (repeat for multiple).", examples=[2024]),
    ] = None,
) -> TeamSummaryResponse:
    result = tools.summarize_team(team, seasons=seasons)

    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result.get("error", "Team not found."))

    return TeamSummaryResponse(
        status=result["status"],
        error=result.get("error"),
        team=result.get("team"),
        seasons=result.get("seasons"),
        summary=result.get("summary"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# /metrics
# ══════════════════════════════════════════════════════════════════════════════

metrics_router = APIRouter(prefix="/metrics", tags=["Model Metrics"])


@metrics_router.get(
    "",
    response_model=MetricsResponse,
    summary="Get trained model performance metrics",
    description=(
        "Returns accuracy, ROC-AUC, Brier score, and a plain-English "
        "interpretation for the win predictor and ATS spread model."
    ),
)
def get_metrics(tools: ToolsDep) -> MetricsResponse:
    result = tools.get_model_metrics()

    def _parse_model(raw: dict | None) -> ModelMetrics | None:
        if raw is None:
            return None
        return ModelMetrics(
            available=raw.get("available", False),
            model_name=raw.get("model_name"),
            accuracy=raw.get("accuracy"),
            roc_auc=raw.get("roc_auc"),
            brier_score=raw.get("brier_score"),
            f1=raw.get("f1"),
            test_season=raw.get("test_season"),
            train_rows=raw.get("train_rows"),
            test_rows=raw.get("test_rows"),
            interpretation=raw.get("interpretation"),
            error=raw.get("error"),
        )

    return MetricsResponse(
        status=result["status"],
        error=result.get("error"),
        win_model=_parse_model(result.get("win_model")),
        spread_model=_parse_model(result.get("spread_model")),
        model_dir=result.get("model_dir"),
    )
