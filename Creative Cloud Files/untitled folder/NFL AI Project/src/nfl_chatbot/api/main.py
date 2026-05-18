"""
NFL Chatbot FastAPI application.

Startup
-------
    uvicorn nfl_chatbot.api.main:app --reload

Or via the helper script:
    python scripts/run_api.py

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
OpenAPI JSON: http://localhost:8000/openapi.json
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from nfl_chatbot.api.routes import (
    chat_router,
    metrics_router,
    players_router,
    predict_router,
    teams_router,
    trends_router,
)
from nfl_chatbot.api.schemas import HealthResponse
from nfl_chatbot.chatbot.llm import LLMResponder
from nfl_chatbot.chatbot.tools import ChatbotTools

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
API_TITLE = "NFL AI Chatbot API"
API_DESCRIPTION = """
Predictive analytics and natural-language Q&A for NFL games.

## Features
- **Chat** — Ask any NFL question in plain English
- **Matchup predictions** — Win probabilities from a trained ML model
- **Player comparisons** — Side-by-side career and season stats
- **Betting trends** — Historical cover rates, over/under analysis
- **Team summaries** — Season records and scoring averages
- **Model metrics** — Accuracy and calibration of the prediction models

## Notes
- Predictions are statistical estimates, **not betting advice**
- Run `python scripts/train_model.py` before using `/predict/matchup`
- Run `python scripts/ingest.py` before using `/players/compare` or `/teams/{team}/summary`
"""

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources once at startup."""
    logger.info("Starting NFL Chatbot API v%s", API_VERSION)
    tools = ChatbotTools()
    app.state.tools = tools
    # LLMResponder wraps rule-based Responder; falls back if no API key set
    app.state.responder = LLMResponder(tools=tools)
    logger.info(
        "Chatbot initialised.  LLM active: %s",
        app.state.responder.using_llm,
    )
    yield
    logger.info("NFL Chatbot API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "NFL AI Chatbot",
            "url": "https://github.com/SakethBoddu247",
        },
        license_info={"name": "MIT"},
    )

    _add_cors(app)
    _add_logging_middleware(app)
    _add_exception_handlers(app)
    _add_routers(app)
    _add_health_route(app)

    return app


# ── CORS ──────────────────────────────────────────────────────────────────────


def _add_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # tighten for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ── Request logging middleware ────────────────────────────────────────────────


def _add_logging_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


# ── Exception handlers ────────────────────────────────────────────────────────


def _add_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(ValidationError)
    async def pydantic_validation_error(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"status": "error", "error": "Validation error", "detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": "Internal server error",
                "detail": str(exc),
            },
        )


# ── Routers ───────────────────────────────────────────────────────────────────


def _add_routers(app: FastAPI) -> None:
    app.include_router(chat_router)
    app.include_router(predict_router)
    app.include_router(players_router)
    app.include_router(trends_router)
    app.include_router(teams_router)
    app.include_router(metrics_router)


# ── Health endpoint ───────────────────────────────────────────────────────────


def _add_health_route(app: FastAPI) -> None:

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="API health check",
        description="Returns 200 whenever the server is running.",
    )
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=API_VERSION,
            timestamp=datetime.now(timezone.utc),
        )


# ── Module-level app instance ─────────────────────────────────────────────────

app = create_app()
