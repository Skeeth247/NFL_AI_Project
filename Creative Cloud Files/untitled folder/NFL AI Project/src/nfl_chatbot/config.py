"""
Configuration system for NFL AI Chatbot.

Loads config.yaml for project settings and .env for secrets/environment
overrides. Exposes a single typed Config object via get_config().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: src/nfl_chatbot/config.py → parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_SEASONS = [2020, 2021, 2022, 2023, 2024]


# ── YAML section models ────────────────────────────────────────────────────────


class NflverseConfig(BaseModel):
    datasets: list[str] = [
        "play_by_play",
        "player_stats",
        "schedules",
        "rosters",
        "injuries",
        "depth_charts",
        "team_stats",
    ]


class DataConfig(BaseModel):
    seasons: list[int] = Field(default_factory=lambda: list(_DEFAULT_SEASONS))
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    external_dir: Path = ROOT / "data" / "external"
    db_path: Path = ROOT / "data" / "db" / "nfl.db"
    nflverse: NflverseConfig = Field(default_factory=NflverseConfig)

    @field_validator("raw_dir", "processed_dir", "external_dir", "db_path", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


class EloConfig(BaseModel):
    k_factor: int = 20
    initial_rating: int = 1500
    home_advantage: int = 48
    regression_to_mean: float = 0.33


class FeaturesConfig(BaseModel):
    rolling_window: int = 4
    elo: EloConfig = Field(default_factory=EloConfig)


class ModelSpec(BaseModel):
    type: str = "xgboost"
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    train_seasons: list[int] = [2020, 2021, 2022]
    val_season: int = 2023
    test_season: int = 2024
    output_dir: Path = ROOT / "models"
    win_predictor: ModelSpec = Field(default_factory=lambda: ModelSpec(type="xgboost"))
    ats_predictor: ModelSpec = Field(default_factory=lambda: ModelSpec(type="xgboost"))
    baseline: ModelSpec = Field(
        default_factory=lambda: ModelSpec(type="logistic_regression")
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


class ChatbotConfig(BaseModel):
    max_tool_turns: int = 6
    system_prompt: str = (
        "You are an expert NFL analyst assistant. You answer questions about "
        "teams, players, game predictions, and betting trends using structured "
        "data tools. Never guess statistics — always call the appropriate tool. "
        "When presenting predictions, always include the model's confidence level "
        "and a brief caveat that predictions are probabilistic, not guaranteed."
    )


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///data/db/nfl.db"
    echo: bool = False
    tables: list[str] = Field(
        default_factory=lambda: [
            "schedules",
            "team_stats",
            "player_stats",
            "game_features",
            "predictions",
        ]
    )


class ScrapingSource(BaseModel):
    name: str
    base_url: str
    allowed: bool = True


class ScrapingConfig(BaseModel):
    rate_limit_seconds: int = 3
    max_retries: int = 3
    timeout_seconds: int = 30
    respect_robots_txt: bool = True
    sources: list[ScrapingSource] = Field(default_factory=list)


class CalibrationConfig(BaseModel):
    n_bins: int = 10
    strategy: str = "uniform"


class EvaluationConfig(BaseModel):
    output_dir: Path = ROOT / "reports"
    metrics: list[str] = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "log_loss",
    ]
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    save_plots: bool = True
    plot_format: str = "png"

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chatbot: ChatbotConfig = Field(default_factory=ChatbotConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ── Environment settings ───────────────────────────────────────────────────────


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"

    # Database (overrides config.yaml when set)
    database_url: str = ""

    # Odds API
    the_odds_api_key: str = ""

    # API server
    api_host: str = ""
    api_port: int = 0
    api_reload: bool = True

    # Scraping
    scrape_rate_limit_seconds: int = 0
    scrape_user_agent: str = "NFL-AI-Research-Bot/1.0 (educational project)"

    # Season override — comma-separated, e.g. "2022,2023,2024"
    nfl_seasons: str = ""

    # Model artifact paths
    model_dir: Path = ROOT / "models"
    win_model_path: Path = ROOT / "models" / "win_predictor.joblib"
    ats_model_path: Path = ROOT / "models" / "ats_predictor.joblib"

    @property
    def seasons_override(self) -> list[int] | None:
        if self.nfl_seasons.strip():
            return [int(s.strip()) for s in self.nfl_seasons.split(",")]
        return None


# ── Merged Config ──────────────────────────────────────────────────────────────


class Config:
    """
    Single access point for all project configuration.

    Priority (highest → lowest):
      1. Environment variables / .env file
      2. config.yaml
      3. Built-in defaults
    """

    def __init__(
        self,
        config_path: Path | None = None,
        env_settings: EnvSettings | None = None,
    ) -> None:
        self.env = env_settings or EnvSettings()
        self.app = _load_yaml(config_path or ROOT / "config.yaml")

        # Apply env overrides onto the parsed YAML config
        if self.env.seasons_override is not None:
            self.app.data.seasons = self.env.seasons_override

        if self.env.database_url:
            self.app.database.url = self.env.database_url

        if self.env.api_host:
            self.app.api.host = self.env.api_host

        if self.env.api_port:
            self.app.api.port = self.env.api_port

        if self.env.scrape_rate_limit_seconds:
            self.app.scraping.rate_limit_seconds = self.env.scrape_rate_limit_seconds

    # ── Data paths ─────────────────────────────────────────────────────────────

    @property
    def seasons(self) -> list[int]:
        return self.app.data.seasons

    @property
    def raw_data_dir(self) -> Path:
        return self.app.data.raw_dir

    @property
    def processed_data_dir(self) -> Path:
        return self.app.data.processed_dir

    @property
    def external_data_dir(self) -> Path:
        return self.app.data.external_dir

    @property
    def db_path(self) -> Path:
        return self.app.data.db_path

    @property
    def database_url(self) -> str:
        return self.app.database.url

    # ── Model artifacts ────────────────────────────────────────────────────────

    @property
    def model_dir(self) -> Path:
        return self.app.models.output_dir

    @property
    def win_model_path(self) -> Path:
        return self.env.win_model_path

    @property
    def ats_model_path(self) -> Path:
        return self.env.ats_model_path

    # ── API keys ───────────────────────────────────────────────────────────────

    @property
    def anthropic_api_key(self) -> str:
        return self.env.anthropic_api_key

    @property
    def openai_api_key(self) -> str:
        return self.env.openai_api_key

    @property
    def the_odds_api_key(self) -> str:
        return self.env.the_odds_api_key

    @property
    def llm_provider(self) -> str:
        return self.env.llm_provider

    @property
    def llm_model(self) -> str:
        if self.env.llm_provider == "openai":
            return self.env.openai_model
        return self.env.anthropic_model

    # ── Convenience ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Config(seasons={self.seasons}, provider={self.llm_provider}, "
            f"db={self.database_url!r})"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> AppConfig:
    """Parse config.yaml into a validated AppConfig. Returns defaults if missing."""
    if not path.exists():
        return AppConfig()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config instance (cached after first call)."""
    return Config()
