"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nfl_chatbot.config import (
    AppConfig,
    Config,
    DataConfig,
    EnvSettings,
    EvaluationConfig,
    ModelsConfig,
    ScrapingConfig,
    _load_yaml,
    get_config,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear the lru_cache before and after every test."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def minimal_yaml(tmp_path: Path) -> Path:
    """config.yaml with only seasons set."""
    cfg = {"data": {"seasons": [2022, 2023, 2024]}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture
def full_yaml(tmp_path: Path) -> Path:
    """config.yaml with all sections populated."""
    cfg = {
        "data": {
            "seasons": [2021, 2022, 2023, 2024],
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "external_dir": "data/external",
            "db_path": "data/db/nfl.db",
        },
        "features": {"rolling_window": 6},
        "models": {
            "train_seasons": [2021, 2022],
            "val_season": 2023,
            "test_season": 2024,
            "output_dir": "models",
            "win_predictor": {"type": "xgboost", "params": {"n_estimators": 300}},
            "ats_predictor": {"type": "xgboost", "params": {"n_estimators": 200}},
            "baseline": {"type": "logistic_regression", "params": {}},
        },
        "scraping": {
            "rate_limit_seconds": 5,
            "max_retries": 2,
            "timeout_seconds": 20,
            "respect_robots_txt": True,
            "sources": [
                {
                    "name": "pro_football_reference",
                    "base_url": "https://www.pro-football-reference.com",
                    "allowed": True,
                }
            ],
        },
        "evaluation": {
            "output_dir": "reports",
            "save_plots": False,
            "plot_format": "svg",
        },
        "database": {"url": "sqlite+aiosqlite:///data/db/nfl.db", "echo": True},
        "api": {"host": "127.0.0.1", "port": 9000, "reload": False},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture
def missing_yaml(tmp_path: Path) -> Path:
    """Path that does not exist."""
    return tmp_path / "nonexistent.yaml"


# ── Default behaviour ──────────────────────────────────────────────────────────


class TestDefaults:
    def test_default_seasons(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.seasons == [2020, 2021, 2022, 2023, 2024]

    def test_five_seasons(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert len(config.seasons) == 5

    def test_seasons_are_ints(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert all(isinstance(s, int) for s in config.seasons)

    def test_default_llm_provider(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.llm_provider == "anthropic"

    def test_raw_data_dir_is_absolute(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.raw_data_dir.is_absolute()

    def test_processed_data_dir_is_absolute(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.processed_data_dir.is_absolute()

    def test_db_path_is_absolute(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.db_path.is_absolute()

    def test_model_dir_is_absolute(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.model_dir.is_absolute()

    def test_win_model_path_is_absolute(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.win_model_path.is_absolute()

    def test_ats_model_path_is_absolute(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.ats_model_path.is_absolute()

    def test_api_keys_default_empty(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.anthropic_api_key == ""
        assert config.openai_api_key == ""

    def test_repr_contains_seasons(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert "seasons" in repr(config)


# ── YAML loading ───────────────────────────────────────────────────────────────


class TestYamlLoading:
    def test_seasons_overridden_by_yaml(self, minimal_yaml):
        config = Config(config_path=minimal_yaml)
        assert config.seasons == [2022, 2023, 2024]

    def test_full_yaml_seasons(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.seasons == [2021, 2022, 2023, 2024]

    def test_rolling_window_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.features.rolling_window == 6

    def test_model_train_seasons_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.models.train_seasons == [2021, 2022]

    def test_win_predictor_params_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.models.win_predictor.params["n_estimators"] == 300

    def test_scraping_rate_limit_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.scraping.rate_limit_seconds == 5

    def test_scraping_sources_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert len(config.app.scraping.sources) == 1
        assert config.app.scraping.sources[0].name == "pro_football_reference"

    def test_evaluation_plot_format_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.evaluation.plot_format == "svg"

    def test_database_echo_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.database.echo is True

    def test_api_port_from_yaml(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.app.api.port == 9000

    def test_missing_yaml_returns_defaults(self, missing_yaml):
        config = Config(config_path=missing_yaml)
        assert config.seasons == [2020, 2021, 2022, 2023, 2024]

    def test_relative_paths_resolved_to_absolute(self, full_yaml):
        config = Config(config_path=full_yaml)
        assert config.raw_data_dir.is_absolute()
        assert config.processed_data_dir.is_absolute()


# ── Environment variable overrides ────────────────────────────────────────────


class TestEnvOverrides:
    def test_seasons_override_from_env(self, monkeypatch, missing_yaml):
        monkeypatch.setenv("NFL_SEASONS", "2023,2024")
        env = EnvSettings()
        config = Config(config_path=missing_yaml, env_settings=env)
        assert config.seasons == [2023, 2024]

    def test_env_seasons_override_yaml_seasons(self, monkeypatch, minimal_yaml):
        monkeypatch.setenv("NFL_SEASONS", "2024")
        env = EnvSettings()
        config = Config(config_path=minimal_yaml, env_settings=env)
        assert config.seasons == [2024]

    def test_database_url_from_env(self, monkeypatch, missing_yaml):
        url = "sqlite+aiosqlite:///custom/nfl.db"
        monkeypatch.setenv("DATABASE_URL", url)
        env = EnvSettings()
        config = Config(config_path=missing_yaml, env_settings=env)
        assert config.database_url == url

    def test_llm_provider_from_env(self, monkeypatch, missing_yaml):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        env = EnvSettings()
        config = Config(config_path=missing_yaml, env_settings=env)
        assert config.llm_provider == "openai"

    def test_llm_model_openai_when_provider_is_openai(self, monkeypatch, missing_yaml):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
        env = EnvSettings()
        config = Config(config_path=missing_yaml, env_settings=env)
        assert config.llm_model == "gpt-4-turbo"

    def test_llm_model_anthropic_when_provider_is_anthropic(self, monkeypatch, missing_yaml):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
        env = EnvSettings()
        config = Config(config_path=missing_yaml, env_settings=env)
        assert config.llm_model == "claude-opus-4-7"

    def test_anthropic_api_key_from_env(self, monkeypatch, missing_yaml):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        env = EnvSettings()
        config = Config(config_path=missing_yaml, env_settings=env)
        assert config.anthropic_api_key == "sk-test-123"

    def test_empty_nfl_seasons_env_does_not_override(self, monkeypatch, minimal_yaml):
        monkeypatch.setenv("NFL_SEASONS", "")
        env = EnvSettings()
        config = Config(config_path=minimal_yaml, env_settings=env)
        # Should keep YAML value, not reset to empty
        assert config.seasons == [2022, 2023, 2024]


# ── _load_yaml helper ──────────────────────────────────────────────────────────


class TestLoadYaml:
    def test_returns_app_config(self, missing_yaml):
        result = _load_yaml(missing_yaml)
        assert isinstance(result, AppConfig)

    def test_empty_file_returns_defaults(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = _load_yaml(p)
        assert result.data.seasons == [2020, 2021, 2022, 2023, 2024]

    def test_partial_yaml_fills_remaining_with_defaults(self, tmp_path):
        p = tmp_path / "partial.yaml"
        p.write_text(yaml.dump({"features": {"rolling_window": 8}}))
        result = _load_yaml(p)
        assert result.features.rolling_window == 8
        assert result.data.seasons == [2020, 2021, 2022, 2023, 2024]


# ── Singleton cache ────────────────────────────────────────────────────────────


class TestGetConfig:
    def test_get_config_returns_config_instance(self):
        result = get_config()
        assert isinstance(result, Config)

    def test_get_config_is_cached(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_cache_clear_returns_new_instance(self):
        c1 = get_config()
        get_config.cache_clear()
        c2 = get_config()
        assert c1 is not c2


# ── Individual model validation ────────────────────────────────────────────────


class TestModelValidation:
    def test_data_config_defaults(self):
        dc = DataConfig()
        assert dc.seasons == [2020, 2021, 2022, 2023, 2024]
        assert dc.raw_dir.is_absolute()

    def test_data_config_seasons_custom(self):
        dc = DataConfig(seasons=[2023, 2024])
        assert dc.seasons == [2023, 2024]

    def test_models_config_defaults(self):
        mc = ModelsConfig()
        assert mc.val_season == 2023
        assert mc.test_season == 2024

    def test_scraping_config_defaults(self):
        sc = ScrapingConfig()
        assert sc.respect_robots_txt is True
        assert sc.rate_limit_seconds == 3

    def test_evaluation_config_has_all_metrics(self):
        ec = EvaluationConfig()
        assert "roc_auc" in ec.metrics
        assert "brier_score" in ec.metrics
        assert "log_loss" in ec.metrics
