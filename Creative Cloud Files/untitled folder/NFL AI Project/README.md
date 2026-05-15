# NFL AI Chatbot

An AI-powered chatbot that analyzes five seasons of NFL data (2020–2024) to answer questions about game predictions, player comparisons, matchup history, and betting trends.

## Features

- **Game winner prediction** — XGBoost classifier trained on team stats, Elo ratings, rest, and situational factors
- **Against-the-spread (ATS) prediction** — separate model targeting spread coverage
- **Conversational interface** — LLM-backed chatbot that calls structured data tools (never hallucinates stats)
- **Honest evaluation** — all metrics (accuracy, ROC-AUC, Brier score, calibration) computed on a held-out 2024 test season
- **Pluggable LLM layer** — swap between Anthropic Claude, OpenAI GPT, or a local model via one env var

## Model Performance

> Metrics are populated after running `make train`. No results are fabricated.

| Model | Accuracy | ROC-AUC | Brier Score |
|---|---|---|---|
| Logistic Regression (baseline) | — | — | — |
| XGBoost Win Predictor | — | — | — |
| XGBoost ATS Predictor | — | — | — |

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/Skeeth247/NFL_AI_Project.git
cd NFL_AI_Project
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Download and load NFL data (2020–2024)
make ingest

# 5. Train and evaluate models
make train

# 6. Start the API backend
make serve-api

# 7. Start the Streamlit frontend (separate terminal)
make serve-ui
```

## Project Structure

```
nfl-ai-chatbot/
├── src/nfl_chatbot/
│   ├── data/          # nflverse loader, ingestion pipeline
│   ├── features/      # team stats, Elo, betting features
│   ├── models/        # win predictor, ATS predictor, evaluator
│   ├── chatbot/       # LLM providers, tool registry, agent loop
│   ├── api/           # FastAPI routes and schemas
│   ├── app/           # Streamlit frontend
│   ├── scraping/      # robots.txt-compliant scraper
│   └── evaluation/    # metrics and calibration plots
├── data/              # raw, processed, db (git-ignored)
├── tests/             # pytest suite
├── scripts/           # ingest_data.py, train_models.py
├── notebooks/         # exploration and experiments
└── docs/              # architecture and API docs
```

## Configuration

All project settings live in `config.yaml`. Secrets go in `.env` (never committed).

Key config options:

| Key | Default | Description |
|---|---|---|
| `data.seasons` | `[2020..2024]` | Seasons to ingest |
| `features.rolling_window` | `4` | Games for rolling averages |
| `models.train_seasons` | `[2020,2021,2022]` | Training set |
| `models.val_season` | `2023` | Validation set |
| `models.test_season` | `2024` | Held-out test set |
| `chatbot.max_tool_turns` | `6` | Max LLM tool-call iterations |

## Available Commands

```bash
make install      # Install production dependencies
make install-dev  # Install with dev tools
make test         # Run pytest with coverage
make lint         # ruff + mypy
make format       # black + ruff --fix
make ingest       # Download and load all NFL data
make train        # Train and evaluate models
make serve-api    # FastAPI on :8000
make serve-ui     # Streamlit on :8501
make clean        # Remove build artifacts
```

## Data Sources

- **nflreadpy / nflverse** — play-by-play, schedules, rosters, injuries (2020–2024)
- **Pro Football Reference** — season-level team stats (scraped with rate limiting, robots.txt verified)
- **The Odds API** — historical betting lines and spreads

## Tech Stack

Python 3.11 · Pandas · NumPy · scikit-learn · XGBoost · FastAPI · Streamlit · SQLite · SQLAlchemy · Anthropic Claude · BeautifulSoup4

## License

MIT
