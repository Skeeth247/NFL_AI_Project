# NFL AI Chatbot

An end-to-end AI system for NFL game prediction, betting-trend analysis, and natural-language Q&A. Built as a full-stack data science and software engineering portfolio project.

The system ingests five seasons of historical NFL data, engineers 75+ leakage-audited features, trains calibrated ensemble classifiers, exposes predictions through a REST API, and surfaces everything through a conversational chatbot and Streamlit dashboard.

---

## Demo

> **Screenshots placeholder** — run `nfl-chatbot run-app` locally to see the live interface.

| Chatbot Q&A | Matchup Prediction | Betting Trends |
|---|---|---|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## Features

| Area | What it does |
|---|---|
| **Data pipeline** | Ingests schedules, rosters, player stats, injuries, and pre-game odds from nflverse (2020–2024). Cleans, deduplicates, and standardises team names across sources. |
| **Feature engineering** | Builds 75+ features across 7 groups: rolling team performance, season-to-date stats, Elo ratings, head-to-head history, rest/schedule context, game-context flags, and market-implied probability. All rolling features apply `shift(1)` before windowing — zero leakage by design. |
| **Leakage audit** | 25 automated checks (`nfl-chatbot evaluate`) verify no current-game data contaminates any feature. Checks cover rolling windows, target column exclusion, Elo pre-game safety, H2H date guards, chronological split integrity, and CV fold ordering. |
| **Model training** | Three calibrated classifiers (Logistic Regression, Random Forest, XGBoost) trained on a chronological split with `CalibratedClassifierCV` (isotonic regression). Best model selected by CV ROC-AUC. |
| **Win predictor** | `WinPredictor.predict()` returns win probabilities, predicted winner, confidence band, and a plain-English explanation of the top driving factors. |
| **ATS predictor** | Separate spread-coverage model using the same architecture. Spread convention follows nflverse: `spread_line < 0` means the home team is favoured. |
| **Betting trends** | `BettingTrends` computes favourite/underdog cover rates, over/under hit rates, team-level ATS records, and spread-bucket breakdowns — all with Wilson score 95% confidence intervals. |
| **Intent classification** | Rule-based regex classifier routes questions into 8 intents: `matchup_prediction`, `player_comparison`, `betting_trend`, `team_summary`, `model_explanation`, `data_question`, `general_help`, `unknown`. |
| **Chatbot responder** | `Responder` dispatches intents to typed tools, formats structured responses with follow-up suggestions and confidence caveats, and optionally routes through an LLM (Anthropic Claude or OpenAI GPT). |
| **REST API** | FastAPI backend with 6 router groups, Pydantic v2 request/response schemas, lifespan-managed state, CORS, structured logging, and graceful degraded responses when models or data are unavailable. |
| **Streamlit app** | Interactive frontend for chatting, comparing matchups, and exploring betting trends. |
| **CLI** | `nfl-chatbot` command covers the full pipeline: `init-db → ingest → clean → features → train → evaluate → chat / run-api / run-app`. |
| **Test suite** | 540+ pytest tests across 12 modules. Covers unit tests, integration tests against in-memory SQLite, API tests with `TestClient`, and model training smoke tests. |
| **Scraping compliance** | All HTTP scraping checks `robots.txt` before fetching, enforces a configurable rate limit (default 3 s), and identifies itself with a descriptive `User-Agent`. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | nflverse (Parquet via GitHub releases), nflreadpy |
| Storage | SQLite via SQLAlchemy 2.0 ORM |
| Feature engineering | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, joblib |
| Model calibration | `CalibratedClassifierCV` (isotonic regression) |
| API | FastAPI, uvicorn, Pydantic v2 |
| Frontend | Streamlit, Plotly |
| LLM integration | Anthropic Claude API, OpenAI API (optional) |
| Web scraping | requests, BeautifulSoup4, lxml |
| Testing | pytest, pytest-cov, httpx |
| Packaging | Hatchling (`pyproject.toml`) |
| Config | pyyaml + pydantic-settings + python-dotenv |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                            │
│  nflverse GitHub releases (schedules, rosters, player stats)    │
│  nflverse odds (pre-game spread lines and totals)               │
│  Injury reports (robots.txt-compliant scraping)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │  NflverseIngester / InjuryScraper
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite Database                              │
│  teams · games · player_stats · rosters                        │
│  injuries · odds · engineered_features                         │
└────────────────────────┬────────────────────────────────────────┘
                         │  SQLAlchemy 2.0 ORM
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Engineering                            │
│  FeatureBuilder                                                 │
│  ├── Rolling window features  (shift(1) before rolling — safe) │
│  ├── Season-expanding means   (shift(1) before expanding)      │
│  ├── Elo ratings              (stored before game update)       │
│  ├── Head-to-head history     (strict gameday < game_date)      │
│  ├── Rest / schedule context  (days since last game)            │
│  └── Market-implied prob      (from pre-game spread line)       │
│                                                                 │
│  LeakageChecker  ←  25 automated audit checks (CI-enforced)    │
└────────────────────────┬────────────────────────────────────────┘
                         │  modeling_table.csv
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Training                                │
│  Chronological split  (all seasons < test_season for train)    │
│  ├── Logistic Regression  (L2, calibrated)                      │
│  ├── Random Forest        (calibrated)                          │
│  └── XGBoost              (calibrated)                          │
│  Best model selected by CV ROC-AUC                             │
│  → best_model.joblib + feature_columns.json + metrics.json     │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌───────────────────────────────────┐
│      FastAPI Backend     │  │        Chatbot Layer              │
│  POST /chat              │  │  IntentClassifier (regex, 8 types)│
│  POST /predict/matchup   │  │  ChatbotTools     (7 typed tools) │
│  GET  /players/compare   │  │  Responder        (structured out)│
│  GET  /trends/betting    │  │  LLMClient        (Claude / GPT)  │
│  GET  /teams/{t}/summary │  └───────────────┬───────────────────┘
│  GET  /metrics           │                  │
└──────────────┬───────────┘                  │
               └──────────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Streamlit Frontend  │
                  │   nfl-chatbot run-app │
                  └───────────────────────┘
```

---

## Data Sources

| Source | What it provides | License |
|---|---|---|
| [nflverse](https://github.com/nflverse/nflverse-data) | Game schedules, team rosters, player stats (2020–2024) | CC BY-SA 4.0 |
| nflverse odds endpoint | Pre-game spread lines and totals | CC BY-SA 4.0 |
| Injury reports (scraped) | Weekly player availability status | robots.txt compliant; see note below |

All spread lines in this project are **pre-game closing lines** as distributed by nflverse. They are not derived from final scores or any post-game data.

---

## Setup

### Requirements

- Python 3.11+
- Git

### Install

```bash
git clone <repo-url>
cd nfl-ai-project

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package and all dependencies
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and add any optional API keys
```

`.env` keys (all optional — the system runs without LLM integration):

```env
ANTHROPIC_API_KEY=sk-ant-...     # Claude API (for --use-llm chat)
OPENAI_API_KEY=sk-...            # OpenAI fallback (optional)
DATABASE_URL=sqlite:///data/db/nfl.db   # default; override for PostgreSQL
```

---

## Running the Pipeline

All commands are available through the `nfl-chatbot` CLI after `pip install -e .`.

### 1 — Initialise the database

```bash
nfl-chatbot init-db
```

Creates all SQLite tables. Safe to re-run against an existing database.

### 2 — Ingest NFL data

```bash
# All datasets, all configured seasons (2020–2024 by default)
nfl-chatbot ingest

# Specific seasons and datasets
nfl-chatbot ingest --seasons 2023 2024 --datasets schedules player_stats
```

Downloads Parquet files from nflverse GitHub releases and upserts them into SQLite. First-run runtime: 2–5 minutes.

### 3 — Clean processed data

```bash
nfl-chatbot clean
```

Deduplicates, normalises team names to nflverse abbreviations, and validates schemas across all CSVs in `data/processed/`.

### 4 — Build the feature table

```bash
nfl-chatbot features

# Custom rolling window or season filter
nfl-chatbot features --window 6 --seasons 2022,2023,2024
```

Outputs `data/processed/modeling_table.csv` (75+ features, one row per game) and upserts a subset into the `engineered_features` table.

---

## Training the Model

```bash
# Train both win predictor and ATS model (default)
nfl-chatbot train

# Win model only, hold out 2024 as the test season
nfl-chatbot train --model win --test-season 2024

# Build features from the database and train in one step
nfl-chatbot train --from-db --model win
```

Artifacts saved to `models/`:

```
models/
├── best_model.joblib          ← best calibrated win predictor
├── feature_columns.json       ← ordered feature list
├── model_metrics.json         ← accuracy, ROC-AUC, Brier score per model
├── feature_importance.csv     ← top predictors ranked
└── spread_model.joblib        ← calibrated ATS predictor
```

---

## Auditing for Data Leakage

```bash
# Source-code checks (no data file required)
nfl-chatbot evaluate

# Full audit including file-based checks
nfl-chatbot evaluate --csv data/processed/modeling_table.csv

# Stop on first failure (CI-friendly)
nfl-chatbot evaluate --fail-fast
```

Runs 25 checks across 9 categories. Exits with code 1 on any FAIL:

```
ROLLING
  [PASS]  rolling_shift_excludes_current_game
  [PASS]  rolling_shift_sum_excludes_current
  [PASS]  season_expanding_excludes_current
  [PASS]  rolling_first_game_is_nan
  [PASS]  rolling_sort_order_safety

TARGET
  [PASS]  feature_columns_no_target
  [PASS]  training_forbidden_set_complete
  [PASS]  select_features_strips_forbidden

H2H
  [PASS]  h2h_strict_less_than
  [PASS]  h2h_null_gameday_fallback
  ...

  19 passed  0 failed  1 warned  4 skipped  (25 total)
AUDIT PASSED — no leakage detected.
```

---

## Running the Chatbot

### Command line

```bash
nfl-chatbot chat "Who wins Chiefs vs Bills this week?"
nfl-chatbot chat "How often do road underdogs cover the spread?"
nfl-chatbot chat "Compare Patrick Mahomes and Josh Allen"

# Route through an LLM for richer narrative answers (API key required)
nfl-chatbot chat "Explain KC's defensive tendencies" --use-llm
```

### REST API

```bash
nfl-chatbot run-api                            # localhost:8000
nfl-chatbot run-api --host 0.0.0.0 --port 8080 --reload
```

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Natural-language Q&A |
| `POST` | `/predict/matchup` | Win probability for any matchup |
| `GET` | `/players/compare` | Side-by-side player stat comparison |
| `GET` | `/trends/betting` | ATS and over/under trend analysis |
| `GET` | `/teams/{team}/summary` | Season stats and recent form |
| `GET` | `/metrics` | Trained model performance metrics |
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | Swagger UI (interactive) |

### Streamlit app

```bash
nfl-chatbot run-app                    # http://localhost:8501
nfl-chatbot run-app --port 8502 --no-browser
```

### Demo mode (no ingestion required)

```bash
# CLI — runs Streamlit with sample data for 8 teams (2022–2023)
nfl-chatbot run-app --demo

# Docker — fastest path from zero to live UI
docker run -p 8501:8501 -e NFL_DEMO_MODE=1 nfl-chatbot app
```

---

## Deployment

### Docker (single container)

```bash
# Build the image
docker build -t nfl-chatbot .

# Run the API on port 8000
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  nfl-chatbot api

# Run Streamlit on port 8501
docker run --rm -p 8501:8501 \
  -e NFL_DEMO_MODE=1 \
  nfl-chatbot app

# Mount your local data and models directories
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  nfl-chatbot api
```

### docker-compose (recommended for local development)

```bash
# Demo mode — zero config, sample data only
docker compose --profile demo up

# Production mode — API + Streamlit + shared volumes
docker compose up

# Run the full ingestion pipeline, then start services
docker compose --profile pipeline run --rm init-db
docker compose --profile pipeline run --rm ingest
docker compose --profile pipeline run --rm train
docker compose up
```

Copy `.env.example` to `.env` and set your API keys before running in production:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...          # optional
NFL_DEMO_MODE=0                # 0 = real data, 1 = sample data
```

### Render.com (free tier)

1. Fork the repository and connect it to your Render account.
2. Click **New → Blueprint** and point Render at `render.yaml` in the repo root.
3. Render creates two web services automatically: `nfl-api` (port 8000) and `nfl-app` (port 8501).
4. Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` via the Render dashboard **Environment** tab — never commit keys to the repo.
5. Both services start in demo mode (`NFL_DEMO_MODE=1`). For real predictions, provision persistent storage, run the ingestion pipeline, and set `NFL_DEMO_MODE=0`.

> **Note:** Render's free tier spins down after 15 minutes of inactivity. The first request after a cold start takes ~30 seconds. Upgrade to a paid plan for always-on deployments.

### Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/db/nfl.db` | SQLAlchemy connection string |
| `MODEL_DIR` | `models` | Directory containing trained `.joblib` artifacts |
| `NFL_DEMO_MODE` | `0` | `1` = demo mode, `0` = real data, absent = auto-detect |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `API_WORKERS` | `1` | Uvicorn worker count |
| `STREAMLIT_PORT` | `8501` | Streamlit port |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | *(empty)* | Claude API key — set at runtime only |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key — set at runtime only |
| `LOG_LEVEL` | `info` | Uvicorn log level (`debug`, `info`, `warning`, `error`) |

---

## Model Performance

Models are evaluated on a **strictly chronological hold-out** — no future-season data leaks into training.

| Model | Accuracy | ROC-AUC | Brier Score | CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

> Performance figures populate after running `nfl-chatbot train` against your local dataset. NFL win prediction is genuinely difficult — well-engineered models typically land in the 60–67% accuracy range. The naive baseline (always pick the home team) is ~57%.

**Top predictive features** (typical importance ranking):

1. `elo_diff` — pre-game Elo gap including home-field adjustment (+48 points)
2. `spread_line` — market-implied probability, the single most efficient signal
3. `implied_home_win_prob` — logit-transformed spread line
4. `home_pts_scored_avg4` / `away_pts_allowed_avg4` — rolling 4-game offensive and defensive efficiency
5. `rest_diff` — days-since-last-game advantage between teams
6. `h2h_home_win_rate` — head-to-head record over the last 5 meetings
7. `home_wins_last4` — recent form (rolling win count)

**Calibration:** all classifiers are wrapped in `CalibratedClassifierCV` (isotonic regression) so that predicted probabilities are reliable — a 65% prediction should be correct ~65% of the time across a large sample.

---

## Chatbot Evaluation

| Dimension | Approach |
|---|---|
| **Intent accuracy** | Regex classifier tested against a labelled question bank across 8 intent classes; precision and recall tracked per class |
| **Tool coverage** | All 7 `ChatbotTools` methods have unit tests; every missing-resource scenario is tested for graceful `status: degraded` response rather than an exception |
| **Response structure** | `Responder` returns a typed `Response` with `confidence`, `has_caveat`, `needs_clarification`, and `follow_ups` fields on every answer |
| **ATS trend confidence** | Wilson score 95% CIs on all cover rates; sample size and interval width are surfaced in every `BettingTrends` summary |
| **Leakage enforcement** | 25 automated checks; any FAIL exits with code 1, blocking model training in CI |
| **API contract** | 540+ pytest tests including `TestClient` integration tests for all 7 FastAPI endpoint groups and validation of 422 responses for malformed inputs |

---

## Project Structure

```
nfl-ai-project/
├── src/nfl_chatbot/
│   ├── api/            FastAPI app, routes, Pydantic schemas
│   ├── app/            Streamlit frontend + Plotly charts
│   ├── chatbot/        Intent classifier, tools, responder, LLM clients
│   ├── data/           SQLAlchemy schema, ingestors, cleaning pipeline
│   ├── evaluation/     BettingTrends, LeakageChecker (25 checks)
│   ├── features/       FeatureBuilder (75+ features, Elo, H2H)
│   ├── models/         ModelTrainer, WinPredictor, SpreadModel, explain
│   ├── scraping/       robots.txt-compliant scraper, injury parser
│   ├── cli.py          nfl-chatbot CLI (9 subcommands)
│   └── config.py       Typed config via pydantic-settings + YAML
├── scripts/            Standalone pipeline scripts (legacy entry points)
├── tests/              540+ pytest tests across 12 modules
├── data/
│   ├── raw/            Cached Parquet files from nflverse
│   ├── processed/      Cleaned CSVs + modeling_table.csv
│   └── db/             nfl.db (SQLite)
├── models/             Trained model artifacts (.joblib, .json)
├── notebooks/          Exploratory data analysis
├── Dockerfile          Two-stage build (deps → runtime, non-root user)
├── docker-compose.yml  API + Streamlit services, demo profile, pipeline helpers
├── docker-entrypoint.sh Routes container CMD to uvicorn / streamlit / CLI
├── render.yaml         Render.com blueprint (two web services)
└── pyproject.toml      Package config + nfl-chatbot CLI entry point
```

---

## Ethical Note

**This project is not gambling advice.**

Win probabilities and ATS cover rates are statistical model outputs trained on historical data. They reflect patterns in past games and carry inherent uncertainty — no model can reliably predict NFL outcomes. Nothing produced by this project should be used to inform financial decisions. Betting-trend analysis is provided for educational and analytical purposes only.

---

## Scraping Compliance

All HTTP scraping follows responsible web access practices:

- **`robots.txt` is checked first.** `RobotsChecker` inspects the disallow rules for each domain before any HTTP request is made. Any disallowed URL raises `ScrapingNotAllowedError` and is skipped without fetching.
- **Rate limiting is enforced.** A minimum gap of 3 seconds (configurable) is maintained between requests to any domain.
- **Descriptive `User-Agent` is set** on all requests so site operators can identify and contact the requester.
- **HTML caching** avoids refetching unchanged pages.

The primary data source (nflverse) is fetched as static Parquet files directly from GitHub releases — no live-site scraping is required for the core data pipeline.

---

## Future Improvements

- [ ] **Player-level features** — incorporate individual EPA, target share, and snap-count-weighted efficiency as game-level aggregates
- [ ] **Opponent-adjusted metrics** — strength-of-schedule normalisation for rolling offensive and defensive stats
- [ ] **Play-by-play features** — third-down conversion rate, red-zone efficiency, and pressure rate from nflverse PBP data
- [ ] **Weather integration** — wind speed, temperature, and precipitation as game-context features
- [ ] **Live injury feed** — automated weekly injury report ingestion with `report_date < gameday` enforcement
- [ ] **PostgreSQL support** — the SQLAlchemy ORM is DB-agnostic; production deployment requires only a `DATABASE_URL` change
- [ ] **Model versioning** — MLflow or DVC for experiment tracking and artifact lineage
- [ ] **LLM tool-calling** — replace the regex intent classifier with a Claude/GPT function-calling loop for richer multi-turn conversations
- [ ] **CI/CD** — GitHub Actions pipeline: lint → test → leakage audit → model retrain on season-start data

---

## Resume Bullet Suggestions

Pick the bullets that best match the role you are applying for.

**Software Engineering / Backend**
- Designed and shipped a production-quality Python package (`nfl-chatbot`) with a 9-subcommand CLI, FastAPI REST API, and Pydantic v2 request validation; enforced clean architecture with lazy-import handlers and typed configuration via pydantic-settings
- Built a `robots.txt`-compliant web scraper with configurable rate limiting and HTML caching; raised `ScrapingNotAllowedError` on disallowed URLs before any HTTP call was made
- Wrote 540+ pytest tests across unit, integration (in-memory SQLite via SQLAlchemy), and API layers; achieved `TestClient` coverage of all 7 FastAPI endpoint groups including 422 validation paths

**Data Engineering / MLOps**
- Ingested and standardised 5 seasons of NFL game, roster, and player data from nflverse Parquet releases into SQLite via a SQLAlchemy 2.0 ORM pipeline with idempotent upserts and configurable season filters
- Implemented a 25-check automated data-leakage audit (`nfl-chatbot evaluate`) covering rolling-window self-inclusion, target column exclusion, Elo pre-game safety, H2H date guards, and chronological CV fold ordering; exits non-zero on any FAIL for CI enforcement
- Engineered 75+ features across 7 groups using `shift(1).rolling(N)` and `shift(1).expanding()` transforms to guarantee zero target leakage; verified correctness with canary-value functional tests

**Data Science / Machine Learning**
- Trained and calibrated three classifiers (Logistic Regression, Random Forest, XGBoost) on a strictly chronological train/test split with `CalibratedClassifierCV` (isotonic regression) for reliable probability estimates; selected best model by season-fold cross-validated ROC-AUC
- Built a `BettingTrends` analytics module computing ATS cover rates, over/under hit rates, and spread-bucket breakdowns with Wilson score 95% confidence intervals across 5 NFL seasons
- Developed a plain-English prediction explainer that maps feature importance rankings to human-readable narratives — top factors, positive/negative drivers, and confidence band — for every model inference

**AI / NLP**
- Built a conversational NFL Q&A system with structured `Responder` output (answer, intent, confidence, follow-up questions, caveat flag); designed fallback-safe tool responses that return `status: degraded` rather than HTTP 5xx when data or models are unavailable
- Integrated Anthropic Claude and OpenAI APIs as optional LLM backends with automatic fallback to a rule-based responder when API keys are absent; all credentials loaded exclusively from environment variables

---

## License

MIT — see [LICENSE](LICENSE).

---

*Data provided by [nflverse](https://github.com/nflverse/nflverse-data) under CC BY-SA 4.0.*
