.PHONY: install lint format test init-db ingest ingest-force ingest-odds features train evaluate api app clean help

PYTHON := python3
PIP    := pip3

help:
	@echo "NFL AI Chatbot — available commands:"
	@echo ""
	@echo "  install    Install all dependencies (prod + dev)"
	@echo "  lint       Run ruff and mypy"
	@echo "  format     Auto-format with black and ruff --fix"
	@echo "  test       Run pytest with coverage"
	@echo "  init-db    Create all database tables (safe to re-run)"
	@echo "  ingest     Download and load NFL data into SQLite"
	@echo "  features   Build feature tables from ingested data"
	@echo "  train      Train win and ATS prediction models"
	@echo "  evaluate   Run evaluation suite and print metrics"
	@echo "  api        Start FastAPI backend on :8000"
	@echo "  app        Start Streamlit frontend on :8501"
	@echo "  clean      Remove build artifacts and caches"

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

test:
	pytest

init-db:
	$(PYTHON) scripts/init_db.py

ingest:
	$(PYTHON) scripts/ingest_nfl_data.py

ingest-force:
	$(PYTHON) scripts/ingest_nfl_data.py --force

ingest-odds:
	$(PYTHON) scripts/ingest_odds.py

features:
	$(PYTHON) scripts/build_features.py

train:
	$(PYTHON) scripts/train_models.py

evaluate:
	$(PYTHON) scripts/evaluate_models.py

api:
	uvicorn nfl_chatbot.api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run src/nfl_chatbot/app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache dist build
