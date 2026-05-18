# ── Stage 1: dependency layer ──────────────────────────────────────────────────
# Separate stage so the heavy pip install is cached between code-only changes.
FROM python:3.11-slim AS deps

WORKDIR /build

# System packages needed by lxml, pyarrow, and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the packaging manifests first (maximise cache hits)
COPY pyproject.toml README.md ./
COPY src/nfl_chatbot/__init__.py src/nfl_chatbot/__init__.py

# Install the package with all runtime dependencies (no dev extras)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e "."


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="NFL AI Chatbot"
LABEL org.opencontainers.image.description="NFL game prediction, betting trends, and chatbot"
LABEL org.opencontainers.image.source="https://github.com/SakethBoddu247/nfl-ai-chatbot"

WORKDIR /app

# Runtime system deps only (libgomp for XGBoost inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pull installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ src/
COPY config.yaml .
COPY data/demo/ data/demo/

# Runtime directories — real data is mounted as volumes
RUN mkdir -p data/raw data/processed data/external data/db models

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Environment defaults ───────────────────────────────────────────────────────
# These can all be overridden at runtime via -e / docker-compose / render.yaml.

ENV DATABASE_URL="sqlite:///data/db/nfl.db"
ENV MODEL_DIR="models"
ENV NFL_DEMO_MODE="0"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"
ENV STREAMLIT_PORT="8501"
ENV LLM_PROVIDER="anthropic"
ENV ANTHROPIC_API_KEY=""
ENV OPENAI_API_KEY=""
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

# Streamlit: disable telemetry and avoid browser-open in headless containers
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
ENV STREAMLIT_SERVER_HEADLESS="true"
ENV STREAMLIT_SERVER_ENABLE_CORS="false"

EXPOSE 8000
EXPOSE 8501

# ── Entrypoint scripts ─────────────────────────────────────────────────────────
# Override CMD at runtime:
#   docker run ... api         → FastAPI on :8000
#   docker run ... app         → Streamlit on :8501
#   docker run ... init-db     → initialise the database
#   docker run ... ingest      → run data ingestion
#   docker run ... train       → train models

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# entrypoint is copied before USER switch so we chown it now
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
 && chown appuser:appuser /usr/local/bin/docker-entrypoint.sh
USER appuser

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["api"]
