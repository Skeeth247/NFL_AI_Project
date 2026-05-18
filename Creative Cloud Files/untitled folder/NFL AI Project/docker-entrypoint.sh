#!/bin/bash
# docker-entrypoint.sh — routes CMD to the right process.
set -euo pipefail

CMD="${1:-api}"

case "$CMD" in

  api)
    echo "Starting NFL AI Chatbot API on ${API_HOST}:${API_PORT}…"
    exec uvicorn nfl_chatbot.api.main:app \
      --host "${API_HOST}" \
      --port "${API_PORT}" \
      --workers "${API_WORKERS:-1}" \
      --log-level "${LOG_LEVEL:-info}"
    ;;

  app)
    echo "Starting NFL AI Chatbot Streamlit app on :${STREAMLIT_PORT}…"
    exec python -m streamlit run src/nfl_chatbot/app/streamlit_app.py \
      --server.port "${STREAMLIT_PORT}" \
      --server.address "0.0.0.0" \
      --server.headless true \
      --server.enableCORS false \
      --server.enableXsrfProtection true
    ;;

  init-db)
    echo "Initialising database…"
    exec python -m nfl_chatbot.cli init-db
    ;;

  ingest)
    echo "Running data ingestion…"
    exec python -m nfl_chatbot.cli ingest "${@:2}"
    ;;

  clean)
    exec python -m nfl_chatbot.cli clean "${@:2}"
    ;;

  features)
    exec python -m nfl_chatbot.cli features "${@:2}"
    ;;

  train)
    echo "Training models…"
    exec python -m nfl_chatbot.cli train "${@:2}"
    ;;

  evaluate)
    exec python -m nfl_chatbot.cli evaluate "${@:2}"
    ;;

  shell)
    exec /bin/bash
    ;;

  *)
    # Fall through: run any nfl-chatbot CLI command directly
    exec python -m nfl_chatbot.cli "$@"
    ;;

esac
