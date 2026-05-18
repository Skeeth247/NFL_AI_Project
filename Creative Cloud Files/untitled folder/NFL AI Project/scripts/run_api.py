#!/usr/bin/env python3
"""
Launch the NFL Chatbot API server.

Usage
-----
    python scripts/run_api.py                        # dev server, localhost:8000
    python scripts/run_api.py --host 0.0.0.0         # expose on all interfaces
    python scripts/run_api.py --port 8080 --reload   # auto-reload on file changes
    python scripts/run_api.py --workers 4            # production multi-process

The server starts at:
  http://localhost:8000/docs       Swagger UI
  http://localhost:8000/redoc      ReDoc
  http://localhost:8000/health     Health check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the src/ directory is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NFL AI Chatbot — API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only, overrides --workers).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (ignored when --reload is set).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed.  Run: pip install uvicorn[standard]")
        sys.exit(1)

    workers = 1 if args.reload else args.workers

    print(f"\n  NFL AI Chatbot API")
    print(f"  http://{args.host}:{args.port}/docs   ← Swagger UI")
    print(f"  http://{args.host}:{args.port}/health ← Health check\n")

    uvicorn.run(
        "nfl_chatbot.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
