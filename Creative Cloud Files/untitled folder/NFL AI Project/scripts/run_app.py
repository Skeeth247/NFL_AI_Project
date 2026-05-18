#!/usr/bin/env python3
"""
Launch the NFL Chatbot Streamlit app.

Usage
-----
    python scripts/run_app.py
    python scripts/run_app.py --port 8502 --no-browser
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NFL AI Chatbot — Streamlit app",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app_path = Path(__file__).parent.parent / "src" / "nfl_chatbot" / "app" / "streamlit_app.py"
    if not app_path.exists():
        print(f"App not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
    ]

    print(f"\n  NFL AI Chatbot — Streamlit App")
    print(f"  http://localhost:{args.port}\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
