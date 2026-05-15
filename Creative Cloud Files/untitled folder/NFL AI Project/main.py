"""Entry point for the NFL AI Chatbot project."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nfl-chatbot",
        description="NFL AI Chatbot — prediction, analysis, and conversational interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("api", help="Start the FastAPI backend server")
    subparsers.add_parser("ui", help="Start the Streamlit frontend")
    subparsers.add_parser("ingest", help="Download and load NFL data into SQLite")
    subparsers.add_parser("train", help="Train and evaluate prediction models")

    args = parser.parse_args()

    if args.command == "api":
        import uvicorn
        uvicorn.run(
            "nfl_chatbot.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )

    elif args.command == "ui":
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/nfl_chatbot/app/app.py"],
            check=True,
        )

    elif args.command == "ingest":
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/ingest_nfl_data.py"],
            check=True,
        )

    elif args.command == "train":
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/train_models.py"],
            check=True,
        )


if __name__ == "__main__":
    main()
