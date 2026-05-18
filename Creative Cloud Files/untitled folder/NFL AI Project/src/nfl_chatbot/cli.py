"""
NFL AI Chatbot — command-line interface.

Entry point: ``nfl-chatbot`` (registered in pyproject.toml).

Commands
--------
  init-db     Create / verify database tables
  ingest      Download NFL data from nflverse into the database
  clean       Clean processed CSV files
  features    Build the modeling feature table
  train       Train win and/or ATS prediction models
  evaluate    Run data-leakage audit checks
  chat        Ask a question and print the answer
  run-api     Start the FastAPI server (uvicorn)
  run-app     Start the Streamlit frontend
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # <project-root>/src/nfl_chatbot/cli.py


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Subcommand handlers
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_init_db(args: argparse.Namespace) -> int:
    _setup_logging()
    from nfl_chatbot.config import get_config
    from nfl_chatbot.data.database import get_engine, init_db, table_row_counts
    from nfl_chatbot.data.schema import ALL_TABLES

    cfg = get_config()
    logging.info("Database : %s", cfg.database_url)

    engine = get_engine(cfg.database_url)
    created = init_db(engine)

    print("\n── Table status ─────────────────────────────────────")
    for name in ALL_TABLES:
        status = "OK" if name in created else "MISSING"
        print(f"  {status:8s}  {name}")

    counts = table_row_counts(engine)
    print("\n── Row counts ───────────────────────────────────────")
    for name, count in counts.items():
        print(f"  {count:>10,d}  {name}")

    print(f"\nDatabase ready: {cfg.db_path}\n")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    _setup_logging(getattr(args, "verbose", False))
    import time

    from nfl_chatbot.config import get_config
    from nfl_chatbot.data.database import get_engine, init_db, table_row_counts
    from nfl_chatbot.data.ingest_nflverse import NflverseIngester

    _ALL_DATASETS = ["teams", "schedules", "player_stats", "rosters"]

    cfg = get_config()
    seasons: list[int] = args.seasons or cfg.seasons
    datasets: list[str] = args.datasets or _ALL_DATASETS

    print(f"\n  Seasons  : {seasons}")
    print(f"  Datasets : {datasets}")
    print(f"  Database : {cfg.database_url}\n")

    engine = get_engine(cfg.database_url)
    init_db(engine)

    if args.force:
        for p in cfg.raw_data_dir.glob("*.parquet"):
            p.unlink()
        for p in cfg.raw_data_dir.glob("*.csv"):
            p.unlink()
        logging.info("Cleared cached raw files")

    ingester = NflverseIngester(
        seasons=seasons,
        raw_dir=cfg.raw_data_dir,
        processed_dir=cfg.processed_data_dir,
        engine=engine,
        rate_limit=cfg.app.scraping.rate_limit_seconds,
    )

    t0 = time.perf_counter()
    results: dict = {}
    for name in datasets:
        method = {
            "teams":        ingester.ingest_teams,
            "schedules":    ingester.ingest_schedules,
            "player_stats": ingester.ingest_player_stats,
            "rosters":      ingester.ingest_rosters,
        }[name]
        results[name] = ingester._run(name, method)  # noqa: SLF001

    elapsed = time.perf_counter() - t0

    print("── Ingestion summary ────────────────────────────────────")
    for name, r in results.items():
        status = "OK" if r.ok else f"FAILED — {r.error[:60]}"
        print(f"  {name:<20} {r.rows_downloaded:>10,d} downloaded  {r.rows_inserted:>10,d} inserted  {status}")

    counts = {k: v for k, v in table_row_counts(engine).items() if v > 0}
    if counts:
        print("\n── Database row counts ───────────────────────────────────")
        for tbl, cnt in sorted(counts.items()):
            print(f"  {cnt:>10,d}  {tbl}")

    n_ok = sum(1 for r in results.values() if r.ok)
    print(f"\nCompleted {n_ok}/{len(results)} datasets in {elapsed:.1f}s\n")
    return 0 if n_ok == len(results) else 1


def _cmd_clean(args: argparse.Namespace) -> int:
    _setup_logging(getattr(args, "verbose", False))
    from nfl_chatbot.data.cleaning import run_pipeline

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        logging.error("Directory not found: %s", processed_dir)
        return 1

    seasons: list[int] | None = None
    if args.seasons.strip():
        try:
            seasons = [int(s.strip()) for s in args.seasons.split(",")]
        except ValueError:
            logging.error("Invalid --seasons value: %r", args.seasons)
            return 1

    logging.info("Cleaning source: %s", processed_dir)
    reports = run_pipeline(processed_dir, seasons=seasons)

    if not reports:
        logging.warning("No CSV files found in %s", processed_dir)
        return 0

    print("\n── Cleaning summary ─────────────────────────────────────")
    print(f"  {'Dataset':<20} {'In':>7} {'Out':>7} {'Dupes':>7} {'Bad teams':>10}")
    print("  " + "─" * 48)
    for _, r in sorted(reports.items()):
        print(f"  {r.table:<20} {r.rows_in:>7} {r.rows_out:>7} "
              f"{r.dupes_dropped:>7} {r.bad_teams_dropped:>10}")
        for w in r.warnings:
            print(f"    WARNING: {w}")
    print()

    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    _setup_logging(getattr(args, "verbose", False))
    from nfl_chatbot.data.database import get_engine, init_db, upsert_dataframe
    from nfl_chatbot.features.team_features import EloConfig, FeatureBuilder, feature_columns

    try:
        from nfl_chatbot.config import get_config
        cfg = get_config()
        elo_cfg = EloConfig(
            k_factor=float(cfg.app.features.elo.k_factor),
            initial_rating=float(cfg.app.features.elo.initial_rating),
            home_advantage=float(cfg.app.features.elo.home_advantage),
            regression_to_mean=float(cfg.app.features.elo.regression_to_mean),
        )
        window = args.window or cfg.app.features.rolling_window
    except Exception:
        logging.warning("Config unavailable — using defaults")
        elo_cfg = EloConfig()
        window = args.window or 4

    engine = get_engine()
    init_db(engine)

    logging.info("Building features (window=%d)…", window)
    builder = FeatureBuilder(engine, window=window, elo=elo_cfg)
    table = builder.build()

    if table.empty:
        logging.error("Feature table is empty — run 'nfl-chatbot ingest' first")
        return 1

    if args.seasons.strip():
        seasons = [int(s.strip()) for s in args.seasons.split(",")]
        before = len(table)
        table = table[table["season"].isin(seasons)].reset_index(drop=True)
        logging.info("Season filter %s: %d → %d rows", seasons, before, len(table))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logging.info("Saved → %s (%d rows × %d cols)", out, len(table), len(table.columns))

    if not args.no_db_upsert:
        _DB_SUBSET = {
            "game_id", "season", "week", "home_team", "away_team",
            "home_elo", "away_elo", "elo_diff", "spread_line", "total_line",
            "implied_home_win_prob", "home_win", "home_covered",
            "home_wins_last4", "away_wins_last4",
            "home_pts_scored_avg4", "away_pts_scored_avg4",
            "home_pts_allowed_avg4", "away_pts_allowed_avg4",
            "home_rest_days", "away_rest_days", "rest_diff",
            "div_game", "playoff_game", "neutral_site",
        }
        keep = [c for c in table.columns if c in _DB_SUBSET]
        n = upsert_dataframe(table[keep], "engineered_features", engine)
        logging.info("Upserted %d rows to engineered_features", n)

    feat_cols = feature_columns(window)
    available = [c for c in feat_cols if c in table.columns]
    missing = [c for c in feat_cols if c not in table.columns]

    print("\n── Feature build summary ────────────────────────────────")
    print(f"  Rows            : {len(table):,}")
    print(f"  Total columns   : {len(table.columns)}")
    print(f"  Feature columns : {len(available)} / {len(feat_cols)}")
    if "season" in table.columns:
        seasons_present = sorted(table["season"].dropna().unique().astype(int).tolist())
        print(f"  Seasons covered : {seasons_present}")
    print(f"  Output          : {out}")
    if missing:
        print(f"  Missing features: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    print()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    _setup_logging(getattr(args, "verbose", False))
    import pandas as pd

    from nfl_chatbot.models.train import ModelTrainer
    from nfl_chatbot.models.spread_model import SpreadModelTrainer

    # Load feature table
    if args.from_db:
        from nfl_chatbot.data.database import get_engine, init_db
        from nfl_chatbot.features.team_features import FeatureBuilder

        engine = get_engine()
        init_db(engine)
        df = FeatureBuilder(engine, window=args.window).build()
        if df.empty:
            logging.error("Feature table empty — run 'nfl-chatbot ingest' first")
            return 1
    else:
        table_path = Path(args.table)
        if not table_path.exists():
            logging.error(
                "Table not found: %s\nRun 'nfl-chatbot features' first, or use --from-db",
                table_path,
            )
            return 1
        df = pd.read_csv(table_path)
        logging.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), table_path)

    if "home_win" not in df.columns:
        logging.error("Table is missing 'home_win' column — was features built correctly?")
        return 1

    # Resolve test season
    test_season = args.test_season
    if not test_season:
        try:
            from nfl_chatbot.config import get_config
            test_season = get_config().app.models.test_season
        except Exception:
            test_season = 2024
    logging.info("Test season: %d", test_season)

    output_dir = Path(args.output_dir) if args.output_dir else None
    exit_code = 0

    print()
    if args.model in ("win", "both"):
        try:
            trainer = ModelTrainer(test_season=test_season)
            result = trainer.train(df)
            saved = trainer.save(result, output_dir=output_dir)
            best = result.metrics[result.best_model_name]
            print(
                f"  Win model   acc={best['accuracy']:.1%}  "
                f"AUC={best['roc_auc']:.4f}  "
                f"best={result.best_model_name}"
            )
            for artifact, path in saved.items():
                logging.debug("  saved %-28s → %s", artifact, path)
        except Exception as exc:
            logging.error("Win model training failed: %s", exc)
            exit_code = 1

    if args.model in ("spread", "both"):
        try:
            trainer = SpreadModelTrainer(test_season=test_season)
            result = trainer.train(df)
            saved = trainer.save(result, output_dir=output_dir)
            best = result.metrics[result.best_model_name]
            print(
                f"  Spread model  acc={best['accuracy']:.1%}  "
                f"AUC={best['roc_auc']:.4f}  "
                f"best={result.best_model_name}"
            )
        except Exception as exc:
            logging.error("Spread model training failed: %s", exc)
            exit_code = exit_code or 1

    print()
    return exit_code


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from nfl_chatbot.evaluation.leakage_checks import run_all_checks

    results = run_all_checks(
        modeling_csv=args.csv,
        feature_json=args.feature_json,
    )

    use_color = sys.stdout.isatty()
    _C = {
        "PASS": "\033[32m", "FAIL": "\033[31m",
        "WARN": "\033[33m", "SKIP": "\033[90m",
        "bold": "\033[1m",  "dim":  "\033[2m",
        "reset": "\033[0m",
    }

    def c(text: str, *styles: str) -> str:
        if not use_color:
            return text
        return "".join(_C.get(s, "") for s in styles) + text + _C["reset"]

    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    any_fail = False
    for cat, cat_results in by_cat.items():
        print(f"\n{c(cat.upper(), 'bold')}")
        for r in cat_results:
            badge = c(f"[{r.status:<4}]", r.status)
            print(f"  {badge}  {r.name}")
            if r.status in ("FAIL", "WARN") and r.detail:
                print(f"         {c(r.detail[:90], 'dim')}")
            if r.status == "FAIL":
                any_fail = True
                if args.fail_fast:
                    print(c("\nStopped at first FAIL (--fail-fast).\n", "FAIL"))
                    return 1

    counts = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print(
        f"\n  {c(str(counts['PASS']), 'PASS')} passed  "
        f"{c(str(counts['FAIL']), 'FAIL')} failed  "
        f"{c(str(counts['WARN']), 'WARN')} warned  "
        f"{c(str(counts['SKIP']), 'SKIP')} skipped  "
        f"({len(results)} total)\n"
    )

    if any_fail:
        print(c("AUDIT FAILED — fix leakage issues before training models.\n", "FAIL"))
        return 1
    print(c("AUDIT PASSED — no leakage detected.\n", "PASS"))
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if not question:
        print("Error: question cannot be empty.", file=sys.stderr)
        return 1

    from nfl_chatbot.chatbot.responder import Responder
    from nfl_chatbot.chatbot.tools import ChatbotTools

    tools = ChatbotTools(engine=None)

    if args.use_llm:
        try:
            from nfl_chatbot.chatbot.llm import auto_client, respond_with_llm
            client = auto_client()
            response = respond_with_llm(question, client=client, tools=tools)
        except Exception as exc:
            logging.warning("LLM unavailable (%s) — falling back to rule-based responder", exc)
            response = Responder(tools=tools).respond(question)
    else:
        response = Responder(tools=tools).respond(question)

    print(f"\n{response.answer}\n")

    if response.follow_ups:
        print("Follow-up questions:")
        for q in response.follow_ups:
            print(f"  • {q}")
        print()

    if response.has_caveat:
        print("Note: This answer may be incomplete — some data is unavailable.\n")

    return 0


def _cmd_run_api(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed.  Run: pip install 'uvicorn[standard]'")
        return 1

    workers = 1 if args.reload else args.workers
    print(f"\n  NFL AI Chatbot API")
    print(f"  http://{args.host}:{args.port}/docs    ← Swagger UI")
    print(f"  http://{args.host}:{args.port}/health  ← Health check\n")

    uvicorn.run(
        "nfl_chatbot.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_level=args.log_level,
    )
    return 0


def _cmd_run_app(args: argparse.Namespace) -> int:
    import os
    import subprocess

    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    if not app_path.exists():
        print(f"Streamlit app not found at {app_path}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
    ]

    env = os.environ.copy()
    if args.demo:
        env["NFL_DEMO_MODE"] = "1"
        print(f"\n  NFL AI Chatbot — Streamlit App  [DEMO MODE]")
    else:
        print(f"\n  NFL AI Chatbot — Streamlit App")
    print(f"  http://localhost:{args.port}\n")

    result = subprocess.run(cmd, env=env)
    return result.returncode


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

_PROCESSED_DIR_DEFAULT = str(_ROOT / "data" / "processed")
_TABLE_DEFAULT = str(_ROOT / "data" / "processed" / "modeling_table.csv")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nfl-chatbot",
        description="NFL AI Chatbot — command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pipeline order:\n"
            "  1. nfl-chatbot init-db\n"
            "  2. nfl-chatbot ingest\n"
            "  3. nfl-chatbot clean\n"
            "  4. nfl-chatbot features\n"
            "  5. nfl-chatbot train\n"
            "  6. nfl-chatbot evaluate\n"
            "  7. nfl-chatbot chat \"Who wins Chiefs vs Bills?\"\n"
            "     nfl-chatbot run-api\n"
            "     nfl-chatbot run-app\n"
        ),
    )
    p.add_argument("--version", action="version", version="nfl-chatbot 0.1.0")

    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── init-db ───────────────────────────────────────────────────────────────
    sub.add_parser(
        "init-db",
        help="Create / verify database tables",
        description="Initialise the SQLite database and create all tables. Safe to re-run.",
    )

    # ── ingest ────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "ingest",
        help="Download NFL data from nflverse into the database",
        description="Fetch teams, schedules, player stats, and rosters from nflverse.",
    )
    sp.add_argument("--seasons", nargs="+", type=int, metavar="YYYY",
                    help="Seasons to ingest (default: from config)")
    sp.add_argument("--datasets", nargs="+",
                    choices=["teams", "schedules", "player_stats", "rosters"],
                    metavar="DATASET",
                    help="Datasets to run (default: all four)")
    sp.add_argument("--force", action="store_true",
                    help="Re-download raw files even if cached copies exist")
    sp.add_argument("-v", "--verbose", action="store_true")

    # ── clean ─────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "clean",
        help="Clean processed CSV files",
        description="Apply cleaning rules to all processed CSVs in data/processed/.",
    )
    sp.add_argument("--processed-dir", default=_PROCESSED_DIR_DEFAULT,
                    help="Directory containing processed CSV files (default: data/processed)")
    sp.add_argument("--seasons", default="",
                    help="Comma-separated seasons to keep, e.g. 2022,2023,2024")
    sp.add_argument("-v", "--verbose", action="store_true")

    # ── features ──────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "features",
        help="Build the modeling feature table",
        description="Engineer 75+ features from the database and write modeling_table.csv.",
    )
    sp.add_argument("--output", default=_TABLE_DEFAULT,
                    help="Output CSV path (default: data/processed/modeling_table.csv)")
    sp.add_argument("--seasons", default="",
                    help="Comma-separated season filter (default: all)")
    sp.add_argument("--window", type=int, default=0,
                    help="Rolling window size in games (default: from config, usually 4)")
    sp.add_argument("--no-db-upsert", action="store_true",
                    help="Skip upserting features into the engineered_features table")
    sp.add_argument("-v", "--verbose", action="store_true")

    # ── train ─────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "train",
        help="Train win and/or ATS prediction models",
        description=(
            "Train Logistic Regression, Random Forest, and XGBoost on a "
            "chronological split and save the best model to models/."
        ),
    )
    sp.add_argument("--model", choices=["win", "spread", "both"], default="both",
                    help="Which model to train (default: both)")
    sp.add_argument("--table", default=_TABLE_DEFAULT,
                    help="Path to modeling_table.csv")
    sp.add_argument("--test-season", type=int, default=None,
                    help="Hold-out test season (default: from config, usually 2024)")
    sp.add_argument("--output-dir", default=None,
                    help="Directory for model artifacts (default: models/)")
    sp.add_argument("--from-db", action="store_true",
                    help="Build the feature table from the DB instead of loading CSV")
    sp.add_argument("--window", type=int, default=4,
                    help="Rolling window used with --from-db (default: 4)")
    sp.add_argument("-v", "--verbose", action="store_true")

    # ── evaluate ──────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "evaluate",
        help="Run data-leakage audit checks",
        description=(
            "Audit the feature pipeline for leakage: rolling windows, target "
            "exclusion, Elo pre-game safety, H2H date guards, and more."
        ),
    )
    sp.add_argument("--csv", metavar="PATH", default=None,
                    help="Path to modeling_table.csv (enables file-based checks)")
    sp.add_argument("--feature-json", metavar="PATH", default=None,
                    help="Path to feature column JSON")
    sp.add_argument("--fail-fast", action="store_true",
                    help="Stop after the first FAIL")

    # ── chat ──────────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "chat",
        help="Ask a question and print the answer",
        description='Example: nfl-chatbot chat "Who wins Chiefs vs Bills this week?"',
    )
    sp.add_argument("question", nargs="+",
                    help="Natural-language question (quote multi-word questions)")
    sp.add_argument("--use-llm", action="store_true",
                    help="Route through the LLM client for richer answers (requires API key)")

    # ── run-api ───────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "run-api",
        help="Start the FastAPI server",
        description="Serve the REST API via uvicorn. Swagger UI at /docs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp.add_argument("--host", default="127.0.0.1", help="Bind host")
    sp.add_argument("--port", type=int, default=8000, help="Bind port")
    sp.add_argument("--reload", action="store_true",
                    help="Enable auto-reload (development only)")
    sp.add_argument("--workers", type=int, default=1,
                    help="Worker processes (ignored with --reload)")
    sp.add_argument("--log-level", default="info",
                    choices=["debug", "info", "warning", "error"])

    # ── run-app ───────────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "run-app",
        help="Start the Streamlit frontend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp.add_argument("--port", type=int, default=8501, help="Port to serve on")
    sp.add_argument("--no-browser", action="store_true",
                    help="Do not open a browser tab automatically")
    sp.add_argument("--demo", action="store_true",
                    help="Force demo mode (sample data, no ingestion required)")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "init-db":  _cmd_init_db,
    "ingest":   _cmd_ingest,
    "clean":    _cmd_clean,
    "features": _cmd_features,
    "train":    _cmd_train,
    "evaluate": _cmd_evaluate,
    "chat":     _cmd_chat,
    "run-api":  _cmd_run_api,
    "run-app":  _cmd_run_app,
}


def cli() -> None:
    """Entry point registered as ``nfl-chatbot`` in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()
    fn = _DISPATCH[args.command]
    sys.exit(fn(args) or 0)


if __name__ == "__main__":
    cli()
