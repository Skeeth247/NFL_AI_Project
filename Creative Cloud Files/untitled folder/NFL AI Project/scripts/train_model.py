#!/usr/bin/env python3
"""
Train NFL prediction models (win and/or spread).

Loads the engineered feature table (built by scripts/build_features.py),
trains Logistic Regression, Random Forest, and XGBoost (if installed) on a
chronological split, and saves the best models + artifacts to models/.

Models
------
  win    : predict whether the home team wins  → models/best_model.joblib
  spread : predict whether the home team covers the spread → models/spread_model.joblib
  both   : train both (default)

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --model win
    python scripts/train_model.py --model spread
    python scripts/train_model.py --model both --test-season 2023
    python scripts/train_model.py --table data/processed/modeling_table.csv --verbose

Exit codes
----------
    0  success (at least one model trained successfully)
    1  table not found or no labeled rows
    2  chronological split failed (not enough seasons)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_chatbot.models.train import ModelTrainer          # noqa: E402
from nfl_chatbot.models.spread_model import SpreadModelTrainer  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NFL win and/or ATS prediction models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["win", "spread", "both"],
        default="both",
        help="Which model to train (default: both)",
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "data" / "processed" / "modeling_table.csv"),
        help="Path to modeling_table.csv (default: data/processed/modeling_table.csv)",
    )
    parser.add_argument(
        "--test-season",
        type=int,
        default=None,
        help="Hold-out test season (default: from config, usually 2024)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write model artifacts (default: models/)",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Build feature table from DB instead of loading CSV",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Rolling window passed to FeatureBuilder when --from-db is set (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_table(args: argparse.Namespace):
    import pandas as pd

    if args.from_db:
        logging.info("Building feature table from database…")
        from nfl_chatbot.data.database import get_engine, init_db
        from nfl_chatbot.features.team_features import FeatureBuilder

        engine = get_engine()
        init_db(engine)
        builder = FeatureBuilder(engine, window=args.window)
        df = builder.build()
        if df.empty:
            logging.error("FeatureBuilder returned an empty table — ingest data first.")
            sys.exit(1)
        logging.info("Feature table built from DB: %d rows", len(df))
        return df

    table_path = Path(args.table)
    if not table_path.exists():
        logging.error(
            "Modeling table not found at %s\n"
            "Run  python scripts/build_features.py  first, or use --from-db",
            table_path,
        )
        sys.exit(1)

    logging.info("Loading modeling table from %s", table_path)
    df = pd.read_csv(table_path)
    logging.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def _resolve_test_season(args: argparse.Namespace) -> int:
    if args.test_season:
        return args.test_season
    try:
        from nfl_chatbot.config import get_config
        return get_config().app.models.test_season
    except Exception:
        return 2024


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_summary(result, title: str) -> None:
    split = result.split_info
    cover_key = "train_cover_rate" if "train_cover_rate" in split else "train_home_win_rate"
    test_cover_key = "test_cover_rate" if "test_cover_rate" in split else "test_home_win_rate"

    print()
    print("─" * 72)
    print(f"  {title}")
    print("─" * 72)
    print(f"  Train seasons  : {split.get('train_seasons')}")
    print(f"  Test season    : {split.get('test_season')}")
    print(f"  Train rows     : {split.get('train_rows'):,}")
    print(f"  Test rows      : {split.get('test_rows'):,}")
    print(f"  Train rate     : {split.get(cover_key, 0):.1%}")
    print(f"  Test rate      : {split.get(test_cover_key, 0):.1%}")
    if split.get("rows_dropped_no_spread"):
        print(f"  Rows w/o spread: {split['rows_dropped_no_spread']:,} dropped")
    print(f"  Features used  : {split.get('n_features')}")
    print()
    print(f"  {'Model':<24}  {'Acc':>6}  {'ROC-AUC':>8}  {'CV-AUC':>7}  {'F1':>6}  {'Brier':>7}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*7}")
    for name, m in result.metrics.items():
        marker = "  ← best" if name == result.best_model_name else ""
        print(
            f"  {name:<24}  {m['accuracy']:>6.4f}  {m['roc_auc']:>8.4f}"
            f"  {m.get('cv_roc_auc', 0):>7.4f}  {m['f1']:>6.4f}"
            f"  {m['brier_score']:>7.4f}{marker}"
        )
    print("─" * 72)

    best_m = result.metrics[result.best_model_name]
    acc = best_m["accuracy"]
    auc = best_m["roc_auc"]

    if "spread" in title.lower():
        # ATS performance benchmarks differ from straight win prediction
        if acc >= 0.55:
            note = "Good ATS performance — significantly above market noise."
        elif acc >= 0.52:
            note = "Modest ATS edge — consistent with a well-calibrated model."
        else:
            note = "Near-random — expected; the spread is designed to be 50/50."
    else:
        if acc >= 0.70:
            note = "Strong predictive performance."
        elif acc >= 0.60:
            note = "Moderate performance — typical for NFL prediction."
        else:
            note = "Weak performance — may need more/richer data."

    print()
    print(f"  Best model ({result.best_model_name}): accuracy={acc:.1%}, ROC-AUC={auc:.4f}")
    print(f"  Assessment: {note}")
    print()


# ── Training routines ──────────────────────────────────────────────────────────

def _run_win_model(df, test_season: int, output_dir) -> int:
    """Train the home-win model. Returns 0 on success, 2 on split failure."""
    import pandas as pd

    if df["home_win"].notna().sum() == 0:
        logging.error("No labeled rows (home_win is NULL). Ingest completed games first.")
        return 1

    logging.info("── WIN MODEL ──────────────────────────────────────────")
    try:
        trainer = ModelTrainer(test_season=test_season)
        result = trainer.train(df)
    except ValueError as exc:
        logging.error("Win model training failed: %s", exc)
        return 2

    saved = trainer.save(result, output_dir=output_dir)
    _print_summary(result, "NFL WIN PREDICTOR — RESULTS")

    print("  Artifacts saved:")
    for name, path in saved.items():
        print(f"    {name:<28} → {path}")
    print()
    return 0


def _run_spread_model(df, test_season: int, output_dir) -> int:
    """Train the ATS model. Returns 0 on success, non-zero on failure."""
    from nfl_chatbot.models.spread_model import derive_home_covered

    # Check whether spread data exists at all
    df_check = derive_home_covered(df)
    n_labeled = int(df_check["home_covered"].notna().sum())
    if n_labeled == 0:
        logging.error(
            "No rows with spread data (home_covered is all-NULL). "
            "Ingest odds/spread data before training the ATS model."
        )
        return 1

    logging.info("── SPREAD MODEL ───────────────────────────────────────")
    logging.info("%d rows have spread labels", n_labeled)

    try:
        trainer = SpreadModelTrainer(test_season=test_season)
        result = trainer.train(df)
    except ValueError as exc:
        logging.error("Spread model training failed: %s", exc)
        return 2

    saved = trainer.save(result, output_dir=output_dir)
    _print_summary(result, "NFL ATS PREDICTOR — RESULTS")

    print("  Artifacts saved:")
    for name, path in saved.items():
        print(f"    {name:<28} → {path}")
    print()
    return 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    df = _load_table(args)
    test_season = _resolve_test_season(args)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if "home_win" not in df.columns:
        logging.error("Table missing 'home_win' column — was build_features.py run?")
        return 1

    exit_code = 0

    if args.model in ("win", "both"):
        rc = _run_win_model(df, test_season, output_dir)
        if rc != 0:
            exit_code = rc

    if args.model in ("spread", "both"):
        rc = _run_spread_model(df, test_season, output_dir)
        if rc != 0 and exit_code == 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
