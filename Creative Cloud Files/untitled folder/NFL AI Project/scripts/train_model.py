#!/usr/bin/env python3
"""
Train the NFL win-prediction model.

Loads the engineered feature table (built by scripts/build_features.py),
trains Logistic Regression, Random Forest, and XGBoost (if installed) on a
chronological split, and saves the best model + artifacts to models/.

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --test-season 2023
    python scripts/train_model.py --table data/processed/modeling_table.csv
    python scripts/train_model.py --output-dir models --verbose

Exit codes
----------
    0  success
    1  table not found or empty after filtering
    2  not enough labeled rows to split chronologically
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_chatbot.models.train import ModelTrainer, train_win_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NFL home-win prediction model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Rolling window size passed to FeatureBuilder when --from-db is set (default: 4)",
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


def _load_table(args: argparse.Namespace):
    """Load or build the modeling table. Returns a DataFrame."""
    import pandas as pd

    if args.from_db:
        logging.info("Building feature table from database…")
        from nfl_chatbot.data.database import get_engine, init_db
        from nfl_chatbot.features.team_features import FeatureBuilder, EloConfig

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


def _print_summary(result) -> None:
    """Print a human-readable summary table of model results."""
    split = result.split_info
    print()
    print("─" * 70)
    print("  NFL WIN PREDICTOR — TRAINING RESULTS")
    print("─" * 70)
    print(f"  Train seasons  : {split.get('train_seasons')}")
    print(f"  Test season    : {split.get('test_season')}")
    print(f"  Train rows     : {split.get('train_rows'):,}")
    print(f"  Test rows      : {split.get('test_rows'):,}")
    print(f"  Train win rate : {split.get('train_home_win_rate', 0):.1%}")
    print(f"  Test win rate  : {split.get('test_home_win_rate', 0):.1%}")
    print(f"  Features used  : {split.get('n_features')}")
    print()
    print(f"  {'Model':<24}  {'Acc':>6}  {'ROC-AUC':>8}  {'F1':>6}  {'Brier':>7}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*7}")
    for name, m in result.metrics.items():
        marker = "  ← best" if name == result.best_model_name else ""
        print(
            f"  {name:<24}  {m['accuracy']:>6.4f}  {m['roc_auc']:>8.4f}"
            f"  {m['f1']:>6.4f}  {m['brier_score']:>7.4f}{marker}"
        )
    print("─" * 70)
    print()

    # Honest assessment
    best_m = result.metrics[result.best_model_name]
    acc = best_m["accuracy"]
    auc = best_m["roc_auc"]
    if acc >= 0.70:
        note = "Strong predictive performance."
    elif acc >= 0.60:
        note = "Moderate performance — typical for NFL prediction."
    else:
        note = "Weak performance — may need more/richer data."
    print(f"  Best model ({result.best_model_name}): accuracy={acc:.1%}, ROC-AUC={auc:.4f}")
    print(f"  Assessment: {note}")
    print()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    df = _load_table(args)
    test_season = _resolve_test_season(args)

    # Check that there are labeled rows
    if "home_win" not in df.columns or df["home_win"].notna().sum() == 0:
        logging.error(
            "No labeled rows (home_win is NULL for all rows). "
            "Ingest completed game data before training."
        )
        return 1

    labeled = df["home_win"].notna().sum()
    logging.info("%d rows with home_win label", labeled)

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        trainer = ModelTrainer(test_season=test_season)
        result = trainer.train(df)
    except ValueError as exc:
        logging.error("Training failed: %s", exc)
        return 2

    saved = trainer.save(result, output_dir=output_dir)

    _print_summary(result)

    print("  Artifacts saved:")
    for name, path in saved.items():
        print(f"    {name:<24} → {path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
