#!/usr/bin/env python3
"""
Build the modeling feature table from the NFL database.

Reads from SQLite (games, team_game_stats, injuries), engineers 75+ features
across seven groups, and writes the final table to:
    data/processed/modeling_table.csv

Also upserts the schema-compatible subset of features into the
engineered_features database table for use by the chatbot API.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --seasons 2022,2023,2024
    python scripts/build_features.py --window 6 --output data/processed/table_w6.csv
    python scripts/build_features.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_chatbot.data.database import get_engine, init_db, upsert_dataframe  # noqa: E402
from nfl_chatbot.features.team_features import (                              # noqa: E402
    EloConfig,
    FeatureBuilder,
    feature_columns,
)


# ── Column subset that maps to the engineered_features DB table ────────────────

_DB_COL_MAP: dict[str, str] = {
    "game_id":                   "game_id",
    "season":                    "season",
    "week":                      "week",
    "home_team":                 "home_team",
    "away_team":                 "away_team",
    "home_elo":                  "home_elo",
    "away_elo":                  "away_elo",
    "elo_diff":                  "elo_diff",
    "home_wins_last4":           "home_wins_last4",
    "away_wins_last4":           "away_wins_last4",
    "home_pts_scored_avg4":      "home_pts_scored_avg4",
    "away_pts_scored_avg4":      "away_pts_scored_avg4",
    "home_pts_allowed_avg4":     "home_pts_allowed_avg4",
    "away_pts_allowed_avg4":     "away_pts_allowed_avg4",
    "home_yds_per_play_avg4":    "home_yds_per_play_avg4",
    "away_yds_per_play_avg4":    "away_yds_per_play_avg4",
    "home_turnover_diff_avg4":   "home_turnover_diff_avg4",
    "away_turnover_diff_avg4":   "away_turnover_diff_avg4",
    "home_rest_days":            "home_rest_days",
    "away_rest_days":            "away_rest_days",
    "rest_diff":                 "rest_diff",
    "div_game":                  "div_game",
    "playoff_game":              "playoff_game",
    "neutral_site":              "neutral_site",
    "spread_line":               "spread_line",
    "total_line":                "total_line",
    "implied_home_win_prob":     "implied_home_win_prob",
    "home_win":                  "home_win",
    "home_covered":              "home_covered",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the NFL modeling feature table from the database."
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "modeling_table.csv"),
        help="Output CSV path (default: data/processed/modeling_table.csv)",
    )
    parser.add_argument(
        "--seasons",
        default="",
        help="Comma-separated season filter, e.g. 2022,2023 (default: all)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Rolling window size in games (default: 4)",
    )
    parser.add_argument(
        "--no-db-upsert",
        action="store_true",
        help="Skip upserting features into the engineered_features database table",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    # ── Parse seasons ──────────────────────────────────────────────────────────
    seasons: list[int] | None = None
    if args.seasons.strip():
        try:
            seasons = [int(s.strip()) for s in args.seasons.split(",")]
        except ValueError:
            logging.error("Invalid --seasons value: %r", args.seasons)
            return 1

    # ── Load config for Elo params ─────────────────────────────────────────────
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
        logging.warning("Could not load config; using defaults")
        elo_cfg = EloConfig()
        window = args.window

    # ── Connect to DB ──────────────────────────────────────────────────────────
    engine = get_engine()
    init_db(engine)

    # ── Build features ─────────────────────────────────────────────────────────
    logging.info("Building features — window=%d", window)
    builder = FeatureBuilder(engine, window=window, elo=elo_cfg)
    table = builder.build()

    if table.empty:
        logging.error("Feature table is empty — ensure the database has been ingested")
        return 1

    # ── Season filter ──────────────────────────────────────────────────────────
    if seasons and "season" in table.columns:
        before = len(table)
        table = table[table["season"].isin(seasons)].reset_index(drop=True)
        logging.info("Season filter %s: %d → %d rows", seasons, before, len(table))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logging.info("Saved modeling table → %s (%d rows × %d cols)",
                 out_path, len(table), len(table.columns))

    # ── Upsert to DB ───────────────────────────────────────────────────────────
    if not args.no_db_upsert:
        db_cols = [c for c in _DB_COL_MAP if c in table.columns]
        db_df = table[db_cols].rename(columns=_DB_COL_MAP)
        n = upsert_dataframe(db_df, "engineered_features", engine)
        logging.info("Upserted %d rows to engineered_features table", n)

    # ── Summary ────────────────────────────────────────────────────────────────
    feat_cols = feature_columns(window)
    available = [c for c in feat_cols if c in table.columns]
    missing_feats = [c for c in feat_cols if c not in table.columns]

    print()
    print("─" * 65)
    print(f"  Rows in modeling table :  {len(table):,}")
    print(f"  Total columns          :  {len(table.columns)}")
    print(f"  Feature columns        :  {len(available)} / {len(feat_cols)}")
    if table.get("season") is not None and not table.empty:
        seasons_in = sorted(table["season"].dropna().unique().astype(int).tolist())
        print(f"  Seasons covered        :  {seasons_in}")
    print(f"  Games with home_win    :  {table['home_win'].notna().sum():,}")
    print(f"  Games with home_covered:  {table.get('home_covered', [None] * len(table)).notna().sum():,}")
    if missing_feats:
        print(f"  Missing features       :  {missing_feats[:5]}{'...' if len(missing_feats) > 5 else ''}")
    print(f"  Output                 :  {out_path}")
    print("─" * 65)
    print()

    if missing_feats:
        logging.warning(
            "%d feature columns missing (likely need team_game_stats ingestion)",
            len(missing_feats),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
