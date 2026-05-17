"""
Player feature engineering and comparison for the NFL AI Chatbot.

Public API
----------
compare_players(player_a, player_b, seasons=None, engine=None, recent_n=4)
    → dict   Chatbot-ready comparison card for two players.

build_player_profile(name_or_id, seasons=None, recent_n=4, engine=None)
    → dict   Single-player profile card.

Both functions accept player names (exact or partial) or nflverse player_id strings.
All stats aggregate across the requested seasons; ``recent_n`` controls the rolling
last-N-games form window. Missing columns and empty tables are handled gracefully —
the output dict always has the same shape, with None for unavailable values.

Position-aware metrics
-----------------------
QB  : passing_yards, passing_tds, interceptions, completion_pct, passer_rating,
      rushing_yards, rushing_tds
RB  : rushing_yards, rushing_tds, carries, yards_per_carry,
      receptions, receiving_yards, receiving_tds
WR/TE: receptions, receiving_yards, receiving_tds, targets, yards_per_reception

Cross-position metrics (always included):
  games_played, total_yards, total_tds, yards_per_game, tds_per_game,
  fantasy_points_total, fantasy_points_per_game, turnovers (INTs for QB, else None)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Position constants ─────────────────────────────────────────────────────────

_QB = frozenset({"QB"})
_RB = frozenset({"RB", "FB"})
_RECEIVER = frozenset({"WR", "TE"})
_ALL_SKILL = _QB | _RB | _RECEIVER

# Metrics shown for each position in the comparison section
_POSITION_METRICS: dict[str, list[str]] = {
    "QB": [
        "games_played", "passing_yards", "passing_tds", "interceptions",
        "completion_pct", "passer_rating", "rushing_yards", "rushing_tds",
        "turnovers", "yards_per_game", "tds_per_game",
    ],
    "RB": [
        "games_played", "rushing_yards", "rushing_tds", "carries",
        "yards_per_carry", "receptions", "receiving_yards", "receiving_tds",
        "yards_per_game", "tds_per_game",
    ],
    "WR": [
        "games_played", "receptions", "receiving_yards", "receiving_tds",
        "targets", "yards_per_reception", "yards_per_game", "tds_per_game",
    ],
    "TE": [
        "games_played", "receptions", "receiving_yards", "receiving_tds",
        "targets", "yards_per_reception", "yards_per_game", "tds_per_game",
    ],
}
_DEFAULT_METRICS = [
    "games_played", "total_yards", "total_tds", "yards_per_game",
    "tds_per_game", "fantasy_points_total", "fantasy_points_per_game",
]

# For percentile computation: higher is better unless listed here
_LOWER_BETTER = frozenset({"interceptions", "turnovers"})

_DEFAULT_RECENT_N = 4


# ── Utility helpers ────────────────────────────────────────────────────────────

def _safe(v: Any) -> Any:
    """Convert NaN/inf to None so the output dict is JSON-safe."""
    if v is None:
        return None
    try:
        if np.isnan(v) or np.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _sum_col(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if col in df.columns else 0.0


def _mean_col(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return float(vals.mean()) if not vals.empty else None


def _round2(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


# ── Player lookup ──────────────────────────────────────────────────────────────

def _load_player_index(engine: Engine) -> pd.DataFrame:
    """
    Return a (player_id, player_name, position, team) frame built from
    player_game_stats and player_season_stats.  Used only for name resolution.
    """
    frames: list[pd.DataFrame] = []
    for table in ("player_game_stats", "player_season_stats"):
        try:
            from nfl_chatbot.data.database import read_table
            df = read_table(
                table, engine,
                columns=["player_id", "player_name", "position", "team"],
            )
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.debug("Skipping %s for player index: %s", table, exc)

    if not frames:
        return pd.DataFrame(columns=["player_id", "player_name", "position", "team"])

    combined = pd.concat(frames, ignore_index=True).drop_duplicates("player_id")
    # Use last team as most recent
    combined = (
        combined.groupby("player_id", sort=False)
        .agg(
            player_name=("player_name", "first"),
            position=("position", "first"),
            team=("team", "last"),
        )
        .reset_index()
    )
    return combined


def _resolve_player(
    name_or_id: str,
    index: pd.DataFrame,
) -> dict[str, str] | None:
    """
    Resolve a player name or player_id string.

    Resolution order:
      1. Exact player_id match
      2. Case-insensitive exact name match
      3. Case-insensitive substring match (first hit by index order)

    Returns a dict with keys player_id, player_name, position, team, or None.
    """
    if index.empty:
        return None

    query = name_or_id.strip()
    lower = query.lower()

    # 1. Exact ID
    hit = index[index["player_id"] == query]
    if not hit.empty:
        return hit.iloc[0].to_dict()

    # 2. Exact name (case-insensitive)
    hit = index[index["player_name"].str.lower() == lower]
    if not hit.empty:
        return hit.iloc[0].to_dict()

    # 3. Substring name
    hit = index[index["player_name"].str.lower().str.contains(lower, regex=False, na=False)]
    if not hit.empty:
        return hit.iloc[0].to_dict()

    return None


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_game_logs(
    player_id: str,
    seasons: list[int] | None,
    engine: Engine,
) -> pd.DataFrame:
    try:
        from nfl_chatbot.data.database import read_table
        df = read_table("player_game_stats", engine)
    except Exception as exc:
        logger.warning("player_game_stats unavailable: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df = df[df["player_id"] == player_id].copy()
    if seasons:
        df = df[df["season"].isin(seasons)]
    return df.sort_values(["season", "week"]).reset_index(drop=True)


def _load_season_rows(
    player_id: str,
    seasons: list[int] | None,
    engine: Engine,
) -> pd.DataFrame:
    try:
        from nfl_chatbot.data.database import read_table
        df = read_table("player_season_stats", engine)
    except Exception as exc:
        logger.warning("player_season_stats unavailable: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    df = df[df["player_id"] == player_id].copy()
    if seasons:
        df = df[df["season"].isin(seasons)]
    return df.sort_values("season").reset_index(drop=True)


def _load_peer_season_rows(
    position: str,
    seasons: list[int] | None,
    engine: Engine,
) -> pd.DataFrame:
    """All player_season_stats rows for a position group, used for percentile ranking."""
    try:
        from nfl_chatbot.data.database import read_table
        df = read_table("player_season_stats", engine)
    except Exception as exc:
        logger.debug("Cannot load peers: %s", exc)
        return pd.DataFrame()

    if df.empty:
        return df

    pos_upper = position.upper()
    df = df[df["position"].str.upper() == pos_upper].copy()
    if seasons:
        df = df[df["season"].isin(seasons)]
    return df


# ── Aggregation ────────────────────────────────────────────────────────────────

def _totals_from_games(games: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-game rows into season totals."""
    if games.empty:
        return {}

    gp = len(games)
    completions = _sum_col(games, "completions")
    attempts = _sum_col(games, "attempts")
    pass_yds = _sum_col(games, "passing_yards")
    pass_tds = _sum_col(games, "passing_tds")
    ints = _sum_col(games, "interceptions")
    carries = _sum_col(games, "carries")
    rush_yds = _sum_col(games, "rushing_yards")
    rush_tds = _sum_col(games, "rushing_tds")
    targets = _sum_col(games, "targets")
    recs = _sum_col(games, "receptions")
    rec_yds = _sum_col(games, "receiving_yards")
    rec_tds = _sum_col(games, "receiving_tds")
    fp = _sum_col(games, "fantasy_points")

    comp_pct = _safe(round(completions / attempts * 100, 1)) if attempts > 0 else None
    ypc = _safe(round(rush_yds / carries, 2)) if carries > 0 else None
    ypr = _safe(round(rec_yds / recs, 2)) if recs > 0 else None
    pr = _safe(_mean_col(games, "passer_rating"))
    total_yds = pass_yds + rush_yds + rec_yds
    total_tds = pass_tds + rush_tds + rec_tds

    return {
        "games_played": gp,
        "passing_yards": _safe(pass_yds) if pass_yds > 0 else None,
        "passing_tds": _safe(int(pass_tds)) if pass_tds > 0 else None,
        "interceptions": _safe(int(ints)) if "interceptions" in games.columns else None,
        "completion_pct": comp_pct,
        "passer_rating": _safe(round(pr, 1)) if pr is not None else None,
        "carries": _safe(int(carries)) if carries > 0 else None,
        "rushing_yards": _safe(rush_yds) if rush_yds > 0 else None,
        "rushing_tds": _safe(int(rush_tds)) if rush_tds > 0 else None,
        "yards_per_carry": ypc,
        "targets": _safe(int(targets)) if targets > 0 else None,
        "receptions": _safe(int(recs)) if recs > 0 else None,
        "receiving_yards": _safe(rec_yds) if rec_yds > 0 else None,
        "receiving_tds": _safe(int(rec_tds)) if rec_tds > 0 else None,
        "yards_per_reception": ypr,
        "total_yards": _safe(total_yds) if total_yds > 0 else None,
        "total_tds": _safe(int(total_tds)) if total_tds > 0 else None,
        "fantasy_points_total": _safe(round(fp, 1)) if fp > 0 else None,
        "turnovers": _safe(int(ints)) if "interceptions" in games.columns else None,
    }


def _totals_from_season_rows(rows: pd.DataFrame) -> dict[str, Any]:
    """Sum multiple season-aggregate rows into a single totals dict."""
    if rows.empty:
        return {}

    gp = int(_sum_col(rows, "games_played")) if "games_played" in rows.columns else len(rows)
    pass_yds = _sum_col(rows, "passing_yards")
    pass_tds = _sum_col(rows, "passing_tds")
    ints = _sum_col(rows, "interceptions")
    carries = _sum_col(rows, "carries")
    rush_yds = _sum_col(rows, "rushing_yards")
    rush_tds = _sum_col(rows, "rushing_tds")
    targets = _sum_col(rows, "targets")
    recs = _sum_col(rows, "receptions")
    rec_yds = _sum_col(rows, "receiving_yards")
    rec_tds = _sum_col(rows, "receiving_tds")
    fp = _sum_col(rows, "fantasy_points_total")

    # Weighted averages for rate stats (weight = games_played per season)
    gp_weights = rows["games_played"].fillna(1) if "games_played" in rows.columns else pd.Series([1] * len(rows))
    total_w = gp_weights.sum()

    def _wavg(col: str) -> float | None:
        if col not in rows.columns:
            return None
        vals = rows[col].fillna(0)
        return float((vals * gp_weights).sum() / total_w) if total_w > 0 else None

    comp_pct = _wavg("completion_pct")
    pr = _wavg("passer_rating")
    ypc = _safe(round(rush_yds / carries, 2)) if carries > 0 else None
    ypr = _safe(round(rec_yds / recs, 2)) if recs > 0 else None
    total_yds = pass_yds + rush_yds + rec_yds
    total_tds = pass_tds + rush_tds + rec_tds

    return {
        "games_played": gp,
        "passing_yards": _safe(pass_yds) if pass_yds > 0 else None,
        "passing_tds": _safe(int(pass_tds)) if pass_tds > 0 else None,
        "interceptions": _safe(int(ints)) if ints >= 0 and "interceptions" in rows.columns else None,
        "completion_pct": _safe(round(comp_pct, 1)) if comp_pct is not None else None,
        "passer_rating": _safe(round(pr, 1)) if pr is not None else None,
        "carries": _safe(int(carries)) if carries > 0 else None,
        "rushing_yards": _safe(rush_yds) if rush_yds > 0 else None,
        "rushing_tds": _safe(int(rush_tds)) if rush_tds > 0 else None,
        "yards_per_carry": ypc,
        "targets": _safe(int(targets)) if targets > 0 else None,
        "receptions": _safe(int(recs)) if recs > 0 else None,
        "receiving_yards": _safe(rec_yds) if rec_yds > 0 else None,
        "receiving_tds": _safe(int(rec_tds)) if rec_tds > 0 else None,
        "yards_per_reception": ypr,
        "total_yards": _safe(total_yds) if total_yds > 0 else None,
        "total_tds": _safe(int(total_tds)) if total_tds > 0 else None,
        "fantasy_points_total": _safe(round(fp, 1)) if fp > 0 else None,
        "turnovers": _safe(int(ints)) if "interceptions" in rows.columns else None,
    }


def _per_game(totals: dict[str, Any], games_played: int) -> dict[str, Any]:
    """Derive per-game averages from season totals."""
    if games_played <= 0:
        return {}

    gp = games_played

    def _div(key: str) -> float | None:
        v = totals.get(key)
        return _safe(round(v / gp, 2)) if v is not None else None

    pass_yds = totals.get("passing_yards") or 0.0
    rush_yds = totals.get("rushing_yards") or 0.0
    rec_yds = totals.get("receiving_yards") or 0.0
    total_yds = (totals.get("total_yards") or (pass_yds + rush_yds + rec_yds))

    pass_tds = totals.get("passing_tds") or 0
    rush_tds = totals.get("rushing_tds") or 0
    rec_tds = totals.get("receiving_tds") or 0
    total_tds = totals.get("total_tds") or (pass_tds + rush_tds + rec_tds)

    fp_total = totals.get("fantasy_points_total")

    return {
        "yards_per_game": _safe(round(total_yds / gp, 1)),
        "tds_per_game": _safe(round(total_tds / gp, 2)),
        "passing_yards_per_game": _div("passing_yards"),
        "rushing_yards_per_game": _div("rushing_yards"),
        "receiving_yards_per_game": _div("receiving_yards"),
        "fantasy_points_per_game": _safe(round(fp_total / gp, 1)) if fp_total is not None else None,
        "receptions_per_game": _div("receptions"),
        "targets_per_game": _div("targets"),
    }


# ── Recent form ────────────────────────────────────────────────────────────────

def _recent_form(games: pd.DataFrame, n: int) -> dict[str, Any]:
    """Per-game averages from the last *n* game-log rows."""
    if games.empty:
        return {"n_games": 0}

    recent = games.tail(n)
    g = len(recent)

    def _avg(col: str) -> float | None:
        if col not in recent.columns:
            return None
        vals = recent[col].dropna()
        return _safe(round(float(vals.mean()), 2)) if not vals.empty else None

    completions = _sum_col(recent, "completions")
    attempts = _sum_col(recent, "attempts")
    comp_pct = _safe(round(completions / attempts * 100, 1)) if attempts > 0 else None

    pass_yds = _avg("passing_yards")
    rush_yds = _avg("rushing_yards")
    rec_yds = _avg("receiving_yards")
    p = pass_yds or 0.0
    r = rush_yds or 0.0
    rc = rec_yds or 0.0
    yds_per_game = _safe(round(p + r + rc, 1)) if (p + r + rc) > 0 else None

    return {
        "n_games": g,
        "passing_yards_avg": pass_yds,
        "passing_tds_avg": _avg("passing_tds"),
        "interceptions_avg": _avg("interceptions"),
        "completion_pct": comp_pct,
        "passer_rating_avg": _avg("passer_rating"),
        "rushing_yards_avg": rush_yds,
        "rushing_tds_avg": _avg("rushing_tds"),
        "receptions_avg": _avg("receptions"),
        "targets_avg": _avg("targets"),
        "receiving_yards_avg": rec_yds,
        "receiving_tds_avg": _avg("receiving_tds"),
        "yards_per_game": yds_per_game,
        "fantasy_points_avg": _avg("fantasy_points"),
    }


# ── Percentiles and ranks ──────────────────────────────────────────────────────

def _percentiles_and_ranks(
    player_totals: dict[str, Any],
    peers: pd.DataFrame,
    player_id: str,
) -> tuple[dict[str, float | None], dict[str, int | None]]:
    """
    Compute percentile (0–100) and rank (1 = best) for each numeric metric
    in *player_totals* against position peers from *player_season_stats*.

    Returns (percentiles_dict, ranks_dict).
    """
    if peers.empty:
        none_map: dict[str, None] = {k: None for k in player_totals}
        return none_map, none_map

    # Aggregate peers to single row per player (sum across seasons like we do for target)
    agg_peers: dict[str, list[float]] = {}
    for col in peers.select_dtypes(include="number").columns:
        agg_peers[col] = peers.groupby("player_id")[col].sum().values.tolist()

    peer_player_ids = peers.groupby("player_id").ngroups
    percentiles: dict[str, float | None] = {}
    ranks: dict[str, int | None] = {}

    numeric_metrics = [
        k for k, v in player_totals.items()
        if v is not None and isinstance(v, (int, float))
    ]

    # Map totals key → season_stats column name
    col_map = {
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "interceptions": "interceptions",
        "completion_pct": "completion_pct",
        "passer_rating": "passer_rating",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
        "games_played": "games_played",
        "fantasy_points_total": "fantasy_points_total",
    }

    for metric in numeric_metrics:
        db_col = col_map.get(metric)
        player_val = player_totals[metric]

        if db_col is None or db_col not in peers.columns or player_val is None:
            percentiles[metric] = None
            ranks[metric] = None
            continue

        peer_vals = (
            peers.groupby("player_id")[db_col]
            .sum()
            .dropna()
        )
        if peer_vals.empty:
            percentiles[metric] = None
            ranks[metric] = None
            continue

        all_vals = peer_vals.values
        lower_is_better = metric in _LOWER_BETTER

        if lower_is_better:
            # Percentile: % of peers with a HIGHER value (worse) than this player
            pct = float(np.mean(all_vals > player_val) * 100)
            rank = int((all_vals < player_val).sum()) + 1
        else:
            pct = float(np.mean(all_vals < player_val) * 100)
            rank = int((all_vals > player_val).sum()) + 1

        percentiles[metric] = round(pct, 1)
        ranks[metric] = rank

    return percentiles, ranks


# ── Profile builder ────────────────────────────────────────────────────────────

def _build_profile(
    info: dict[str, str],
    seasons: list[int] | None,
    recent_n: int,
    engine: Engine,
    peers: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a full profile dict for a resolved player.

    Prefers season-aggregate rows (player_season_stats) and falls back to
    aggregating from game logs when season rows are unavailable.
    """
    player_id = info["player_id"]
    position = (info.get("position") or "").upper()

    game_logs = _load_game_logs(player_id, seasons, engine)
    season_rows = _load_season_rows(player_id, seasons, engine)

    # Prefer season rows; fall back to game logs
    if not season_rows.empty:
        totals = _totals_from_season_rows(season_rows)
        data_source = "player_season_stats"
    elif not game_logs.empty:
        totals = _totals_from_games(game_logs)
        data_source = "player_game_stats"
    else:
        totals = {}
        data_source = "none"

    gp = totals.get("games_played") or 0
    per_game = _per_game(totals, gp) if gp > 0 else {}

    # Merge per_game into totals for a flat stats dict
    flat_totals = {**totals, **per_game}

    recent = _recent_form(game_logs, recent_n) if not game_logs.empty else {"n_games": 0}
    percentiles, rank_map = _percentiles_and_ranks(totals, peers, player_id)

    # Seasons covered
    covered: list[int] = []
    if not season_rows.empty and "season" in season_rows.columns:
        covered = sorted(season_rows["season"].dropna().astype(int).unique().tolist())
    elif not game_logs.empty and "season" in game_logs.columns:
        covered = sorted(game_logs["season"].dropna().astype(int).unique().tolist())

    pos_metrics = _POSITION_METRICS.get(position, _DEFAULT_METRICS)

    return {
        "player_id": player_id,
        "name": info.get("player_name", "Unknown"),
        "position": position or None,
        "team": info.get("team"),
        "seasons_covered": covered,
        "data_source": data_source,
        "stats": {k: flat_totals.get(k) for k in pos_metrics},
        "totals": totals,
        "per_game": per_game,
        "recent_form": recent,
        "percentiles": {k: percentiles.get(k) for k in pos_metrics if k in percentiles},
        "ranks": {k: rank_map.get(k) for k in pos_metrics if k in rank_map},
    }


# ── Metric comparison ──────────────────────────────────────────────────────────

def _compare_metric(
    metric: str,
    val_a: Any,
    val_b: Any,
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    """
    Build a head-to-head comparison dict for a single metric.

    Returns advantage, absolute diff, and pct_diff (relative to val_b baseline).
    """
    if val_a is None and val_b is None:
        return {
            "player_a": None, "player_b": None,
            "advantage": None, "diff": None, "pct_diff": None,
        }
    if val_a is None:
        return {
            "player_a": None, "player_b": val_b,
            "advantage": name_b, "diff": None, "pct_diff": None,
        }
    if val_b is None:
        return {
            "player_a": val_a, "player_b": None,
            "advantage": name_a, "diff": None, "pct_diff": None,
        }

    diff = round(float(val_a) - float(val_b), 2)
    baseline = abs(float(val_b))
    pct_diff = _safe(round(diff / baseline * 100, 1)) if baseline > 0 else None

    lower_better = metric in _LOWER_BETTER
    if lower_better:
        advantage = name_a if diff < 0 else (name_b if diff > 0 else "tie")
    else:
        advantage = name_a if diff > 0 else (name_b if diff < 0 else "tie")

    return {
        "player_a": val_a,
        "player_b": val_b,
        "advantage": advantage,
        "diff": abs(diff),
        "pct_diff": abs(pct_diff) if pct_diff is not None else None,
    }


def _build_comparison(
    profile_a: dict,
    profile_b: dict,
    position: str,
) -> dict[str, Any]:
    """Build metric-by-metric comparison between two profiles."""
    pos_metrics = _POSITION_METRICS.get(position.upper(), _DEFAULT_METRICS)
    name_a = profile_a["name"]
    name_b = profile_b["name"]

    comparison: dict[str, Any] = {}
    a_edges: list[str] = []
    b_edges: list[str] = []

    for metric in pos_metrics:
        val_a = profile_a["totals"].get(metric) or profile_a["per_game"].get(metric)
        val_b = profile_b["totals"].get(metric) or profile_b["per_game"].get(metric)

        # yards_per_game / tds_per_game live in per_game
        if val_a is None:
            val_a = profile_a["per_game"].get(metric)
        if val_b is None:
            val_b = profile_b["per_game"].get(metric)

        result = _compare_metric(metric, val_a, val_b, name_a, name_b)
        comparison[metric] = result

        adv = result["advantage"]
        if adv == name_a:
            a_edges.append(metric)
        elif adv == name_b:
            b_edges.append(metric)

    return {
        "metrics": comparison,
        "player_a_edges": a_edges,
        "player_b_edges": b_edges,
        "player_a_edge_count": len(a_edges),
        "player_b_edge_count": len(b_edges),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_player_profile(
    name_or_id: str,
    seasons: list[int] | None = None,
    recent_n: int = _DEFAULT_RECENT_N,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """
    Build a single-player profile card.

    Args:
        name_or_id: Player name (full or partial) or nflverse player_id.
        seasons:    List of seasons to include. None = all available.
        recent_n:   Number of most recent game logs for form calculation.
        engine:     SQLAlchemy Engine. Uses get_engine() if None.

    Returns:
        Dict with keys: player_id, name, position, team, seasons_covered,
        stats, totals, per_game, recent_form, percentiles, ranks.
        All missing values are None rather than omitted.
    """
    if engine is None:
        from nfl_chatbot.data.database import get_engine
        engine = get_engine()

    index = _load_player_index(engine)
    info = _resolve_player(name_or_id, index)

    if info is None:
        logger.warning("Player not found: %r", name_or_id)
        return {
            "error": f"Player not found: {name_or_id!r}",
            "name": name_or_id,
            "player_id": None,
            "position": None,
            "team": None,
        }

    position = (info.get("position") or "").upper()
    peers = _load_peer_season_rows(position, seasons, engine)

    return _build_profile(info, seasons, recent_n, engine, peers)


def compare_players(
    player_a: str,
    player_b: str,
    seasons: list[int] | None = None,
    engine: Engine | None = None,
    recent_n: int = _DEFAULT_RECENT_N,
) -> dict[str, Any]:
    """
    Compare two NFL players across all tracked metrics.

    Args:
        player_a: Name or player_id of the first player.
        player_b: Name or player_id of the second player.
        seasons:  Seasons to include. None = all available.
        engine:   SQLAlchemy Engine. Uses get_engine() if None.
        recent_n: Games for recent-form window (default 4).

    Returns a dict with the following top-level keys:

        player_a        Full profile card for player A (see build_player_profile).
        player_b        Full profile card for player B.
        comparison      Per-metric head-to-head dicts, including advantage and diff.
        summary         Edge counts, position match flag, overall winner heuristic.
        metadata        seasons, recent_n, position context, generated_at.

    All numeric values are Python int/float or None (never NaN/inf), so the
    result can be serialised directly with json.dumps().
    """
    if engine is None:
        from nfl_chatbot.data.database import get_engine
        engine = get_engine()

    index = _load_player_index(engine)

    info_a = _resolve_player(player_a, index)
    info_b = _resolve_player(player_b, index)

    errors: list[str] = []
    if info_a is None:
        errors.append(f"Player not found: {player_a!r}")
    if info_b is None:
        errors.append(f"Player not found: {player_b!r}")
    if errors:
        return {"error": " | ".join(errors), "player_a": None, "player_b": None}

    # Determine shared position for comparison metrics
    pos_a = (info_a.get("position") or "").upper()
    pos_b = (info_b.get("position") or "").upper()
    position = pos_a if pos_a == pos_b else pos_a  # use A's position when mixed

    # Load peer data once (covers both players if same position)
    peers = _load_peer_season_rows(position, seasons, engine)

    profile_a = _build_profile(info_a, seasons, recent_n, engine, peers)
    profile_b = _build_profile(info_b, seasons, recent_n, engine, peers)

    comparison = _build_comparison(profile_a, profile_b, position)

    # Overall heuristic: most edges wins; fantasy points_per_game as tiebreaker
    fpg_a = profile_a["per_game"].get("fantasy_points_per_game") or 0.0
    fpg_b = profile_b["per_game"].get("fantasy_points_per_game") or 0.0
    ec_a = comparison["player_a_edge_count"]
    ec_b = comparison["player_b_edge_count"]
    if ec_a > ec_b:
        overall_edge = info_a.get("player_name")
    elif ec_b > ec_a:
        overall_edge = info_b.get("player_name")
    elif fpg_a != fpg_b:
        overall_edge = info_a["player_name"] if fpg_a > fpg_b else info_b["player_name"]
    else:
        overall_edge = "even"

    summary = {
        "player_a_name": info_a.get("player_name"),
        "player_b_name": info_b.get("player_name"),
        "player_a_position": pos_a or None,
        "player_b_position": pos_b or None,
        "same_position": pos_a == pos_b,
        "comparison_position": position or None,
        "player_a_edges": comparison["player_a_edges"],
        "player_b_edges": comparison["player_b_edges"],
        "player_a_edge_count": ec_a,
        "player_b_edge_count": ec_b,
        "overall_edge": overall_edge,
    }

    from datetime import date
    return {
        "player_a": profile_a,
        "player_b": profile_b,
        "comparison": comparison["metrics"],
        "summary": summary,
        "metadata": {
            "seasons": seasons,
            "recent_n_games": recent_n,
            "position_context": position or None,
            "generated_at": date.today().isoformat(),
        },
    }
