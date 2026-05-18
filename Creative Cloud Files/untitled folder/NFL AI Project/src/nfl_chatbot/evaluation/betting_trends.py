"""
Betting trend analysis for the NFL AI Chatbot.

Answers questions like:
  - Which teams cover most often at home?
  - How often do underdogs cover?
  - What happens when the spread is greater than 7?
  - What teams go over the total most often?
  - Did the model outperform random guessing?

All public functions accept either a ``BettingTrends`` instance (for chaining)
or a raw DataFrame. The DataFrame must have at minimum:

    game_id, season, week, home_team, away_team,
    home_score, away_score, spread_line, total_line

``home_covered`` and ``over_hit`` are derived automatically if absent.

Spread convention (nflverse)
-----------------------------
``spread_line`` is from the *home* team's perspective.
  Negative  → home team is favored  (e.g. -6.5 = home gives 6.5 pts)
  Positive  → away team is favored  (e.g. +3.0 = home gets 3 pts)

  home_covered = 1  iff  home_score + spread_line > away_score
  over_hit     = 1  iff  home_score + away_score  > total_line

Return format
-------------
Every analysis function returns a dict::

    {
        "summary": {...},        # top-level statistics
        "table": pd.DataFrame,   # structured table for display / export
        "plot_path": Path | None # path to saved figure, or None
    }

Plots are saved under <plot_dir> (default: data/processed/plots/).
They require matplotlib; if matplotlib is absent the plot key is silently None.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spread buckets (abs value of spread_line)
_SPREAD_BUCKETS: list[tuple[str, float, float]] = [
    ("Pick'em (0–1)", 0.0, 1.0),
    ("Small (1.5–3)", 1.5, 3.0),
    ("Field Goal (3.5–6.5)", 3.5, 6.5),
    ("Touchdown (7–10)", 7.0, 10.0),
    ("Large (10.5–14)", 10.5, 14.0),
    ("Blowout (14+)", 14.5, 99.0),
]

_MIN_GAMES = 10  # minimum sample for reliable rates


# ── Statistical helpers ────────────────────────────────────────────────────────

def _wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, center - margin), min(1.0, center + margin)


def _rate_row(mask: pd.Series, label_mask: pd.Series | None = None) -> dict:
    """
    Compute count, covers, rate, and 95% CI from two boolean masks.

    *mask*        — rows included in the denominator
    *label_mask*  — rows where the event occurred (defaults to mask itself)
    """
    if label_mask is None:
        label_mask = mask
    n = int(mask.sum())
    k = int((mask & label_mask).sum())
    lo, hi = _wilson_ci(n, k)
    return {
        "games": n,
        "covers": k,
        "rate": round(k / n, 4) if n > 0 else None,
        "ci_95_lo": round(lo, 4),
        "ci_95_hi": round(hi, 4),
    }


def _bucket_label(abs_spread: float) -> str:
    for label, lo, hi in _SPREAD_BUCKETS:
        if lo <= abs_spread <= hi:
            return label
    return "Large (10.5–14)" if abs_spread <= 14 else "Blowout (14+)"


# ── Plot helper ────────────────────────────────────────────────────────────────

def _ensure_plot_dir(out_dir: Path | str | None) -> Path:
    if out_dir is None:
        try:
            from nfl_chatbot.config import get_config
            base = Path(get_config().app.data.processed_dir)
        except Exception:
            base = Path("data/processed")
        out_dir = base / "plots"
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ── BettingTrends class ────────────────────────────────────────────────────────

class BettingTrends:
    """
    Betting trend analytics for a dataset of NFL games.

    Parameters
    ----------
    df : pd.DataFrame
        Games table. Must include game_id, season, week, home_team, away_team,
        home_score, away_score, spread_line, total_line.
        ``home_covered`` and ``over_hit`` are derived automatically if absent.
    plot_dir : Path | str | None
        Directory for saved figures. Defaults to data/processed/plots/.

    Usage::

        trends = BettingTrends.from_csv("data/processed/modeling_table.csv")
        print(trends.favorite_cover_rate()["summary"])
        trends.plot_cover_rates_by_team()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        plot_dir: Path | str | None = None,
    ) -> None:
        self._raw = df.copy()
        self._df = self._prepare(df)
        self._plot_dir = plot_dir

    # ── Factories ──────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "BettingTrends":
        """Load from a modeling_table.csv produced by build_features.py."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Modeling table not found: {p}")
        return cls(pd.read_csv(p), **kwargs)

    @classmethod
    def from_engine(cls, engine, **kwargs) -> "BettingTrends":
        """Query games + odds directly from the database."""
        query = """
            SELECT
                g.game_id, g.season, g.week, g.game_type,
                g.home_team, g.away_team,
                g.home_score, g.away_score,
                g.home_win, g.spread_line, g.total_line,
                ef.home_covered
            FROM games g
            LEFT JOIN engineered_features ef USING (game_id)
            WHERE g.home_score IS NOT NULL
        """
        df = pd.read_sql(query, engine)
        return cls(df, **kwargs)

    # ── Data preparation ───────────────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required = {"home_score", "away_score", "spread_line", "total_line"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("BettingTrends: missing columns %s — some analyses will be empty", missing)

        # ── Derive home_covered ────────────────────────────────────────────────
        if "home_covered" not in df.columns or df["home_covered"].isna().all():
            has_spread = (
                df["home_score"].notna()
                & df["away_score"].notna()
                & df["spread_line"].notna()
            ) if all(c in df.columns for c in ("home_score", "away_score", "spread_line")) else pd.Series(False, index=df.index)

            df["home_covered"] = np.nan
            if has_spread.any():
                df.loc[has_spread, "home_covered"] = (
                    (df.loc[has_spread, "home_score"] + df.loc[has_spread, "spread_line"])
                    > df.loc[has_spread, "away_score"]
                ).astype(float)
        else:
            df["home_covered"] = pd.to_numeric(df["home_covered"], errors="coerce")

        # ── Derive over_hit ────────────────────────────────────────────────────
        if "over_hit" not in df.columns or df["over_hit"].isna().all():
            has_total = (
                df["home_score"].notna()
                & df["away_score"].notna()
                & df["total_line"].notna()
            ) if all(c in df.columns for c in ("home_score", "away_score", "total_line")) else pd.Series(False, index=df.index)

            df["over_hit"] = np.nan
            if has_total.any():
                df.loc[has_total, "over_hit"] = (
                    (df.loc[has_total, "home_score"] + df.loc[has_total, "away_score"])
                    > df.loc[has_total, "total_line"]
                ).astype(float)
        else:
            df["over_hit"] = pd.to_numeric(df["over_hit"], errors="coerce")

        # ── Derived columns ────────────────────────────────────────────────────
        if "spread_line" in df.columns:
            df["abs_spread"] = df["spread_line"].abs()
            df["favorite_is_home"] = df["spread_line"] < 0
            df["spread_bucket"] = df["abs_spread"].apply(
                lambda x: _bucket_label(x) if pd.notna(x) else None
            )
            # favorite_covered: home covered when home is fav; away covered when away is fav
            fav_home = df["favorite_is_home"].fillna(False)
            df["favorite_covered"] = np.where(
                fav_home,
                df["home_covered"],
                1.0 - df["home_covered"],
            )
            df.loc[df["home_covered"].isna(), "favorite_covered"] = np.nan

        if "home_score" in df.columns and "away_score" in df.columns:
            df["total_points"] = df["home_score"].add(df["away_score"])

        return df

    # ── Filtered view ──────────────────────────────────────────────────────────

    def _data(self, seasons: list[int] | int | None = None) -> pd.DataFrame:
        df = self._df
        if seasons is not None:
            s = [seasons] if isinstance(seasons, int) else list(seasons)
            df = df[df["season"].isin(s)]
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # Public analysis methods
    # ══════════════════════════════════════════════════════════════════════════

    def favorite_cover_rate(
        self,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        How often do favorites cover the spread?

        Returns summary + breakdown table (home_fav vs road_fav).
        """
        df = self._data(seasons)
        spread_df = df[df["spread_line"].notna() & df["home_covered"].notna()].copy()

        has_spread = pd.Series(True, index=spread_df.index)
        fav_home = spread_df["favorite_is_home"]
        fav_covered = spread_df["favorite_covered"].astype(bool)

        overall = _rate_row(has_spread, fav_covered)
        home_fav = _rate_row(fav_home, fav_covered)
        road_fav = _rate_row(~fav_home, fav_covered)

        table = pd.DataFrame([
            {"group": "All Favorites", **overall},
            {"group": "Home Favorite", **home_fav},
            {"group": "Road Favorite", **road_fav},
        ])
        table["rate_pct"] = (table["rate"] * 100).round(1).astype(str) + "%"
        table["ci_95"] = (
            "(" + (table["ci_95_lo"] * 100).round(1).astype(str) + "%, "
            + (table["ci_95_hi"] * 100).round(1).astype(str) + "%)"
        )

        summary = {
            "favorites_cover_rate": overall["rate"],
            "home_favorite_cover_rate": home_fav["rate"],
            "road_favorite_cover_rate": road_fav["rate"],
            "total_games": overall["games"],
            "note": (
                "Favorites cover roughly 50–54% of the time; "
                "the market is efficient but slight edges exist."
            ),
        }

        plot_path = self.plot_cover_rates_by_team(plot_dir=plot_dir or self._plot_dir)

        return {"summary": summary, "table": table, "plot_path": plot_path}

    def underdog_cover_rate(
        self,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """How often do underdogs cover the spread?"""
        df = self._data(seasons)
        spread_df = df[df["spread_line"].notna() & df["home_covered"].notna()].copy()

        has_spread = pd.Series(True, index=spread_df.index)
        fav_home = spread_df["favorite_is_home"]
        # Underdog covered = favorite did NOT cover
        dog_covered = ~spread_df["favorite_covered"].astype(bool)

        overall = _rate_row(has_spread, dog_covered)
        home_dog = _rate_row(~fav_home, dog_covered)  # home is underdog
        road_dog = _rate_row(fav_home, dog_covered)   # away is underdog

        table = pd.DataFrame([
            {"group": "All Underdogs", **overall},
            {"group": "Home Underdog", **home_dog},
            {"group": "Road Underdog", **road_dog},
        ])
        table["rate_pct"] = (table["rate"] * 100).round(1).astype(str) + "%"

        summary = {
            "underdogs_cover_rate": overall["rate"],
            "home_underdog_cover_rate": home_dog["rate"],
            "road_underdog_cover_rate": road_dog["rate"],
            "total_games": overall["games"],
            "note": (
                "Underdogs cover ≈ 46–50% — the mirror of favorite coverage, "
                "adjusted for pushes."
            ),
        }
        return {"summary": summary, "table": table, "plot_path": None}

    def home_favorite_cover_rate(
        self,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """Cover rate when the home team is the favorite (spread_line < 0)."""
        df = self._data(seasons)
        mask = df["favorite_is_home"] & df["home_covered"].notna()
        sub = df[mask]

        overall = _rate_row(pd.Series(True, index=sub.index), sub["home_covered"].astype(bool))

        # Breakdown by spread bucket
        rows = []
        for bucket, lo, hi in _SPREAD_BUCKETS:
            bm = mask & (df["abs_spread"] >= lo) & (df["abs_spread"] <= hi)
            sub_b = df[bm]
            if len(sub_b) < _MIN_GAMES:
                continue
            r = _rate_row(pd.Series(True, index=sub_b.index), sub_b["home_covered"].astype(bool))
            rows.append({"spread_bucket": bucket, "spread_range": f"{lo}–{hi}", **r})

        table = pd.DataFrame(rows) if rows else pd.DataFrame()
        summary = {
            "home_favorite_cover_rate": overall["rate"],
            "games": overall["games"],
            "ci_95": (overall["ci_95_lo"], overall["ci_95_hi"]),
            "note": "Home favorites tend to cover slightly more than road favorites.",
        }
        return {"summary": summary, "table": table, "plot_path": None}

    def road_underdog_cover_rate(
        self,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """Cover rate for road underdogs (away team is underdog: spread_line < 0)."""
        df = self._data(seasons)
        # Road underdog = home is favorite (spread_line < 0), but away (underdog) covers
        mask = df["favorite_is_home"].fillna(False) & df["home_covered"].notna()
        sub = df[mask]
        away_covered = ~sub["home_covered"].astype(bool)

        overall = _rate_row(pd.Series(True, index=sub.index), away_covered)

        # Breakdown by spread magnitude
        rows = []
        for bucket, lo, hi in _SPREAD_BUCKETS:
            bm = mask & (df["abs_spread"] >= lo) & (df["abs_spread"] <= hi)
            sub_b = df[bm]
            if len(sub_b) < _MIN_GAMES:
                continue
            dog_cov = ~sub_b["home_covered"].astype(bool)
            r = _rate_row(pd.Series(True, index=sub_b.index), dog_cov)
            rows.append({"spread_bucket": bucket, **r})

        table = pd.DataFrame(rows) if rows else pd.DataFrame()
        summary = {
            "road_underdog_cover_rate": overall["rate"],
            "games": overall["games"],
            "ci_95": (overall["ci_95_lo"], overall["ci_95_hi"]),
            "note": (
                "Road underdogs historically cover slightly more often than expected "
                "in large-spread games — teams may ease up or dogs play loose."
            ),
        }
        return {"summary": summary, "table": table, "plot_path": None}

    def over_under_hit_rate(
        self,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Overall over/under hit rate, plus breakdown by season and total bucket."""
        df = self._data(seasons)
        ou_df = df[df["over_hit"].notna() & df["total_line"].notna()]

        has_ou = pd.Series(True, index=ou_df.index)
        over_hit = ou_df["over_hit"].astype(bool)
        overall = _rate_row(has_ou, over_hit)

        # By season
        season_rows = []
        if "season" in ou_df.columns:
            for season, grp in ou_df.groupby("season"):
                r = _rate_row(pd.Series(True, index=grp.index), grp["over_hit"].astype(bool))
                avg_total = grp["total_line"].mean()
                season_rows.append({
                    "season": season,
                    "games": r["games"],
                    "over_count": r["covers"],
                    "over_rate": r["rate"],
                    "avg_total_line": round(avg_total, 1) if pd.notna(avg_total) else None,
                })
        season_table = pd.DataFrame(season_rows).sort_values("season") if season_rows else pd.DataFrame()

        # By total bucket
        total_rows = []
        for lo, hi in [(30, 38), (38.5, 44), (44.5, 48), (48.5, 52), (52.5, 99)]:
            label = f"{lo}–{hi}" if hi < 99 else f"{lo}+"
            bm = ou_df["total_line"].between(lo, hi)
            sub = ou_df[bm]
            if len(sub) < _MIN_GAMES:
                continue
            r = _rate_row(pd.Series(True, index=sub.index), sub["over_hit"].astype(bool))
            total_rows.append({"total_bucket": label, **r})
        total_table = pd.DataFrame(total_rows) if total_rows else pd.DataFrame()

        summary = {
            "overall_over_rate": overall["rate"],
            "overall_under_rate": round(1.0 - (overall["rate"] or 0.5), 4),
            "games": overall["games"],
            "ci_95": (overall["ci_95_lo"], overall["ci_95_hi"]),
            "avg_total_line": (
                round(ou_df["total_line"].mean(), 1)
                if "total_line" in ou_df.columns else None
            ),
            "note": "Over/under is designed for ~50% hit rate; public tends to bet overs.",
        }

        plot_path = self.plot_over_rates_by_team(plot_dir=plot_dir or self._plot_dir)

        return {
            "summary": summary,
            "table": season_table,
            "total_bucket_table": total_table,
            "plot_path": plot_path,
        }

    def team_cover_rate(
        self,
        team: str,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """
        Coverage rate for a specific team (home + away combined and split).

        *team* is a three-letter nflverse abbreviation (e.g. "KC", "NE").
        """
        df = self._data(seasons)
        team_u = team.upper()

        rows_home = df[(df["home_team"] == team_u) & df["home_covered"].notna()]
        rows_away = df[(df["away_team"] == team_u) & df["home_covered"].notna()]

        home_cov = rows_home["home_covered"].astype(bool)
        away_cov = ~rows_away["home_covered"].astype(bool)

        all_games = pd.concat([
            rows_home.assign(_covered=home_cov),
            rows_away.assign(_covered=away_cov),
        ])

        overall = _rate_row(
            pd.Series(True, index=all_games.index),
            all_games["_covered"].astype(bool),
        )
        as_home = _rate_row(
            pd.Series(True, index=rows_home.index), home_cov
        )
        as_away = _rate_row(
            pd.Series(True, index=rows_away.index), away_cov
        )

        # As favorite / as underdog
        fav_mask_home = rows_home["favorite_is_home"].fillna(False)
        fav_mask_away = ~rows_away["favorite_is_home"].fillna(True)

        as_fav_home = rows_home[fav_mask_home]
        as_fav_away = rows_away[fav_mask_away]
        as_dog_home = rows_home[~fav_mask_home]
        as_dog_away = rows_away[~fav_mask_away]

        as_favorite = _rate_row(
            pd.Series(True, index=pd.concat([as_fav_home, as_fav_away]).index),
            pd.concat([
                as_fav_home["home_covered"].astype(bool),
                (~as_fav_away["home_covered"].astype(bool)),
            ]),
        )
        as_underdog = _rate_row(
            pd.Series(True, index=pd.concat([as_dog_home, as_dog_away]).index),
            pd.concat([
                as_dog_home["home_covered"].astype(bool),
                (~as_dog_away["home_covered"].astype(bool)),
            ]),
        )

        table = pd.DataFrame([
            {"split": "Overall", **overall},
            {"split": "As Home Team", **as_home},
            {"split": "As Away Team", **as_away},
            {"split": "As Favorite", **as_favorite},
            {"split": "As Underdog", **as_underdog},
        ])
        table["rate_pct"] = (table["rate"] * 100).round(1).astype(str) + "%"

        summary = {
            "team": team_u,
            "overall_cover_rate": overall["rate"],
            "home_cover_rate": as_home["rate"],
            "away_cover_rate": as_away["rate"],
            "as_favorite_cover_rate": as_favorite["rate"],
            "as_underdog_cover_rate": as_underdog["rate"],
            "total_games": overall["games"],
        }
        return {"summary": summary, "table": table, "plot_path": None}

    def team_over_rate(
        self,
        team: str,
        seasons: list[int] | int | None = None,
    ) -> dict[str, Any]:
        """How often do this team's games go over the total?"""
        df = self._data(seasons)
        team_u = team.upper()

        rows = df[
            ((df["home_team"] == team_u) | (df["away_team"] == team_u))
            & df["over_hit"].notna()
        ]
        over = rows["over_hit"].astype(bool)
        overall = _rate_row(pd.Series(True, index=rows.index), over)

        # By season
        season_rows = []
        if "season" in rows.columns:
            for season, grp in rows.groupby("season"):
                r = _rate_row(pd.Series(True, index=grp.index), grp["over_hit"].astype(bool))
                avg_pts = grp["total_points"].mean() if "total_points" in grp.columns else None
                season_rows.append({
                    "season": season,
                    "games": r["games"],
                    "over_count": r["covers"],
                    "over_rate": r["rate"],
                    "avg_total_pts": round(avg_pts, 1) if avg_pts is not None and pd.notna(avg_pts) else None,
                })

        season_table = pd.DataFrame(season_rows).sort_values("season") if season_rows else pd.DataFrame()

        summary = {
            "team": team_u,
            "overall_over_rate": overall["rate"],
            "games": overall["games"],
            "ci_95": (overall["ci_95_lo"], overall["ci_95_hi"]),
            "avg_total_pts": (
                round(rows["total_points"].mean(), 1)
                if "total_points" in rows.columns else None
            ),
        }
        return {"summary": summary, "table": season_table, "plot_path": None}

    def spread_bucket_analysis(
        self,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Coverage and over rates broken down by spread size.

        Answers: "What happens when the spread is greater than 7?"
        """
        df = self._data(seasons)
        spread_df = df[df["spread_line"].notna() & df["abs_spread"].notna()]

        rows = []
        for bucket, lo, hi in _SPREAD_BUCKETS:
            bm = spread_df["abs_spread"].between(lo, hi, inclusive="both")
            sub = spread_df[bm]
            if len(sub) < _MIN_GAMES:
                continue

            cov_mask = sub["home_covered"].notna()
            cov = _rate_row(cov_mask, sub["home_covered"][cov_mask].astype(bool))

            fav_cov_mask = sub["favorite_covered"].notna()
            fav_cov = _rate_row(fav_cov_mask, sub["favorite_covered"][fav_cov_mask].astype(bool))

            ou_mask = sub["over_hit"].notna()
            ou = _rate_row(ou_mask, sub["over_hit"][ou_mask].astype(bool)) if ou_mask.any() else {"rate": None, "games": 0, "covers": 0}

            rows.append({
                "spread_bucket": bucket,
                "n_games": len(sub),
                "home_cover_rate": cov["rate"],
                "favorite_cover_rate": fav_cov["rate"],
                "over_rate": ou.get("rate"),
                "avg_total_pts": (
                    round(sub["total_points"].mean(), 1)
                    if "total_points" in sub.columns and sub["total_points"].notna().any()
                    else None
                ),
                "avg_spread": round(sub["abs_spread"].mean(), 1),
            })

        table = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not table.empty:
            for col in ("home_cover_rate", "favorite_cover_rate", "over_rate"):
                if col in table.columns:
                    table[f"{col}_pct"] = (table[col] * 100).round(1).astype(str) + "%"

        summary = {
            "buckets_analyzed": len(rows),
            "note": (
                "Coverage rate by spread size reveals market efficiency: "
                "large spreads (7+) may show different dynamics than close games."
            ),
        }

        plot_path = self.plot_spread_buckets(
            table=table, plot_dir=plot_dir or self._plot_dir
        )

        return {"summary": summary, "table": table, "plot_path": plot_path}

    def model_edge_analysis(
        self,
        predictions_df: pd.DataFrame | None = None,
        metrics_path: str | Path | None = None,
        spread_metrics_path: str | Path | None = None,
        plot_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Compare model performance against baselines.

        Answers: "Did the model outperform random guessing?"

        Baselines:
          - always_favorite  : predict the favorite wins / covers every game
          - always_home      : predict home team wins every game
          - random           : 50% accuracy by definition

        If *predictions_df* is provided it must have columns:
            predicted_winner, actual_winner  (or home_covered equivalents)

        Otherwise loads saved metrics from *metrics_path* (model_metrics.json).
        """
        results: dict[str, Any] = {}

        # ── Load saved model metrics ───────────────────────────────────────────
        def _load_metrics(path: Path | str | None, default_name: str) -> dict:
            if path is None:
                try:
                    from nfl_chatbot.config import get_config
                    model_dir = Path(get_config().app.models.output_dir)
                except Exception:
                    model_dir = Path("models")
                path = model_dir / default_name
            p = Path(path)
            if not p.exists():
                return {}
            try:
                doc = json.loads(p.read_text())
                best = doc.get("best_model", "")
                m = doc.get("models", {}).get(best, {})
                return {
                    "model_name": best,
                    "accuracy": m.get("accuracy"),
                    "roc_auc": m.get("roc_auc"),
                    "brier_score": m.get("brier_score"),
                    "f1": m.get("f1"),
                    "test_season": doc.get("split", {}).get("test_season"),
                    "train_rows": doc.get("split", {}).get("train_rows"),
                    "test_rows": doc.get("split", {}).get("test_rows"),
                    "task": doc.get("task", "unknown"),
                }
            except Exception as exc:
                logger.warning("Could not load metrics from %s: %s", p, exc)
                return {}

        win_metrics = _load_metrics(metrics_path, "model_metrics.json")
        spread_metrics = _load_metrics(spread_metrics_path, "spread_model_metrics.json")

        # ── Compute baselines from data ────────────────────────────────────────
        df = self._df
        spread_df = df[df["spread_line"].notna() & df["home_covered"].notna()]
        games_df = df[df["home_win"].notna()] if "home_win" in df.columns else pd.DataFrame()

        # Always-home baseline
        home_win_rate = float(games_df["home_win"].mean()) if len(games_df) > 0 else 0.575
        # Always-favorite baseline (favorite covers rate)
        fav_cover_rate = float(spread_df["favorite_covered"].mean()) if len(spread_df) > 0 else 0.52

        baselines = {
            "random": {"accuracy": 0.50, "note": "50/50 guess"},
            "always_home_wins": {
                "accuracy": round(home_win_rate, 4),
                "note": f"Always predict home team wins ({home_win_rate:.1%} of time)",
            },
            "always_favorite_covers": {
                "accuracy": round(fav_cover_rate, 4),
                "note": f"Always predict favorite covers ({fav_cover_rate:.1%} of time)",
            },
        }

        # ── Compute model edge ─────────────────────────────────────────────────
        comparison_rows = []

        if win_metrics.get("accuracy") is not None:
            acc = win_metrics["accuracy"]
            comparison_rows += [
                {
                    "model": f"Win Predictor ({win_metrics.get('model_name', 'best')})",
                    "accuracy": acc,
                    "roc_auc": win_metrics.get("roc_auc"),
                    "vs_random": round(acc - 0.50, 4),
                    "vs_always_home": round(acc - home_win_rate, 4),
                    "test_season": win_metrics.get("test_season"),
                },
            ]
        if spread_metrics.get("accuracy") is not None:
            acc = spread_metrics["accuracy"]
            comparison_rows += [
                {
                    "model": f"ATS Predictor ({spread_metrics.get('model_name', 'best')})",
                    "accuracy": acc,
                    "roc_auc": spread_metrics.get("roc_auc"),
                    "vs_random": round(acc - 0.50, 4),
                    "vs_always_favorite": round(acc - fav_cover_rate, 4),
                    "test_season": spread_metrics.get("test_season"),
                },
            ]

        for name, b in baselines.items():
            comparison_rows.append({
                "model": f"Baseline: {name}",
                "accuracy": b["accuracy"],
                "roc_auc": None,
                "note": b["note"],
            })

        table = pd.DataFrame(comparison_rows) if comparison_rows else pd.DataFrame()

        summary = {
            "win_model": win_metrics,
            "spread_model": spread_metrics,
            "baselines": baselines,
            "win_vs_random": (
                round(win_metrics["accuracy"] - 0.50, 4)
                if win_metrics.get("accuracy") else None
            ),
            "spread_vs_random": (
                round(spread_metrics["accuracy"] - 0.50, 4)
                if spread_metrics.get("accuracy") else None
            ),
            "note": (
                "Win accuracy >60% is solid for NFL. "
                "ATS accuracy >52% is meaningful given efficient markets."
            ),
        }

        plot_path = self.plot_model_edge(
            win_metrics=win_metrics,
            baselines=baselines,
            plot_dir=plot_dir or self._plot_dir,
        )

        return {"summary": summary, "table": table, "plot_path": plot_path}

    # ── All-team aggregates (for ranking tables) ────────────────────────────────

    def all_teams_cover_rate(
        self,
        seasons: list[int] | int | None = None,
        min_games: int = _MIN_GAMES,
    ) -> pd.DataFrame:
        """
        Cover rate for every team that appears in the data.

        Returns a DataFrame sorted by cover_rate descending, useful for
        building the "which teams cover most" ranking table.
        """
        df = self._data(seasons)
        spread_df = df[df["spread_line"].notna() & df["home_covered"].notna()]

        teams = set(spread_df["home_team"].dropna()) | set(spread_df["away_team"].dropna())
        rows = []
        for team in sorted(teams):
            home = spread_df[spread_df["home_team"] == team]
            away = spread_df[spread_df["away_team"] == team]
            home_c = home["home_covered"].astype(bool)
            away_c = ~away["home_covered"].astype(bool)

            all_cov = pd.concat([home_c, away_c])
            n = len(all_cov)
            k = int(all_cov.sum())
            if n < min_games:
                continue

            lo, hi = _wilson_ci(n, k)
            rows.append({
                "team": team,
                "games": n,
                "covers": k,
                "cover_rate": round(k / n, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "home_cover_rate": round(home_c.mean(), 4) if len(home_c) else None,
                "away_cover_rate": round(away_c.mean(), 4) if len(away_c) else None,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("cover_rate", ascending=False)
            .reset_index(drop=True)
        )

    def all_teams_over_rate(
        self,
        seasons: list[int] | int | None = None,
        min_games: int = _MIN_GAMES,
    ) -> pd.DataFrame:
        """Over rate for every team, sorted by over_rate descending."""
        df = self._data(seasons)
        ou_df = df[df["over_hit"].notna()]

        teams = set(ou_df["home_team"].dropna()) | set(ou_df["away_team"].dropna())
        rows = []
        for team in sorted(teams):
            sub = ou_df[(ou_df["home_team"] == team) | (ou_df["away_team"] == team)]
            n = len(sub)
            k = int(sub["over_hit"].sum())
            if n < min_games:
                continue

            lo, hi = _wilson_ci(n, k)
            avg_pts = sub["total_points"].mean() if "total_points" in sub.columns else None
            rows.append({
                "team": team,
                "games": n,
                "over_count": k,
                "over_rate": round(k / n, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "avg_total_pts": round(avg_pts, 1) if avg_pts is not None and pd.notna(avg_pts) else None,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("over_rate", ascending=False)
            .reset_index(drop=True)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Plot methods
    # ══════════════════════════════════════════════════════════════════════════

    def plot_cover_rates_by_team(
        self,
        n_top: int = 20,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> Path | None:
        """
        Horizontal bar chart: top and bottom N teams by cover rate.
        Saved to <plot_dir>/cover_rates_by_team.png.
        """
        plt = _try_import_mpl()
        if plt is None:
            return None

        rank_df = self.all_teams_cover_rate(seasons=seasons)
        if rank_df.empty:
            return None

        out_dir = _ensure_plot_dir(plot_dir or self._plot_dir)
        out_path = out_dir / "cover_rates_by_team.png"

        # Show top N and bottom N, joined with a gap marker
        top = rank_df.head(n_top)
        bot = rank_df.tail(n_top).iloc[::-1]
        combined = pd.concat([top, bot]).drop_duplicates("team")

        fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.4)))
        colors = [
            "#2196F3" if r >= 0.5 else "#EF5350"
            for r in combined["cover_rate"]
        ]
        bars = ax.barh(combined["team"], combined["cover_rate"] * 100, color=colors, edgecolor="white")
        ax.axvline(50, color="black", linewidth=1.2, linestyle="--", label="50% line")

        # Error bars
        xerr = [
            (combined["cover_rate"] - combined["ci_lo"]) * 100,
            (combined["ci_hi"] - combined["cover_rate"]) * 100,
        ]
        ax.errorbar(
            combined["cover_rate"] * 100, combined["team"],
            xerr=xerr, fmt="none", color="gray", capsize=3, linewidth=1,
        )

        ax.set_xlabel("Cover Rate (%)", fontsize=11)
        ax.set_title("NFL Team Cover Rates (ATS)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(30, 75)

        for bar, row in zip(bars, combined.itertuples()):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row.cover_rate*100:.1f}% ({row.games}g)",
                va="center", fontsize=7.5,
            )

        _seasons_label = f"  (seasons: {seasons})" if seasons else ""
        ax.set_title(f"NFL Team Cover Rates (ATS){_seasons_label}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cover-rates plot → %s", out_path)
        return out_path

    def plot_spread_buckets(
        self,
        table: pd.DataFrame | None = None,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> Path | None:
        """
        Bar chart: favorite cover rate and over rate by spread bucket.
        Saved to <plot_dir>/spread_bucket_analysis.png.

        *table* may be passed directly to avoid recomputing; if None the bucket
        analysis is computed fresh from *seasons*.
        """
        plt = _try_import_mpl()
        if plt is None:
            return None

        if table is None:
            result = self.spread_bucket_analysis(seasons=seasons)
            table = result["table"]
        if table is None or table.empty:
            return None

        out_dir = _ensure_plot_dir(plot_dir or self._plot_dir)
        out_path = out_dir / "spread_bucket_analysis.png"

        x = range(len(table))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 5))
        bar1 = ax.bar(
            [xi - width / 2 for xi in x],
            (table["favorite_cover_rate"] * 100).fillna(0),
            width=width, label="Favorite Cover %", color="#42A5F5", edgecolor="white",
        )
        bar2 = ax.bar(
            [xi + width / 2 for xi in x],
            (table["over_rate"].fillna(0) * 100),
            width=width, label="Over %", color="#66BB6A", edgecolor="white",
        )

        ax.axhline(50, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(table["spread_bucket"], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Rate (%)", fontsize=11)
        ax.set_ylim(30, 75)
        ax.set_title("Cover & Over Rates by Spread Bucket", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)

        for bar in [bar1, bar2]:
            for b in bar:
                h = b.get_height()
                if h > 0:
                    ax.text(
                        b.get_x() + b.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
                    )

        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved spread-bucket plot → %s", out_path)
        return out_path

    def plot_over_rates_by_team(
        self,
        n_top: int = 20,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> Path | None:
        """
        Horizontal bar chart: teams ranked by over/under rate.
        Saved to <plot_dir>/over_rates_by_team.png.
        """
        plt = _try_import_mpl()
        if plt is None:
            return None

        rank_df = self.all_teams_over_rate(seasons=seasons)
        if rank_df.empty:
            return None

        out_dir = _ensure_plot_dir(plot_dir or self._plot_dir)
        out_path = out_dir / "over_rates_by_team.png"

        top = rank_df.head(n_top)
        bot = rank_df.tail(n_top).iloc[::-1]
        combined = pd.concat([top, bot]).drop_duplicates("team")

        fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.4)))
        colors = [
            "#4CAF50" if r >= 0.5 else "#FF7043"
            for r in combined["over_rate"]
        ]
        ax.barh(combined["team"], combined["over_rate"] * 100, color=colors, edgecolor="white")
        ax.axvline(50, color="black", linewidth=1.2, linestyle="--", label="50% line")

        ax.set_xlabel("Over Rate (%)", fontsize=11)
        ax.set_title("NFL Team Over/Under Rates", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(25, 80)
        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved over-rates plot → %s", out_path)
        return out_path

    def plot_model_edge(
        self,
        win_metrics: dict | None = None,
        baselines: dict | None = None,
        plot_dir: Path | str | None = None,
    ) -> Path | None:
        """
        Bar chart comparing model accuracy vs baselines.
        Saved to <plot_dir>/model_edge.png.
        """
        plt = _try_import_mpl()
        if plt is None:
            return None

        if not win_metrics or win_metrics.get("accuracy") is None:
            return None

        out_dir = _ensure_plot_dir(plot_dir or self._plot_dir)
        out_path = out_dir / "model_edge.png"

        baselines = baselines or {}
        labels, accs, colors = [], [], []

        # Baseline bars first
        for name, b in baselines.items():
            acc = b.get("accuracy")
            if acc is None:
                continue
            labels.append(name.replace("_", "\n"))
            accs.append(acc * 100)
            colors.append("#B0BEC5")

        # Model bar
        labels.append(f"Win Model\n({win_metrics.get('model_name', 'best')})")
        accs.append(win_metrics["accuracy"] * 100)
        colors.append("#1565C0")

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, accs, color=colors, edgecolor="white", width=0.55)
        ax.axhline(50, color="red", linewidth=1.2, linestyle="--", label="Random (50%)")

        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("Model Accuracy vs Baselines", fontsize=13, fontweight="bold")
        ax.set_ylim(40, max(accs) + 8)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved model-edge plot → %s", out_path)
        return out_path

    # ── Summary report ─────────────────────────────────────────────────────────

    def summary_report(
        self,
        seasons: list[int] | int | None = None,
        plot_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Run all analyses and return a combined report dict.

        Useful for batch export or chatbot context pre-loading.
        """
        return {
            "favorite_cover_rate": self.favorite_cover_rate(seasons=seasons, plot_dir=plot_dir),
            "underdog_cover_rate": self.underdog_cover_rate(seasons=seasons),
            "home_favorite_cover_rate": self.home_favorite_cover_rate(seasons=seasons),
            "road_underdog_cover_rate": self.road_underdog_cover_rate(seasons=seasons),
            "over_under_hit_rate": self.over_under_hit_rate(seasons=seasons, plot_dir=plot_dir),
            "spread_bucket_analysis": self.spread_bucket_analysis(seasons=seasons, plot_dir=plot_dir),
            "model_edge_analysis": self.model_edge_analysis(plot_dir=plot_dir),
            "all_teams_cover_rate": self.all_teams_cover_rate(seasons=seasons),
            "all_teams_over_rate": self.all_teams_over_rate(seasons=seasons),
        }


# ── Module-level convenience functions ────────────────────────────────────────
# Each wraps the corresponding BettingTrends method so callers can skip
# creating an instance for one-off queries.

def _trends(df_or_path) -> BettingTrends:
    if isinstance(df_or_path, BettingTrends):
        return df_or_path
    if isinstance(df_or_path, pd.DataFrame):
        return BettingTrends(df_or_path)
    return BettingTrends.from_csv(df_or_path)


def favorite_cover_rate(df, seasons=None, **kw) -> dict:
    """Overall favorite cover rate. See BettingTrends.favorite_cover_rate."""
    return _trends(df).favorite_cover_rate(seasons=seasons, **kw)


def underdog_cover_rate(df, seasons=None, **kw) -> dict:
    """Overall underdog cover rate. See BettingTrends.underdog_cover_rate."""
    return _trends(df).underdog_cover_rate(seasons=seasons, **kw)


def home_favorite_cover_rate(df, seasons=None, **kw) -> dict:
    """Cover rate when home team is the favorite."""
    return _trends(df).home_favorite_cover_rate(seasons=seasons, **kw)


def road_underdog_cover_rate(df, seasons=None, **kw) -> dict:
    """Cover rate for road underdogs."""
    return _trends(df).road_underdog_cover_rate(seasons=seasons, **kw)


def over_under_hit_rate(df, seasons=None, **kw) -> dict:
    """Overall over/under hit rate. See BettingTrends.over_under_hit_rate."""
    return _trends(df).over_under_hit_rate(seasons=seasons, **kw)


def team_cover_rate(df, team: str, seasons=None) -> dict:
    """Cover rate for a specific team. See BettingTrends.team_cover_rate."""
    return _trends(df).team_cover_rate(team, seasons=seasons)


def team_over_rate(df, team: str, seasons=None) -> dict:
    """Over rate for a specific team's games. See BettingTrends.team_over_rate."""
    return _trends(df).team_over_rate(team, seasons=seasons)


def spread_bucket_analysis(df, seasons=None, **kw) -> dict:
    """Cover/over rates broken down by spread bucket."""
    return _trends(df).spread_bucket_analysis(seasons=seasons, **kw)


def model_edge_analysis(df, **kw) -> dict:
    """Model vs baseline comparison. See BettingTrends.model_edge_analysis."""
    return _trends(df).model_edge_analysis(**kw)
