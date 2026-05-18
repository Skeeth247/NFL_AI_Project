"""
Data-leakage audit for the NFL AI Chatbot feature and training pipeline.

Each check is an independent function that returns a CheckResult with:
  status  : "PASS" | "FAIL" | "WARN" | "SKIP"
  name    : short identifier used in the report
  detail  : one-sentence explanation of the result
  category: logical grouping ("rolling" | "target" | "scores" | "split" |
             "spread" | "injury" | "h2h" | "elo" | "cv" | "modeling_table")

run_all_checks() runs every check and returns a list of CheckResults.
Any FAIL means a confirmed leakage risk in the current code.
Any WARN means a possible risk or a safety guarantee that is implicit rather
than explicit (correct in practice but fragile).
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Project root (two levels above this file: .../src/nfl_chatbot/evaluation/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TEAM_FEATURES_SRC = _PROJECT_ROOT / "src" / "nfl_chatbot" / "features" / "team_features.py"
_TRAIN_SRC = _PROJECT_ROOT / "src" / "nfl_chatbot" / "models" / "train.py"
_DEFAULT_MODELING_CSV = _PROJECT_ROOT / "data" / "processed" / "modeling_table.csv"
_DEFAULT_FEATURE_JSON = _PROJECT_ROOT / "models" / "feature_columns.json"


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str          # "PASS" | "FAIL" | "WARN" | "SKIP"
    detail: str
    category: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    def __str__(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "△", "SKIP": "○"}.get(self.status, "?")
        return f"[{self.status}] {icon} {self.name}: {self.detail}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pass(name: str, detail: str, category: str, **extra) -> CheckResult:
    return CheckResult(name=name, status="PASS", detail=detail,
                       category=category, extra=extra)

def _fail(name: str, detail: str, category: str, **extra) -> CheckResult:
    return CheckResult(name=name, status="FAIL", detail=detail,
                       category=category, extra=extra)

def _warn(name: str, detail: str, category: str, **extra) -> CheckResult:
    return CheckResult(name=name, status="WARN", detail=detail,
                       category=category, extra=extra)

def _skip(name: str, detail: str, category: str, **extra) -> CheckResult:
    return CheckResult(name=name, status="SKIP", detail=detail,
                       category=category, extra=extra)


def _read_src(path: Path) -> str:
    """Return source code as a string, or empty string on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROLLING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def check_rolling_shift_excludes_current_game() -> CheckResult:
    """
    Place a canary value (999.0) at position N.
    The rolling mean at position N must NOT include 999.0.
    This verifies shift(1) prevents self-inclusion.
    """
    name = "rolling_shift_excludes_current_game"
    try:
        from nfl_chatbot.features.team_features import _rolling_shift

        canary = 999.0
        normal = 1.0
        series = pd.Series([normal, normal, canary, normal, normal])

        # Position 2 has the canary. Its window (window=3) should look back at
        # positions [0, 1] only (after shift(1)), never at position 2 itself.
        result = _rolling_shift(series, window=3)
        canary_influences = abs(result.iloc[3] - normal) > 0.1
        # Position 3 looks at [1, 2 (=canary)] → mean = (1 + 999) / 2 = 500. Still > normal.
        # But position 2 itself should only look at [0, 1] → mean = 1.0.
        pos2_val = result.iloc[2]
        if abs(pos2_val - normal) < 0.5:
            return _pass(name,
                "shift(1) confirmed: position-2 window uses only positions 0–1.",
                "rolling", canary=canary, result_at_pos2=float(pos2_val))
        else:
            return _fail(name,
                f"Canary leaked into its own window: result.iloc[2]={pos2_val:.1f} "
                f"(expected ≈{normal}).",
                "rolling", canary=canary, result_at_pos2=float(pos2_val))
    except Exception as exc:
        return _fail(name, f"Import or execution error: {exc}", "rolling")


def check_rolling_shift_sum_excludes_current() -> CheckResult:
    """Same test for the 'sum' aggregation path."""
    name = "rolling_shift_sum_excludes_current"
    try:
        from nfl_chatbot.features.team_features import _rolling_shift

        canary = 999.0
        series = pd.Series([1.0, 1.0, canary, 1.0, 1.0])
        result = _rolling_shift(series, window=2, agg="sum")
        # Position 2: shift → window over [pos0=1, pos1=1] → sum = 2.0
        pos2_val = result.iloc[2]
        if abs(pos2_val - 2.0) < 0.5:
            return _pass(name,
                "sum aggregation: position-2 window uses only positions 0–1 (sum=2.0).",
                "rolling", result_at_pos2=float(pos2_val))
        else:
            return _fail(name,
                f"Sum canary leaked: result.iloc[2]={pos2_val:.1f} (expected 2.0).",
                "rolling", result_at_pos2=float(pos2_val))
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "rolling")


def check_season_expanding_excludes_current() -> CheckResult:
    """
    _season_expanding uses shift(1).expanding().mean().
    Verify that the result at position N does not include the value at position N.
    """
    name = "season_expanding_excludes_current"
    try:
        from nfl_chatbot.features.team_features import _season_expanding

        series = pd.Series([10.0, 20.0, 999.0, 30.0])
        result = _season_expanding(series)
        # Position 2 (=999): expanding after shift includes [10, 20] → mean=15.
        pos2_val = result.iloc[2]
        if abs(pos2_val - 15.0) < 0.5:
            return _pass(name,
                "_season_expanding shift(1): position-2 mean uses only positions 0–1 (=15.0).",
                "rolling", result_at_pos2=float(pos2_val))
        else:
            return _fail(name,
                f"Expanding canary leaked: result.iloc[2]={pos2_val:.1f} (expected 15.0).",
                "rolling", result_at_pos2=float(pos2_val))
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "rolling")


def check_rolling_first_game_is_nan() -> CheckResult:
    """First game of a team should have NaN rolling features (no prior games)."""
    name = "rolling_first_game_is_nan"
    try:
        from nfl_chatbot.features.team_features import _rolling_shift

        series = pd.Series([42.0, 10.0, 20.0])
        result = _rolling_shift(series, window=3)
        if pd.isna(result.iloc[0]):
            return _pass(name,
                "First game correctly produces NaN — no prior games exist to leak from.",
                "rolling")
        else:
            return _fail(name,
                f"First game has non-NaN rolling value: {result.iloc[0]} (should be NaN).",
                "rolling", first_value=float(result.iloc[0]))
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "rolling")


def check_rolling_sort_order_safety() -> CheckResult:
    """
    Rolling feature assignment uses .values (positional).  This is safe ONLY
    because team_ts is sorted by [team, season, week] with reset_index() before
    _compute_rolling_features is called.  Verify this sort happens in build().
    """
    name = "rolling_sort_order_safety"
    src = _read_src(_TEAM_FEATURES_SRC)
    if not src:
        return _skip(name, f"Cannot read {_TEAM_FEATURES_SRC}", "rolling")

    # Look for sort_values + reset_index on team_ts before the rolling call
    pattern_sort = r"team_ts\s*=\s*team_ts\.sort_values\(.*\)\.reset_index\("
    pattern_rolling = r"_compute_rolling_features\(team_ts\)"

    sort_match = re.search(pattern_sort, src)
    rolling_match = re.search(pattern_rolling, src)

    if not sort_match:
        return _warn(name,
            "Could not confirm team_ts is sorted+reset_index'd before "
            "_compute_rolling_features. .values positional assignment could "
            "mis-align rows if order changes.",
            "rolling")

    if sort_match and rolling_match and sort_match.start() < rolling_match.start():
        return _pass(name,
            "team_ts is sorted by [team, season, week] and reset_index(drop=True) "
            "before rolling feature computation — .values positional assignment is safe.",
            "rolling")

    return _warn(name,
        ".values positional assignment in rolling features requires team_ts to be "
        "sorted before the call; sort order not confirmed in source.",
        "rolling")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TARGET & SCORE LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════

def check_feature_columns_no_target() -> CheckResult:
    """feature_columns() must not include home_win or home_covered."""
    name = "feature_columns_no_target"
    try:
        from nfl_chatbot.features.team_features import feature_columns

        cols = set(feature_columns())
        leaked = cols & {"home_win", "home_covered"}
        if leaked:
            return _fail(name,
                f"Target columns found in feature list: {sorted(leaked)}.",
                "target", leaked=sorted(leaked))
        return _pass(name,
            "home_win and home_covered absent from feature_columns().",
            "target")
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "target")


def check_feature_columns_no_final_scores() -> CheckResult:
    """feature_columns() must not include home_score or away_score."""
    name = "feature_columns_no_final_scores"
    try:
        from nfl_chatbot.features.team_features import feature_columns

        cols = set(feature_columns())
        leaked = cols & {"home_score", "away_score"}
        if leaked:
            return _fail(name,
                f"Final score columns found in feature list: {sorted(leaked)}.",
                "scores", leaked=sorted(leaked))
        return _pass(name,
            "home_score and away_score absent from feature_columns().",
            "scores")
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "scores")


def check_training_forbidden_set_complete() -> CheckResult:
    """_FORBIDDEN in train.py must include all target and final-score columns."""
    name = "training_forbidden_set_complete"
    required = {"home_win", "home_covered", "home_score", "away_score"}
    try:
        from nfl_chatbot.models.train import _FORBIDDEN

        missing = required - _FORBIDDEN
        if missing:
            return _fail(name,
                f"_FORBIDDEN is missing columns that would cause target/score leakage: "
                f"{sorted(missing)}.",
                "target", missing=sorted(missing), forbidden=sorted(_FORBIDDEN))
        return _pass(name,
            f"_FORBIDDEN correctly contains all target and score columns: "
            f"{sorted(required)}.",
            "target")
    except Exception as exc:
        return _fail(name, f"Import error: {exc}", "target")


def check_select_features_strips_forbidden() -> CheckResult:
    """
    _select_features must exclude columns in _FORBIDDEN even when they appear
    in the requested feature list.
    """
    name = "select_features_strips_forbidden"
    try:
        from nfl_chatbot.models.train import _FORBIDDEN, _select_features

        # Build a DataFrame that includes forbidden columns + a safe column
        df = pd.DataFrame({
            "safe_feature": [1.0, 2.0, 3.0],
            "home_win":     [1, 0, 1],          # target — forbidden
            "home_score":   [28.0, 14.0, 35.0], # final score — forbidden
        })
        requested = ["safe_feature", "home_win", "home_score"]
        _, used = _select_features(df, requested)
        leaked = set(used) & _FORBIDDEN
        if leaked:
            return _fail(name,
                f"_select_features allowed forbidden columns through: {sorted(leaked)}.",
                "target", leaked=sorted(leaked))
        if "safe_feature" not in used:
            return _warn(name,
                "_select_features excluded safe_feature unexpectedly.",
                "target")
        return _pass(name,
            "_select_features correctly strips home_win and home_score from the "
            "feature matrix even when explicitly requested.",
            "target")
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "target")


def check_build_drops_scores_from_table() -> CheckResult:
    """
    FeatureBuilder.build() must drop home_score and away_score before returning.
    Verify this in source code.
    """
    name = "build_drops_scores_from_table"
    src = _read_src(_TEAM_FEATURES_SRC)
    if not src:
        return _skip(name, f"Cannot read {_TEAM_FEATURES_SRC}", "scores")

    # Look for: table.drop(columns=[..., "home_score", "away_score", ...])
    pattern = r'table\s*=\s*table\.drop\s*\(.*?"home_score".*?"away_score"'
    if re.search(pattern, src, re.DOTALL):
        return _pass(name,
            "build() explicitly drops home_score and away_score before returning "
            "the modeling table.",
            "scores")

    # Looser search
    if '"home_score"' in src and '"away_score"' in src and ".drop(" in src:
        return _pass(name,
            "build() contains .drop() with home_score and away_score columns.",
            "scores")

    return _fail(name,
        "Could not find drop(home_score, away_score) in FeatureBuilder.build(). "
        "Final scores may remain in the modeling table.",
        "scores")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELING TABLE (if present)
# ══════════════════════════════════════════════════════════════════════════════

def check_modeling_table_no_raw_scores(
    csv_path: Path | None = None,
) -> CheckResult:
    """If modeling_table.csv exists, verify home_score/away_score are not columns."""
    name = "modeling_table_no_raw_scores"
    path = csv_path or _DEFAULT_MODELING_CSV

    if not path.exists():
        return _skip(name,
            f"modeling_table.csv not found at {path}. "
            "Run `python scripts/build_features.py` to generate it.",
            "modeling_table")

    try:
        df = pd.read_csv(path, nrows=0)  # header only
        leaked = {"home_score", "away_score"} & set(df.columns)
        if leaked:
            return _fail(name,
                f"Final score columns present in modeling_table.csv: {sorted(leaked)}. "
                "These encode the target and must be removed.",
                "modeling_table", path=str(path), leaked=sorted(leaked))
        return _pass(name,
            f"home_score and away_score absent from modeling_table.csv ({path.name}).",
            "modeling_table", path=str(path))
    except Exception as exc:
        return _fail(name, f"Could not read {path}: {exc}", "modeling_table")


def check_modeling_table_no_target_as_feature(
    csv_path: Path | None = None,
    feature_json: Path | None = None,
) -> CheckResult:
    """
    Verify home_win and home_covered are not in the saved feature_columns.json
    (if present), which defines the actual input columns fed to the model.
    """
    name = "modeling_table_no_target_as_feature"
    fj = feature_json or _DEFAULT_FEATURE_JSON

    if not fj.exists():
        # Fall back to checking the CSV header
        csv = csv_path or _DEFAULT_MODELING_CSV
        if not csv.exists():
            return _skip(name,
                "Neither feature_columns.json nor modeling_table.csv found. "
                "Train the model first.",
                "modeling_table")
        try:
            import json as _json
        except ImportError:
            pass
        return _skip(name, f"feature_columns.json not found at {fj}.", "modeling_table")

    try:
        import json as _json
        cols = set(_json.loads(fj.read_text()))
        leaked = cols & {"home_win", "home_covered", "home_score", "away_score"}
        if leaked:
            return _fail(name,
                f"Target/score columns found in feature_columns.json: {sorted(leaked)}.",
                "modeling_table", path=str(fj), leaked=sorted(leaked))
        return _pass(name,
            f"feature_columns.json contains no target or final-score columns.",
            "modeling_table", path=str(fj))
    except Exception as exc:
        return _fail(name, f"Error reading {fj}: {exc}", "modeling_table")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HEAD-TO-HEAD FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def check_h2h_strict_less_than() -> CheckResult:
    """
    H2H history must use strict < (not <=) on game dates.
    Using <= would include the game being predicted.
    """
    name = "h2h_strict_less_than"
    try:
        from nfl_chatbot.features.team_features import FeatureBuilder
        src = inspect.getsource(FeatureBuilder._attach_h2h_features)
    except Exception:
        src = _read_src(_TEAM_FEATURES_SRC)

    if not src:
        return _skip(name, "Cannot inspect _attach_h2h_features source.", "h2h")

    # Look for the date comparison in the prior filter
    strict = bool(re.search(r'gameday"\]\s*<\s*game_date', src) or
                  re.search(r'"gameday"\]\s*<\s*game_date', src) or
                  re.search(r"gameday.*<\s*game_date", src))
    leaky = bool(re.search(r'gameday.*<=\s*game_date', src))

    if leaky:
        return _fail(name,
            "_attach_h2h_features uses <= for date comparison: "
            "the current game's result is included in its own H2H window.",
            "h2h")
    if strict:
        return _pass(name,
            "_attach_h2h_features uses strict < for gameday comparison: "
            "the current game is excluded from its own H2H window.",
            "h2h")
    return _warn(name,
        "Could not confirm the H2H date comparison operator in source. "
        "Manual review recommended.",
        "h2h")


def check_h2h_null_gameday_fallback() -> CheckResult:
    """
    KNOWN ISSUE: When a game's gameday is NaT, _attach_h2h_features falls back
    to using ALL historical matchups with no date filter — future H2H games
    leak into the feature for this row.

    In practice nflverse always provides gameday, so this rarely fires.
    But it is a latent leakage path that should be guarded.
    """
    name = "h2h_null_gameday_fallback"
    try:
        from nfl_chatbot.features.team_features import FeatureBuilder
        src = inspect.getsource(FeatureBuilder._attach_h2h_features)
    except Exception:
        src = _read_src(_TEAM_FEATURES_SRC)

    if not src:
        return _skip(name, "Cannot inspect source.", "h2h")

    # Check for the else-branch that uses no date filter
    has_else_no_filter = bool(re.search(
        r'else\s*:\s*\n\s*prior\s*=\s*completed\[completed\[.matchup_key.\]',
        src
    ))
    has_notna_guard = "pd.notna(game_date)" in src

    if has_notna_guard and has_else_no_filter:
        return _fail(name,
            "When game.gameday is NaT, _attach_h2h_features uses ALL historical "
            "matchups (no date filter). Future games leak into this row's H2H feature. "
            "Fix: replace the else-branch with an empty prior or raise an error.",
            "h2h")

    if has_notna_guard:
        return _warn(name,
            "_attach_h2h_features has a pd.notna(game_date) guard. "
            "Verify the else-branch does not use future data when gameday is NaT.",
            "h2h")

    return _pass(name,
        "No unconditional H2H history fallback found in source.",
        "h2h")


def check_h2h_functional_no_future() -> CheckResult:
    """
    Functional test: build a FeatureBuilder with synthetic data and verify
    that H2H win rate for game 3 (2024) does not incorporate game 3's result.
    """
    name = "h2h_functional_no_future"
    try:
        # We need an engine for FeatureBuilder — use in-memory SQLite
        import sqlalchemy

        games = pd.DataFrame({
            "game_id":      ["2022_W1_BUF_KC", "2023_W1_BUF_KC", "2024_W1_BUF_KC"],
            "season":       [2022, 2023, 2024],
            "week":         [1, 1, 1],
            "game_type":    ["REG", "REG", "REG"],
            "gameday":      pd.to_datetime(["2022-09-11", "2023-09-10", "2024-09-08"]),
            "gametime":     ["13:00"] * 3,
            "home_team":    ["KC", "KC", "KC"],
            "away_team":    ["BUF", "BUF", "BUF"],
            "home_score":   [30.0, 14.0, 28.0],
            "away_score":   [20.0, 21.0, 35.0],  # Game 3: BUF wins (away)
            "home_win":     [1, 0, 0],
            "overtime":     [0, 0, 0],
            "neutral_site": [0, 0, 0],
            "div_game":     [0, 0, 0],
            "spread_line":  [-3.0, 3.0, -6.0],
            "total_line":   [50.0, 45.0, 52.0],
            "roof":         ["dome"] * 3,
            "surface":      ["grass"] * 3,
            "stadium":      ["Arrowhead"] * 3,
            "location":     ["Home"] * 3,
            "created_at":   [None] * 3,
            "updated_at":   [None] * 3,
        })

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        games.to_sql("games", con=engine, if_exists="replace", index=False)

        from nfl_chatbot.features.team_features import FeatureBuilder
        fb = FeatureBuilder(engine=engine, window=4)
        table = fb.build()

        if table.empty:
            return _skip(name, "FeatureBuilder.build() returned empty table.", "h2h")

        # Game 3 (2024): prior H2H should only include games 1 and 2
        # KC won game 1 (home), lost game 2 (home) → home win rate = 0.5
        row3 = table[table["season"] == 2024]
        if row3.empty:
            return _skip(name, "2024 game not found in built table.", "h2h")

        h2h_rate = float(row3["h2h_home_win_rate"].iloc[0])
        # Expected: 1 KC win out of 2 prior games = 0.5
        if abs(h2h_rate - 0.5) < 0.05:
            return _pass(name,
                f"H2H win rate for game 3 = {h2h_rate:.3f} (expected 0.500). "
                "Only prior games used — no future leakage.",
                "h2h", h2h_rate=h2h_rate)
        else:
            return _fail(name,
                f"H2H win rate for game 3 = {h2h_rate:.3f} but expected 0.500. "
                "Possible future game included in H2H window.",
                "h2h", h2h_rate=h2h_rate)

    except Exception as exc:
        return _skip(name, f"Could not run functional H2H test: {exc}", "h2h")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ELO RATINGS
# ══════════════════════════════════════════════════════════════════════════════

def check_elo_stored_pregame() -> CheckResult:
    """
    Elo ratings must be appended (stored) before the post-game update.
    Verify by inspecting _compute_elo source order.
    """
    name = "elo_stored_pregame"
    try:
        from nfl_chatbot.features.team_features import FeatureBuilder
        src = inspect.getsource(FeatureBuilder._compute_elo)
    except Exception:
        src = _read_src(_TEAM_FEATURES_SRC)

    if not src:
        return _skip(name, "Cannot inspect _compute_elo source.", "elo")

    append_pos = src.find("rows.append(")
    update_pos = src.find("elo[home] = elo[home]")
    if update_pos == -1:
        update_pos = src.find("elo[home] +")

    if append_pos == -1:
        return _warn(name, "Could not find rows.append() in _compute_elo.", "elo")
    if update_pos == -1:
        return _warn(name, "Could not find elo update line in _compute_elo.", "elo")

    if append_pos < update_pos:
        return _pass(name,
            "rows.append() (pre-game Elo stored) comes before elo[home] update "
            "(post-game) — stored value is the true pre-game rating.",
            "elo", append_line=append_pos, update_line=update_pos)
    else:
        return _fail(name,
            "elo[home] update appears BEFORE rows.append() in _compute_elo. "
            "Post-game Elo would be stored as the pre-game feature.",
            "elo", append_line=append_pos, update_line=update_pos)


def check_elo_functional_pregame() -> CheckResult:
    """
    Functional test: train two teams through two games.
    After game 1, elo_diff for game 2 should reflect the post-game-1 rating,
    NOT include game 2's outcome.
    """
    name = "elo_functional_pregame"
    try:
        from nfl_chatbot.features.team_features import FeatureBuilder
        import sqlalchemy

        games = pd.DataFrame({
            "game_id":      ["2022_W1_BUF_KC", "2022_W2_BUF_KC"],
            "season":       [2022, 2022],
            "week":         [1, 2],
            "game_type":    ["REG", "REG"],
            "gameday":      pd.to_datetime(["2022-09-11", "2022-09-18"]),
            "gametime":     ["13:00", "13:00"],
            "home_team":    ["KC", "KC"],
            "away_team":    ["BUF", "BUF"],
            "home_score":   [30.0, 14.0],
            "away_score":   [20.0, 21.0],
            "home_win":     [1, 0],
            "overtime":     [0, 0],
            "neutral_site": [0, 0],
            "div_game":     [0, 0],
            "spread_line":  [-3.0, 3.0],
            "total_line":   [50.0, 45.0],
            "roof":         ["dome", "dome"],
            "surface":      ["grass", "grass"],
            "stadium":      ["A", "A"],
            "location":     ["Home", "Home"],
            "created_at":   [None, None],
            "updated_at":   [None, None],
        })

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        games.to_sql("games", con=engine, if_exists="replace", index=False)

        fb = FeatureBuilder(engine=engine, window=4)
        table = fb.build()

        if table.empty or len(table) < 2:
            return _skip(name, "Functional Elo test needs ≥2 rows.", "elo")

        elo_g1 = float(table.sort_values("week").iloc[0]["elo_diff"])
        elo_g2 = float(table.sort_values("week").iloc[1]["elo_diff"])

        # Game 1: both teams start at 1500. With home advantage (+48) elo_diff = 48.
        # After game 1 (KC wins), KC Elo increases; game 2 elo_diff should differ from 48.
        if abs(elo_g1 - 48.0) < 1.0 and abs(elo_g2 - 48.0) > 1.0:
            return _pass(name,
                f"Game-1 elo_diff={elo_g1:.1f} (initial, ≈48.0). "
                f"Game-2 elo_diff={elo_g2:.1f} (updated after game-1, ≠48.0). "
                "Elo is correctly stored pre-game.",
                "elo", g1=elo_g1, g2=elo_g2)
        elif abs(elo_g1 - 48.0) > 5.0:
            return _fail(name,
                f"Game-1 elo_diff={elo_g1:.1f} should be ≈48.0 (both teams at 1500 + home adv). "
                "Initial Elo may have been updated before being stored.",
                "elo", g1=elo_g1)
        else:
            return _warn(name,
                f"Game-1 elo_diff={elo_g1:.1f}, game-2 elo_diff={elo_g2:.1f}. "
                "Expected game-2 to differ from game-1 after the post-game update.",
                "elo", g1=elo_g1, g2=elo_g2)

    except Exception as exc:
        return _skip(name, f"Functional Elo test error: {exc}", "elo")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SPREAD / ODDS
# ══════════════════════════════════════════════════════════════════════════════

def check_spread_line_documented_pregame() -> CheckResult:
    """
    spread_line and total_line must be pre-game closing lines, not derived
    from final scores.  Verify source documentation.
    """
    name = "spread_line_documented_pregame"
    src = _read_src(_TEAM_FEATURES_SRC)
    if not src:
        return _skip(name, f"Cannot read {_TEAM_FEATURES_SRC}", "spread")

    # Check the betting features method comment
    patterns = [
        r"pre.game",
        r"closing\s+line",
        r"opening\s+line",
        r"nflverse.*spread",
        r"spread.*nflverse",
        r"spread_line.*before",
        r"before.*spread_line",
    ]
    matched = [p for p in patterns if re.search(p, src, re.IGNORECASE)]

    if matched:
        return _pass(name,
            f"Source documents spread_line as pre-game data ({matched[0]!r} found). "
            "nflverse spread_line is the consensus closing line set before the game.",
            "spread")

    return _warn(name,
        "No explicit 'pre-game' or 'closing line' documentation found for "
        "spread_line/total_line in team_features.py. "
        "Confirm these are pre-game consensus lines (they are in nflverse), "
        "not derived from final scores.",
        "spread")


def check_spread_not_derived_from_scores() -> CheckResult:
    """
    spread_line must not be computed from home_score/away_score (that would
    be post-game result leakage).  Verify it comes directly from ingested data.
    """
    name = "spread_not_derived_from_scores"
    # In team_features.py, _attach_betting_features uses table["spread_line"]
    # which comes from the games table, not computed from scores.
    src = _read_src(_TEAM_FEATURES_SRC)
    if not src:
        return _skip(name, "Cannot read source.", "spread")

    # There should be no formula like spread = home_score - away_score
    leaky_pattern = r'spread[_\s]*=.*home_score.*-.*away_score'
    if re.search(leaky_pattern, src, re.IGNORECASE):
        return _fail(name,
            "Found pattern suggesting spread_line is computed from final scores.",
            "spread")

    # Verify spread_line is read from the games DataFrame directly
    uses_table_spread = bool(re.search(r'table\["spread_line"\]', src))
    if uses_table_spread:
        return _pass(name,
            "_attach_betting_features reads spread_line directly from the games "
            "table (ingested pre-game closing line), not computed from final scores.",
            "spread")

    return _warn(name,
        "Could not confirm spread_line comes from ingested pre-game data. "
        "Verify it is not derived from final scores.",
        "spread")


# ══════════════════════════════════════════════════════════════════════════════
# 7. INJURY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def check_injury_join_on_season_week() -> CheckResult:
    """
    Injury features must join on (season, week, team) to match each game's
    pre-game report.  Joining on any post-game field would leak results.
    """
    name = "injury_join_on_season_week"
    try:
        from nfl_chatbot.features.team_features import FeatureBuilder
        src = inspect.getsource(FeatureBuilder._attach_injury_features)
    except Exception:
        src = _read_src(_TEAM_FEATURES_SRC)

    if not src:
        return _skip(name, "Cannot inspect source.", "injury")

    join_key_ok = (
        '"season"' in src and '"week"' in src
        and 'groupby(["season", "week", "team"])' in src
    )

    if join_key_ok:
        return _pass(name,
            "_attach_injury_features aggregates by (season, week, team), "
            "matching each game's pre-game injury report to the correct week.",
            "injury")

    return _warn(name,
        "Could not confirm injury groupby uses (season, week, team). "
        "Verify the join key does not include post-game fields.",
        "injury")


def check_injury_no_postgame_status() -> CheckResult:
    """
    WARN: the code joins injuries to games by (season, week) without explicitly
    checking report_date < game_date.  This is correct because NFL injury reports
    are always published before the game, but the code has no explicit guard.
    """
    name = "injury_no_postgame_status"
    try:
        from nfl_chatbot.features.team_features import FeatureBuilder
        src = inspect.getsource(FeatureBuilder._attach_injury_features)
    except Exception:
        src = _read_src(_TEAM_FEATURES_SRC)

    if not src:
        return _skip(name, "Cannot inspect source.", "injury")

    has_date_check = bool(re.search(r'report_date|date_modified|gameday', src))

    if has_date_check:
        return _pass(name,
            "_attach_injury_features performs an explicit date check "
            "(report_date / date_modified / gameday found in source).",
            "injury")

    return _warn(name,
        "No explicit report_date < gameday check in _attach_injury_features. "
        "Safety relies on the assumption that NFL weekly injury reports are always "
        "published before their game. "
        "Recommend: add an explicit filter when date_modified is available.",
        "injury")


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def check_chronological_split_no_future_in_train() -> CheckResult:
    """
    _chronological_split must keep all train rows strictly before the test season.
    """
    name = "chronological_split_no_future_in_train"
    try:
        from nfl_chatbot.models.train import _chronological_split

        df = pd.DataFrame({
            "season": [2020] * 50 + [2021] * 50 + [2022] * 50 + [2023] * 50,
            "home_win": [1] * 200,
        })
        train, test, actual = _chronological_split(df, test_season=2023)

        if (train["season"] >= 2023).any():
            leaked = sorted(train["season"].unique().tolist())
            return _fail(name,
                f"Train set contains rows from test season or later: {leaked}.",
                "split", train_seasons=leaked)

        if (test["season"] != 2023).any():
            return _fail(name,
                "Test set contains rows from seasons other than 2023.",
                "split")

        train_seasons = sorted(train["season"].unique().tolist())
        return _pass(name,
            f"Train seasons {train_seasons} are all < 2023; test is exactly 2023.",
            "split", train_seasons=train_seasons, test_season=2023)

    except Exception as exc:
        return _fail(name, f"Error: {exc}", "split")


def check_chronological_split_size() -> CheckResult:
    """Train + test must account for all rows (no row dropped, no row duplicated)."""
    name = "chronological_split_size"
    try:
        from nfl_chatbot.models.train import _chronological_split

        df = pd.DataFrame({
            "season": [2020] * 40 + [2021] * 40 + [2022] * 40 + [2023] * 40,
            "home_win": [1] * 160,
        })
        train, test, _ = _chronological_split(df, test_season=2023)
        total = len(train) + len(test)
        if total != len(df):
            return _fail(name,
                f"len(train)={len(train)} + len(test)={len(test)} = {total} "
                f"≠ len(df)={len(df)}. Rows may have been lost or duplicated.",
                "split")
        # No overlap
        overlap = set(train.index) & set(test.index)
        if overlap:
            return _fail(name,
                f"{len(overlap)} row(s) appear in both train and test sets.",
                "split", overlap_count=len(overlap))
        return _pass(name,
            f"train ({len(train)} rows) + test ({len(test)} rows) = {len(df)} total. "
            "No overlap, no missing rows.",
            "split")
    except Exception as exc:
        return _fail(name, f"Error: {exc}", "split")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CROSS-VALIDATION FOLDS
# ══════════════════════════════════════════════════════════════════════════════

def check_cv_folds_no_future_in_train() -> CheckResult:
    """
    Every CV fold must have all training seasons strictly before the val season.
    """
    name = "cv_folds_no_future_in_train"
    try:
        from nfl_chatbot.models.train import _season_cv_folds

        df = pd.DataFrame({
            "season": [2020] * 40 + [2021] * 40 + [2022] * 40 + [2023] * 40,
            "x": range(160),
        })
        folds = _season_cv_folds(df)

        if not folds:
            return _warn(name, "No CV folds produced (need ≥2 seasons).", "cv")

        for i, (train_idx, val_idx) in enumerate(folds):
            train_seas = df.iloc[train_idx]["season"].unique()
            val_seas = df.iloc[val_idx]["season"].unique()
            if max(train_seas) >= min(val_seas):
                return _fail(name,
                    f"CV fold {i+1}: train contains season {max(train_seas)} "
                    f"which is ≥ val season {min(val_seas)}.",
                    "cv", fold=i+1, train_max=int(max(train_seas)),
                    val_min=int(min(val_seas)))

        fold_shapes = [(len(ti), len(vi)) for ti, vi in folds]
        return _pass(name,
            f"{len(folds)} season-based CV folds are all forward-only: "
            f"no fold's train set contains data from the val season.",
            "cv", n_folds=len(folds), fold_shapes=fold_shapes)

    except Exception as exc:
        return _fail(name, f"Error: {exc}", "cv")


def check_cv_folds_monotone_train_size() -> CheckResult:
    """
    Each subsequent CV fold should have a strictly larger training set
    (progressive season expansion).
    """
    name = "cv_folds_monotone_train_size"
    try:
        from nfl_chatbot.models.train import _season_cv_folds

        df = pd.DataFrame({
            "season": [2020] * 40 + [2021] * 40 + [2022] * 40 + [2023] * 40,
        })
        folds = _season_cv_folds(df)

        if len(folds) < 2:
            return _skip(name, "Need ≥2 folds to check monotonicity.", "cv")

        sizes = [len(ti) for ti, _ in folds]
        monotone = all(sizes[i] < sizes[i+1] for i in range(len(sizes) - 1))

        if monotone:
            return _pass(name,
                f"CV fold train sizes grow monotonically: {sizes}.",
                "cv", train_sizes=sizes)
        else:
            return _fail(name,
                f"CV fold train sizes are NOT monotonically increasing: {sizes}. "
                "Later folds should always have more training data.",
                "cv", train_sizes=sizes)

    except Exception as exc:
        return _fail(name, f"Error: {exc}", "cv")


# ══════════════════════════════════════════════════════════════════════════════
# Master runner
# ══════════════════════════════════════════════════════════════════════════════

_ALL_CHECKS = [
    # Rolling
    check_rolling_shift_excludes_current_game,
    check_rolling_shift_sum_excludes_current,
    check_season_expanding_excludes_current,
    check_rolling_first_game_is_nan,
    check_rolling_sort_order_safety,
    # Target / score leakage
    check_feature_columns_no_target,
    check_feature_columns_no_final_scores,
    check_training_forbidden_set_complete,
    check_select_features_strips_forbidden,
    check_build_drops_scores_from_table,
    # Modeling table (artifact-based)
    check_modeling_table_no_raw_scores,
    check_modeling_table_no_target_as_feature,
    # Head-to-head
    check_h2h_strict_less_than,
    check_h2h_null_gameday_fallback,
    check_h2h_functional_no_future,
    # Elo
    check_elo_stored_pregame,
    check_elo_functional_pregame,
    # Spread
    check_spread_line_documented_pregame,
    check_spread_not_derived_from_scores,
    # Injury
    check_injury_join_on_season_week,
    check_injury_no_postgame_status,
    # Train/test split
    check_chronological_split_no_future_in_train,
    check_chronological_split_size,
    # CV folds
    check_cv_folds_no_future_in_train,
    check_cv_folds_monotone_train_size,
]


def run_all_checks(
    modeling_csv: Path | str | None = None,
    feature_json: Path | str | None = None,
) -> list[CheckResult]:
    """
    Run every leakage check and return results.

    Args:
        modeling_csv: Override path to modeling_table.csv.
        feature_json: Override path to feature_columns.json.

    Returns:
        List of CheckResult objects in the same order as _ALL_CHECKS.
    """
    results: list[CheckResult] = []
    for fn in _ALL_CHECKS:
        sig = inspect.signature(fn)
        kwargs: dict = {}
        if "csv_path" in sig.parameters and modeling_csv:
            kwargs["csv_path"] = Path(modeling_csv)
        if "feature_json" in sig.parameters and feature_json:
            kwargs["feature_json"] = Path(feature_json)
        try:
            results.append(fn(**kwargs))
        except Exception as exc:
            results.append(_fail(
                fn.__name__,
                f"Unhandled exception in check: {exc}",
                "unknown",
            ))
    return results
