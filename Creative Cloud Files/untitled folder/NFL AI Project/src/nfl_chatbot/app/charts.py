"""
Plotly chart utilities for the NFL AI Chatbot Streamlit app.

Every function returns a ``plotly.graph_objects.Figure`` so callers can
render it with ``st.plotly_chart(fig, use_container_width=True)`` or
further customise it before display.

All charts share a consistent palette and minimal layout so they look
coherent whether Streamlit is in light or dark mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette ───────────────────────────────────────────────────────────────────

_NAVY    = "#1e3a5f"
_GOLD    = "#e8a020"
_BLUE    = "#3b82f6"
_GREEN   = "#22c55e"
_RED     = "#ef4444"
_AMBER   = "#f59e0b"
_SLATE   = "#94a3b8"
_BG      = "rgba(0,0,0,0)"          # transparent — adapts to light/dark theme
_GRID    = "rgba(128,128,128,0.12)"
_TEXT    = "rgba(200,200,200,0.9)"  # visible in both themes

_TEAM_A_COLOR = _BLUE
_TEAM_B_COLOR = _GOLD

# Shared layout defaults
_BASE_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="rgba(30,30,30,0.9)", font_color="white"),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=_TEXT)), **_BASE_LAYOUT)
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 1. Feature importance
# ══════════════════════════════════════════════════════════════════════════════

def feature_importance_chart(
    df: pd.DataFrame,
    title: str = "Feature Importance (Top 20)",
    max_features: int = 20,
    importance_col: str = "avg_importance",
    feature_col: str = "feature",
) -> go.Figure:
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    df : DataFrame with at least *feature_col* and *importance_col* columns.
    """
    if df is None or df.empty:
        return _empty_chart("No feature importance data available.")

    top = (
        df[[feature_col, importance_col]]
        .nlargest(max_features, importance_col)
        .sort_values(importance_col)
        .copy()
    )
    top[feature_col] = top[feature_col].str.replace("_", " ").str.title()

    # Colour gradient: most important = navy, least = lighter blue
    n = len(top)
    colours = [
        f"rgba(30,58,95,{0.4 + 0.6 * (i / max(n - 1, 1))})" for i in range(n)
    ]

    fig = go.Figure(go.Bar(
        x=top[importance_col],
        y=top[feature_col],
        orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Mean importance",
        yaxis_title=None,
        height=max(300, n * 22 + 80),
    )
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model accuracy by season
# ══════════════════════════════════════════════════════════════════════════════

def accuracy_by_season_chart(
    df: pd.DataFrame,
    title: str = "Model Accuracy by Season",
    season_col: str = "season",
    accuracy_col: str = "accuracy",
    baseline_col: str | None = "home_win_rate",
) -> go.Figure:
    """
    Bar chart with an optional baseline line showing per-season accuracy.

    *df* should have at least a *season_col* and *accuracy_col*.
    If *baseline_col* exists it is drawn as a reference line.
    """
    if df is None or df.empty:
        return _empty_chart("No per-season accuracy data available.")

    df = df.sort_values(season_col)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[season_col].astype(str),
        y=df[accuracy_col],
        marker_color=_BLUE,
        name="Model accuracy",
        hovertemplate="Season %{x}<br>Accuracy: %{y:.1%}<extra></extra>",
    ))

    if baseline_col and baseline_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[season_col].astype(str),
            y=df[baseline_col],
            mode="lines+markers",
            line=dict(color=_AMBER, dash="dash", width=2),
            marker=dict(size=6),
            name="Home-win baseline",
            hovertemplate="Season %{x}<br>Baseline: %{y:.1%}<extra></extra>",
        ))

    fig.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        xaxis_title="Season",
        yaxis_title="Accuracy",
        legend=dict(orientation="h", y=1.08),
        height=350,
        barmode="group",
    )
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def confusion_matrix_chart(
    matrix: list[list[int]] | np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """
    Annotated heatmap for a 2×2 or N×N confusion matrix.

    Parameters
    ----------
    matrix : 2D array-like, rows = actual, cols = predicted.
    labels : class labels; defaults to ["Away Win", "Home Win"].
    """
    if labels is None:
        labels = ["Away Win", "Home Win"]

    cm = np.array(matrix)
    if cm.size == 0:
        return _empty_chart("No confusion matrix data.")

    # Normalise to percentage of actual (row-normalised)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    text = [
        [f"<b>{int(cm[r, c])}</b><br>({cm_norm[r, c]:.1%})" for c in range(cm.shape[1])]
        for r in range(cm.shape[0])
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Predicted:<br>{l}" for l in labels],
        y=[f"Actual:<br>{l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#1e2533"], [0.5, "#1e4fa5"], [1, _BLUE]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
    ))
    fig.update_layout(height=300)
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Calibration curve
# ══════════════════════════════════════════════════════════════════════════════

def calibration_curve_chart(
    y_prob: list[float] | np.ndarray,
    y_true: list[int] | np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    model_label: str = "Model",
) -> go.Figure:
    """
    Reliability diagram: predicted probability vs actual win rate per bin.

    A perfectly calibrated model falls on the diagonal.
    Points above the diagonal mean the model underestimates probability;
    below means it overestimates.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if len(y_prob) == 0:
        return _empty_chart("No calibration data available.")

    bins = np.linspace(0, 1, n_bins + 1)
    bin_mid, frac_pos, counts = [], [], []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_mid.append((bins[i] + bins[i + 1]) / 2)
            frac_pos.append(y_true[mask].mean())
            counts.append(int(mask.sum()))

    fig = go.Figure()

    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=_SLATE, dash="dash", width=1.5),
        name="Perfect calibration",
        hoverinfo="skip",
    ))

    # Model calibration
    fig.add_trace(go.Scatter(
        x=bin_mid,
        y=frac_pos,
        mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(
            size=[max(6, min(20, c / 5)) for c in counts],
            color=_BLUE,
            line=dict(color="white", width=1.5),
        ),
        name=model_label,
        hovertemplate=(
            "Predicted: %{x:.0%}<br>"
            "Actual: %{y:.0%}<br>"
            "Count: %{customdata}<extra></extra>"
        ),
        customdata=counts,
    ))

    fig.update_layout(
        xaxis=dict(title="Mean predicted probability", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title="Fraction of positives", tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", y=1.08),
        height=350,
    )
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Team cover rate
# ══════════════════════════════════════════════════════════════════════════════

def team_cover_rate_chart(
    df: pd.DataFrame,
    title: str = "ATS Cover Rate by Team",
    team_col: str = "team",
    rate_col: str = "cover_rate",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    n_teams: int = 32,
) -> go.Figure:
    """
    Horizontal bar chart of team cover rates with optional 95% CI error bars.

    Bars above 50% are coloured green; below are red.
    """
    if df is None or df.empty:
        return _empty_chart("No team cover rate data available.")

    df = df.copy().sort_values(rate_col, ascending=True).head(n_teams)
    rates = df[rate_col].values
    colours = [_GREEN if r >= 0.50 else _RED for r in rates]

    error_y: dict[str, Any] = {}
    if ci_low_col and ci_high_col and ci_low_col in df.columns and ci_high_col in df.columns:
        error_y = dict(
            type="data",
            symmetric=False,
            array=(df[ci_high_col] - df[rate_col]).tolist(),
            arrayminus=(df[rate_col] - df[ci_low_col]).tolist(),
            color=_SLATE,
            thickness=1.5,
            width=4,
        )

    fig = go.Figure(go.Bar(
        x=rates,
        y=df[team_col],
        orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        error_x=error_y if error_y else None,
        hovertemplate="%{y}  Cover rate: %{x:.1%}<extra></extra>",
    ))

    # 50% reference line
    fig.add_vline(x=0.5, line=dict(color=_SLATE, dash="dash", width=1.5))
    fig.add_annotation(
        x=0.5, y=1.02, xref="x", yref="paper",
        text="50%", showarrow=False,
        font=dict(size=10, color=_SLATE),
    )

    fig.update_layout(
        xaxis=dict(title="Cover rate", tickformat=".0%", range=[0.2, 0.8]),
        yaxis_title=None,
        height=max(350, len(df) * 20 + 80),
    )
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Player comparison — grouped bar + radar toggle
# ══════════════════════════════════════════════════════════════════════════════

def player_comparison_chart(
    stats_a: dict[str, float],
    stats_b: dict[str, float],
    name_a: str = "Player A",
    name_b: str = "Player B",
    chart_type: str = "bar",
    title: str | None = None,
    max_metrics: int = 10,
) -> go.Figure:
    """
    Side-by-side grouped bar chart (default) or radar chart for two players.

    Parameters
    ----------
    stats_a / stats_b : dict mapping metric name → float value.
    chart_type : "bar" or "radar".
    """
    if not stats_a and not stats_b:
        return _empty_chart("No player stats available.")

    # Shared numeric keys, limited to max_metrics
    keys = [
        k for k in stats_a.keys()
        if k in stats_b and isinstance(stats_a.get(k), (int, float))
        and stats_a.get(k) is not None and stats_b.get(k) is not None
    ][:max_metrics]

    if not keys:
        return _empty_chart("No comparable numeric stats found.")

    labels = [k.replace("_", " ").title() for k in keys]
    vals_a = [float(stats_a[k]) for k in keys]
    vals_b = [float(stats_b[k]) for k in keys]
    display_title = title or f"{name_a} vs {name_b}"

    if chart_type == "radar":
        return _player_radar(labels, vals_a, vals_b, name_a, name_b, display_title)

    fig = go.Figure()
    x = list(range(len(labels)))
    width = 0.35

    fig.add_trace(go.Bar(
        x=[xi - width / 2 for xi in x],
        y=vals_a,
        name=name_a,
        width=width,
        marker_color=_TEAM_A_COLOR,
        hovertemplate=f"{name_a}: %{{y:.1f}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[xi + width / 2 for xi in x],
        y=vals_b,
        name=name_b,
        width=width,
        marker_color=_TEAM_B_COLOR,
        hovertemplate=f"{name_b}: %{{y:.1f}}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(tickvals=x, ticktext=labels, tickangle=-30),
        yaxis_title="Value",
        legend=dict(orientation="h", y=1.08),
        barmode="group",
        height=380,
    )
    return _apply_base(fig, display_title)


def _player_radar(
    labels: list[str],
    vals_a: list[float],
    vals_b: list[float],
    name_a: str,
    name_b: str,
    title: str,
) -> go.Figure:
    """Radar chart variant for player_comparison_chart."""
    # Normalise to 0–100 scale per metric so different units are comparable
    combined = [max(a, b) for a, b in zip(vals_a, vals_b)]
    norm_a = [100 * (a / m) if m > 0 else 0 for a, m in zip(vals_a, combined)]
    norm_b = [100 * (b / m) if m > 0 else 0 for b, m in zip(vals_b, combined)]

    # Close the polygon
    cats = labels + [labels[0]]
    norm_a = norm_a + [norm_a[0]]
    norm_b = norm_b + [norm_b[0]]

    fig = go.Figure()
    fill_colors = ["rgba(59,130,246,0.15)", "rgba(232,160,32,0.15)"]
    for (vals, name, color), fill in zip(
        [(norm_a, name_a, _TEAM_A_COLOR), (norm_b, name_b, _TEAM_B_COLOR)],
        fill_colors,
    ):
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=name,
            line=dict(color=color, width=2),
            fillcolor=fill,
            hovertemplate=f"{name}: %{{r:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 110])),
        legend=dict(orientation="h", y=1.08),
        height=400,
    )
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Win probability display
# ══════════════════════════════════════════════════════════════════════════════

def win_probability_chart(
    home_prob: float,
    away_prob: float,
    home_team: str = "Home",
    away_team: str = "Away",
    title: str = "Win Probability",
) -> go.Figure:
    """
    Split horizontal bar showing home vs away win probability.

    Renders as a single stacked bar — visually intuitive and clean.
    """
    home_prob = max(0.0, min(1.0, home_prob))
    away_prob = 1.0 - home_prob

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[home_prob],
        y=[""],
        orientation="h",
        name=home_team,
        marker=dict(color=_BLUE, line=dict(width=0)),
        text=[f"<b>{home_team}</b><br>{home_prob:.0%}"],
        textposition="inside" if home_prob > 0.25 else "outside",
        insidetextanchor="middle",
        hovertemplate=f"{home_team} win probability: {home_prob:.1%}<extra></extra>",
        width=0.5,
    ))
    fig.add_trace(go.Bar(
        x=[away_prob],
        y=[""],
        orientation="h",
        name=away_team,
        marker=dict(color=_GOLD, line=dict(width=0)),
        text=[f"<b>{away_team}</b><br>{away_prob:.0%}"],
        textposition="inside" if away_prob > 0.25 else "outside",
        insidetextanchor="middle",
        hovertemplate=f"{away_team} win probability: {away_prob:.1%}<extra></extra>",
        width=0.5,
    ))

    # Midpoint marker
    fig.add_vline(x=0.5, line=dict(color=_SLATE, dash="dot", width=1))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=True,
        legend=dict(orientation="h", y=1.3, x=0.5, xanchor="center"),
        height=140,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return _apply_base(fig, title)


# ══════════════════════════════════════════════════════════════════════════════
# Loader helpers (used by Streamlit pages)
# ══════════════════════════════════════════════════════════════════════════════

def load_feature_importance(model_dir: str, kind: str = "win") -> pd.DataFrame | None:
    """
    Load feature importance CSV from *model_dir*.

    kind : "win" → feature_importance.csv
           "spread" → spread_feature_importance.csv
    """
    from pathlib import Path

    fname = "feature_importance.csv" if kind == "win" else "spread_feature_importance.csv"
    path = Path(model_dir) / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "feature" not in df.columns:
            return None
        imp_cols = [c for c in df.columns if c != "feature"]
        if not imp_cols:
            return None
        df["avg_importance"] = df[imp_cols].mean(axis=1)
        return df
    except Exception:
        return None


def load_confusion_matrix(metrics_json: dict, model_name: str | None = None) -> list | None:
    """Extract the confusion matrix array from a model_metrics.json dict."""
    best = model_name or metrics_json.get("best_model", "")
    model_data = metrics_json.get("models", {}).get(best, {})
    cm = model_data.get("confusion_matrix")
    if cm and len(cm) >= 2:
        return cm
    return None


def compute_calibration_data(
    model_dir: str,
    modeling_csv: str,
    test_season: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the trained model and modeling table, predict on the test season,
    and return (y_prob, y_true) arrays for the calibration curve.

    Returns None if any required artifact is missing.
    """
    from pathlib import Path

    model_path = Path(model_dir) / "best_model.joblib"
    feat_path = Path(model_dir) / "feature_columns.json"
    metrics_path = Path(model_dir) / "model_metrics.json"
    csv_path = Path(modeling_csv)

    if not all(p.exists() for p in [model_path, feat_path, metrics_path, csv_path]):
        return None

    try:
        import json, joblib
        from pathlib import Path as _P

        pipeline = joblib.load(model_path)
        feature_cols = json.loads(feat_path.read_text())
        metrics = json.loads(metrics_path.read_text())

        if test_season is None:
            test_season = metrics.get("split", {}).get("test_season")

        df = pd.read_csv(csv_path)
        if test_season:
            test_df = df[df["season"] == test_season].copy()
        else:
            test_df = df.copy()

        if test_df.empty or "home_win" not in test_df.columns:
            return None

        X = test_df[feature_cols].fillna(0)
        y = test_df["home_win"].values
        y_prob = pipeline.predict_proba(X)[:, 1]
        return y_prob, y

    except Exception:
        return None


# ── Utility ───────────────────────────────────────────────────────────────────

def _empty_chart(message: str) -> go.Figure:
    """Return a blank figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=_SLATE),
    )
    fig.update_layout(height=200, **_BASE_LAYOUT)
    return fig
