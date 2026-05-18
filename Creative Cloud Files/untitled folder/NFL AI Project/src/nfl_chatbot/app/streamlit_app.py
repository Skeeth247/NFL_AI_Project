"""
NFL AI Chatbot — Streamlit frontend.

Run:
    streamlit run src/nfl_chatbot/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

# Ensure src/ is on the path when run directly via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from nfl_chatbot.app.charts import (
    feature_importance_chart,
    accuracy_by_season_chart,
    confusion_matrix_chart,
    calibration_curve_chart,
    team_cover_rate_chart,
    player_comparison_chart,
    win_probability_chart,
    load_feature_importance,
    load_confusion_matrix,
    compute_calibration_data,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="NFL AI Chatbot",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "NFL AI Chatbot — built with Python, FastAPI, and Streamlit."},
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Tighten overall padding */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Metric card borders */
[data-testid="metric-container"] {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

/* Section divider */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* Intent badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-blue   { background: #1d4ed820; color: #3b82f6; }
.badge-green  { background: #16a34a20; color: #22c55e; }
.badge-amber  { background: #d9770620; color: #f59e0b; }
.badge-gray   { background: #37415120; color: #9ca3af; }

/* Probability bar */
.prob-bar-wrap { background: rgba(128,128,128,0.12); border-radius: 6px; overflow: hidden; height: 10px; }
.prob-bar-fill { height: 10px; border-radius: 6px; }

/* Chat message meta */
.chat-meta { font-size: 0.75rem; color: #888; margin-top: 0.25rem; }

/* Factor bullet */
.factor-row {
    padding: 0.4rem 0.6rem;
    border-left: 3px solid;
    border-radius: 0 4px 4px 0;
    margin-bottom: 0.3rem;
    font-size: 0.88rem;
}
.factor-pos { border-color: #22c55e; background: #16a34a10; }
.factor-neg { border-color: #ef4444; background: #dc262610; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

NFL_TEAMS: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",  "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals","CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",     "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars","KC": "Kansas City Chiefs",
    "LAC": "LA Chargers",       "LAR": "LA Rams",
    "LV":  "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE":  "New England Patriots",
    "NO":  "New Orleans Saints","NYG": "New York Giants",
    "NYJ": "New York Jets",     "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers","SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers","TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",  "WAS": "Washington Commanders",
}

TEAM_OPTIONS = [f"{name} ({abbr})" for abbr, name in sorted(NFL_TEAMS.items(), key=lambda x: x[1])]
TEAM_NAME_TO_ABBR = {f"{name} ({abbr})": abbr for abbr, name in NFL_TEAMS.items()}

TREND_QUESTIONS = [
    "How often do favorites cover the spread?",
    "How often do underdogs cover the spread?",
    "How often do home favorites cover?",
    "How often do road underdogs cover?",
    "What teams go over the total most often?",
    "What happens when the spread is greater than 7 points?",
    "Does the model outperform a random baseline?",
]

# ── Demo mode ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading demo data…")
def _get_demo_mode() -> bool:
    from nfl_chatbot.demo.loader import is_demo_mode
    return is_demo_mode()


def _demo_banner() -> None:
    st.warning(
        "**Demo mode** — using sample data for 8 teams (2022–2023). "
        "Run `nfl-chatbot ingest` then `nfl-chatbot train` for real model results.",
        icon="🏈",
    )


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models and data…")
def _load_tools():
    if _get_demo_mode():
        from nfl_chatbot.demo.loader import load_demo_env
        return load_demo_env().tools
    from nfl_chatbot.chatbot.tools import ChatbotTools
    return ChatbotTools()

@st.cache_resource(show_spinner=False)
def _load_responder():
    from nfl_chatbot.chatbot.responder import Responder
    return Responder(tools=_load_tools())

# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'

def _confidence_color(conf: float) -> str:
    if conf >= 0.70:
        return "green"
    if conf >= 0.50:
        return "amber"
    return "gray"

def _prob_bar(prob: float, color: str = "#3b82f6") -> str:
    pct = int(prob * 100)
    return (
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )

def _section(title: str) -> None:
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)

def _data_warning(msg: str) -> None:
    st.warning(msg, icon="⚠️")

def _no_model_notice() -> None:
    st.info(
        "No trained model found. Run `python scripts/train_model.py` to enable predictions.",
        icon="ℹ️",
    )

def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"

def _round(v, d=1) -> str:
    if v is None:
        return "—"
    return str(round(v, d))

# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🏈 NFL AI Chatbot")
        st.markdown(
            "<span style='font-size:0.82rem;color:#888;'>"
            "Predictive analytics powered by machine learning."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Chatbot",
                "Matchup Predictor",
                "Player Comparison",
                "Betting Trends",
                "Model Performance",
                "About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Data status indicator
        if _get_demo_mode():
            st.info("Demo mode · 56 games · 8 teams", icon="🏈")
        else:
            try:
                tools = _load_tools()
                summary = tools.get_available_data_summary()
                db = summary.get("database", {})
                games = db.get("games", {})
                count = games.get("count", 0)
                models = summary.get("model_artifacts", {})
                win_ok = models.get("win_model", {}).get("available", False)
                if count > 0 and win_ok:
                    st.success(f"{count:,} games · Model ready", icon="✅")
                elif count > 0:
                    st.warning(f"{count:,} games · No model", icon="⚠️")
                else:
                    st.error("No data ingested", icon="❌")
            except Exception:
                st.error("Tools unavailable", icon="❌")

    return page

# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Chatbot
# ══════════════════════════════════════════════════════════════════════════════

def render_chatbot() -> None:
    st.title("NFL Chatbot")
    if _get_demo_mode():
        _demo_banner()
    st.markdown(
        "Ask anything about NFL teams, players, predictions, or betting trends. "
        "The chatbot classifies your question, calls the right analytical tool, "
        "and returns a data-driven answer.",
    )

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                intent = meta.get("intent", "")
                conf = meta.get("confidence", 0.0)
                follow_ups = meta.get("follow_ups", [])
                color = _confidence_color(conf)
                st.markdown(
                    f'{_badge(intent.replace("_", " "), color)} '
                    f'<span class="chat-meta">{conf:.0%} confidence</span>',
                    unsafe_allow_html=True,
                )
                if follow_ups and not meta.get("needs_clarification"):
                    with st.expander("Suggested follow-ups", expanded=False):
                        for fu in follow_ups[:3]:
                            st.markdown(f"- {fu}")

    # Input
    if prompt := st.chat_input("Ask about a matchup, player, team, or betting trend…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                try:
                    responder = _load_responder()
                    resp = responder.respond(prompt)
                    st.markdown(resp.answer)
                    color = _confidence_color(resp.confidence)
                    st.markdown(
                        f'{_badge(resp.intent.replace("_", " "), color)} '
                        f'<span class="chat-meta">{resp.confidence:.0%} confidence</span>',
                        unsafe_allow_html=True,
                    )
                    if resp.follow_ups and not resp.needs_clarification:
                        with st.expander("Suggested follow-ups", expanded=False):
                            for fu in resp.follow_ups[:3]:
                                st.markdown(f"- {fu}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resp.answer,
                        "meta": {
                            "intent": resp.intent,
                            "confidence": resp.confidence,
                            "follow_ups": resp.follow_ups,
                            "needs_clarification": resp.needs_clarification,
                        },
                    })
                except Exception as exc:
                    st.error(f"Error: {exc}")

    if st.session_state.messages:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Matchup Predictor
# ══════════════════════════════════════════════════════════════════════════════

def render_matchup() -> None:
    st.title("Matchup Predictor")
    if _get_demo_mode():
        _demo_banner()
    st.markdown(
        "Select two teams to generate a win probability estimate from the trained model."
    )

    col1, col2, col3, col4 = st.columns([3, 3, 1.5, 1.5])
    with col1:
        home_sel = st.selectbox("Home team", TEAM_OPTIONS, index=TEAM_OPTIONS.index("Kansas City Chiefs (KC)"))
    with col2:
        away_sel = st.selectbox("Away team", TEAM_OPTIONS, index=TEAM_OPTIONS.index("Buffalo Bills (BUF)"))
    with col3:
        season = st.number_input("Season", min_value=2000, max_value=2030, value=None, placeholder="Optional")
    with col4:
        week = st.number_input("Week", min_value=1, max_value=22, value=None, placeholder="Optional")

    home_abbr = TEAM_NAME_TO_ABBR[home_sel]
    away_abbr = TEAM_NAME_TO_ABBR[away_sel]

    if home_abbr == away_abbr:
        st.warning("Home and away teams must be different.")
        return

    if st.button("Run Prediction", type="primary", use_container_width=False):
        with st.spinner("Running model…"):
            try:
                tools = _load_tools()
                result = tools.predict_matchup(
                    home_abbr, away_abbr,
                    season=int(season) if season else None,
                    week=int(week) if week else None,
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        if result["status"] == "error":
            _no_model_notice()
            return

        pred = result.get("prediction") or {}
        expl = result.get("explanation") or {}
        home_prob = pred.get("home_win_prob")
        away_prob = pred.get("away_win_prob")
        winner = pred.get("predicted_winner", home_abbr)
        confidence = pred.get("confidence", "unknown")

        st.markdown("---")

        # Win probability chart
        if home_prob is not None and away_prob is not None:
            fig = win_probability_chart(home_prob, away_prob, home_abbr, away_abbr)
            st.plotly_chart(fig, use_container_width=True)

        # Summary metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(f"{home_abbr} win prob", _pct(home_prob))
        with m2:
            st.metric(f"{away_abbr} win prob", _pct(away_prob))
        with m3:
            conf_color = _confidence_color(home_prob or 0.5)
            st.metric("Predicted winner", winner)
            st.markdown(_badge(f"Confidence: {confidence}", conf_color), unsafe_allow_html=True)

        if result.get("status") == "degraded":
            _data_warning(result.get("error", "Limited historical data for this matchup."))

        # Explanation
        headline = expl.get("headline")
        if headline:
            st.markdown("---")
            st.markdown(f"**{headline}**")
            prob_stmt = expl.get("probability_statement", "")
            if prob_stmt:
                st.markdown(prob_stmt)

        # Key factors
        key_factors = expl.get("key_factors") or []
        pos_factors = expl.get("positive_factors") or []
        neg_factors = expl.get("negative_factors") or []

        if key_factors or pos_factors or neg_factors:
            st.markdown("---")
            _section("Model factors")
            f1, f2 = st.columns(2)
            with f1:
                st.markdown(f"**Favoring {winner}**")
                for f in (pos_factors or key_factors)[:4]:
                    txt = f.get("plain_english", str(f)) if isinstance(f, dict) else str(f)
                    st.markdown(
                        f'<div class="factor-row factor-pos">✓ {txt}</div>',
                        unsafe_allow_html=True,
                    )
            with f2:
                st.markdown("**Working against**")
                for f in neg_factors[:4]:
                    txt = f.get("plain_english", str(f)) if isinstance(f, dict) else str(f)
                    st.markdown(
                        f'<div class="factor-row factor-neg">✗ {txt}</div>',
                        unsafe_allow_html=True,
                    )

        # Warnings
        warnings = expl.get("warnings") or []
        for w in warnings:
            st.warning(w)

        st.markdown(
            "<small>Predictions are statistical estimates and not betting advice.</small>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Player Comparison
# ══════════════════════════════════════════════════════════════════════════════

def render_players() -> None:
    st.title("Player Comparison")
    if _get_demo_mode():
        _demo_banner()
    st.markdown("Compare two players across key performance metrics.")

    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        player_a = st.text_input("Player A", placeholder="Patrick Mahomes")
    with col2:
        player_b = st.text_input("Player B", placeholder="Josh Allen")
    with col3:
        season = st.number_input(
            "Season filter", min_value=2000, max_value=2030,
            value=None, placeholder="All seasons",
        )

    if not st.button("Compare", type="primary"):
        st.markdown("---")
        st.info("Enter two player names above and click Compare.", icon="ℹ️")
        return

    if not player_a.strip() or not player_b.strip():
        st.warning("Enter names for both players.")
        return

    with st.spinner("Fetching player data…"):
        try:
            tools = _load_tools()
            result = tools.compare_players(
                player_a.strip(),
                player_b.strip(),
                seasons=[int(season)] if season else None,
            )
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
            return

    if result["status"] == "error":
        st.error(result.get("error", "Could not compare players."))
        st.info("Make sure player data has been ingested: `python scripts/ingest.py`")
        return

    if result["status"] == "degraded":
        _data_warning(result.get("error", "Player database unavailable."))
        return

    comp = result.get("comparison") or {}
    a_data = comp.get("player_a") or {}
    b_data = comp.get("player_b") or {}
    a_stats = a_data.get("aggregate_stats") or {}
    b_stats = b_data.get("aggregate_stats") or {}
    a_name = a_data.get("display_name", player_a)
    b_name = b_data.get("display_name", player_b)
    position = a_data.get("position", "")

    st.markdown("---")
    header_col1, header_col2 = st.columns(2)
    with header_col1:
        st.markdown(f"### {a_name}")
        if position:
            st.markdown(_badge(position, "blue"), unsafe_allow_html=True)
    with header_col2:
        st.markdown(f"### {b_name}")
        if position:
            st.markdown(_badge(position, "amber"), unsafe_allow_html=True)

    # Visual comparison chart
    if a_stats or b_stats:
        st.markdown("---")
        chart_type = st.radio(
            "Chart type", ["Grouped bar", "Radar"], horizontal=True, label_visibility="collapsed"
        )
        fig = player_comparison_chart(
            a_stats, b_stats, a_name, b_name,
            chart_type="radar" if chart_type == "Radar" else "bar",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Stats table
    if a_stats or b_stats:
        st.markdown("---")
        _section("Career / filtered stats")
        all_keys = list(dict.fromkeys(list(a_stats.keys()) + list(b_stats.keys())))
        rows = []
        for k in all_keys[:20]:
            a_val = a_stats.get(k)
            b_val = b_stats.get(k)
            a_str = f"{a_val:.1f}" if isinstance(a_val, float) else (str(a_val) if a_val is not None else "—")
            b_str = f"{b_val:.1f}" if isinstance(b_val, float) else (str(b_val) if b_val is not None else "—")
            rows.append({"Stat": k.replace("_", " ").title(), a_name: a_str, b_name: b_str})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Detailed stats unavailable. Check that player data has been ingested.")

    # Percentile ranks
    a_pcts = a_data.get("percentiles") or {}
    b_pcts = b_data.get("percentiles") or {}
    if a_pcts or b_pcts:
        st.markdown("---")
        _section(f"Percentile ranks vs {position} peers")
        pct_rows = []
        for k in list(a_pcts.keys())[:8]:
            pct_rows.append({
                "Metric": k.replace("_", " ").title(),
                f"{a_name} (pct)": f"{a_pcts.get(k, 0):.0f}th" if a_pcts.get(k) is not None else "—",
                f"{b_name} (pct)": f"{b_pcts.get(k, 0):.0f}th" if b_pcts.get(k) is not None else "—",
            })
        if pct_rows:
            st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Betting Trends
# ══════════════════════════════════════════════════════════════════════════════

def render_trends() -> None:
    st.title("Betting Trends")
    if _get_demo_mode():
        _demo_banner()
    st.markdown("Explore historical cover rates, over/under trends, and spread analysis.")

    col1, col2 = st.columns([3, 1])
    with col1:
        preset = st.selectbox("Common questions", ["Custom…"] + TREND_QUESTIONS)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # vertical align

    if preset == "Custom…":
        question = st.text_input("Custom question", placeholder="How often do the Eagles cover at home?")
    else:
        question = preset

    if not st.button("Analyze", type="primary"):
        st.markdown("---")
        st.info("Select a question above and click Analyze.", icon="ℹ️")
        return

    if not question.strip():
        st.warning("Enter a question.")
        return

    with st.spinner("Running analysis…"):
        try:
            tools = _load_tools()
            result = tools.answer_betting_trend(question.strip())
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    if result["status"] == "degraded":
        _data_warning(result.get("error", "No modeling table found."))
        st.info("Run `python scripts/build_features.py` to enable betting trend analysis.")
        return

    st.markdown("---")

    # Plain summary — prominent
    plain = result.get("plain_summary", "")
    if plain:
        st.markdown(f"### {plain}")

    data = result.get("result") or {}
    summary = data.get("summary") or {}
    trend_type = result.get("trend_type", "")

    # Wilson CI
    ci = summary.get("wilson_95_ci") or {}
    if ci.get("low") is not None:
        lo, hi = ci["low"], ci["high"]
        c1, c2, c3 = st.columns(3)
        rate_key = next((k for k in summary if "rate" in k and isinstance(summary[k], float)), None)
        if rate_key:
            c1.metric("Rate", _pct(summary[rate_key]))
        c2.metric("95% CI lower", _pct(lo))
        c3.metric("95% CI upper", _pct(hi))

    # Table results + chart
    for key in ("table", "by_season", "buckets"):
        rows = data.get(key)
        if isinstance(rows, list) and rows:
            df_trend = pd.DataFrame(rows)
            # Team cover rate chart when table has team + cover_rate columns
            if "team" in df_trend.columns and "cover_rate" in df_trend.columns:
                fig = team_cover_rate_chart(
                    df_trend,
                    title="ATS Cover Rate by Team",
                    ci_low_col="ci_low" if "ci_low" in df_trend.columns else None,
                    ci_high_col="ci_high" if "ci_high" in df_trend.columns else None,
                )
                st.plotly_chart(fig, use_container_width=True)
            # Formatted table
            _section("Detailed breakdown")
            df_display = df_trend.copy()
            for col in df_display.columns:
                if (
                    pd.api.types.is_float_dtype(df_display[col])
                    and col not in ("n", "count", "games", "ci_low", "ci_high", "spread")
                    and df_display[col].between(0, 1).all(skipna=True)
                ):
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                    )
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            break

    # Season breakdown
    by_season = data.get("by_season")
    if isinstance(by_season, list) and by_season:
        _section("By season")
        st.dataframe(pd.DataFrame(by_season), use_container_width=True, hide_index=True)

    st.markdown(
        "<small>Historical trends are informational and do not constitute betting advice.</small>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Page 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════

def render_model_performance() -> None:
    st.title("Model Performance")
    if _get_demo_mode():
        _demo_banner()
    st.markdown(
        "Training metrics for the win probability and against-the-spread models. "
        "Both models use chronological splits — no future data leaks into training."
    )

    try:
        tools = _load_tools()
        result = tools.get_model_metrics()
    except Exception as exc:
        st.error(f"Could not load metrics: {exc}")
        return

    if result["status"] == "error":
        _no_model_notice()
        return

    win = result.get("win_model") or {}
    spread = result.get("spread_model") or {}

    tab_win, tab_spread = st.tabs(["Win Probability Model", "ATS Spread Model"])

    with tab_win:
        _render_model_tab(win, "win")

    with tab_spread:
        _render_model_tab(spread, "spread")

    # Feature importance
    model_dir = result.get("model_dir", "models")
    _render_feature_importance(model_dir)


def _render_model_tab(metrics: dict, kind: str) -> None:
    if not metrics.get("available"):
        msg = (
            "Win model not trained yet. Run `python scripts/train_model.py --model win`."
            if kind == "win" else
            "Spread model not trained yet. Run `python scripts/train_model.py --model spread`."
        )
        st.info(msg, icon="ℹ️")
        return

    model_name = (metrics.get("model_name") or "").replace("_", " ").title()
    if model_name:
        st.markdown(f"**Best model:** {model_name}")

    interp = metrics.get("interpretation")
    if interp:
        st.success(interp)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", _pct(metrics.get("accuracy")))
    c2.metric("ROC-AUC", _round(metrics.get("roc_auc"), 3))
    c3.metric("F1 Score", _round(metrics.get("f1"), 3))
    c4.metric("Brier Score", _round(metrics.get("brier_score"), 3))

    # Split info
    test_season = metrics.get("test_season")
    train_rows = metrics.get("train_rows")
    test_rows = metrics.get("test_rows")
    if test_season or train_rows:
        st.markdown("---")
        _section("Data split")
        s1, s2, s3 = st.columns(3)
        s1.metric("Test season", str(test_season) if test_season else "—")
        s2.metric("Training games", f"{train_rows:,}" if train_rows else "—")
        s3.metric("Test games", f"{test_rows:,}" if test_rows else "—")
        st.markdown(
            "<small>Model trained on all seasons before the test season. "
            "No data from the test season was seen during training.</small>",
            unsafe_allow_html=True,
        )

    # Confusion matrix + calibration curve
    cm_raw = metrics.get("confusion_matrix")
    if cm_raw:
        st.markdown("---")
        vis1, vis2 = st.columns(2)
        with vis1:
            labels = (
                ["Didn't Cover", "Covered"] if kind == "spread"
                else ["Away Win", "Home Win"]
            )
            fig = confusion_matrix_chart(cm_raw, labels=labels)
            st.plotly_chart(fig, use_container_width=True)
        with vis2:
            # Compute calibration from model artifacts
            try:
                tools = _load_tools()
                model_dir = str(tools._model_dir)
                data_dir = str(tools._data_dir)
                csv_path = f"{data_dir}/modeling_table.csv"
                cal = compute_calibration_data(model_dir, csv_path, test_season)
                if cal is not None:
                    y_prob, y_true = cal
                    fig2 = calibration_curve_chart(y_prob, y_true)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Run the model pipeline to generate calibration data.", icon="ℹ️")
            except Exception:
                pass


def _render_feature_importance(model_dir: str) -> None:
    df = load_feature_importance(model_dir, kind="win")
    if df is None:
        return

    st.markdown("---")
    _section("Feature importance (top 20)")
    fig = feature_importance_chart(df)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page 6 — About
# ══════════════════════════════════════════════════════════════════════════════

def render_about() -> None:
    st.title("About This Project")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
NFL AI Chatbot is an end-to-end machine learning system for NFL game prediction
and natural-language analytics. It ingests five-plus seasons of play-by-play and
game data, engineers over 75 features, trains calibrated ML classifiers, and
surfaces results through a conversational interface.

The project was built as a portfolio demonstration of production-grade data
engineering, ML pipelines, and application development in Python.
""")

        st.markdown("### How it works")
        st.markdown("""
1. **Data ingestion** — nflverse game data, player stats, injury reports, and betting odds
2. **Feature engineering** — rolling form, Elo ratings, rest days, spread context
3. **Model training** — Logistic Regression, Random Forest, XGBoost with seasonal CV and calibration
4. **Prediction** — win probability + against-the-spread model
5. **Chatbot** — rule-based intent classification routes questions to the right analytical tool
6. **Optional LLM rewrite** — if an API key is set, Claude rewrites answers into natural prose
""")

        st.markdown("### Tech stack")

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown("**Data & ML**")
            st.markdown("""
- Python 3.11+
- pandas / numpy
- scikit-learn
- XGBoost
- SQLite / SQLAlchemy
""")
        with tc2:
            st.markdown("**API & Frontend**")
            st.markdown("""
- FastAPI
- Pydantic v2
- Streamlit
- uvicorn
""")
        with tc3:
            st.markdown("**Chatbot**")
            st.markdown("""
- Rule-based intent classifier
- Anthropic Claude (optional)
- OpenAI (optional)
- Local LLM (optional)
""")

        st.markdown("### Data sources")
        st.markdown("""
| Source | Contents |
|---|---|
| nflverse | Play-by-play, schedules, rosters, team stats (2000–present) |
| Kaggle / The Odds API | Historical betting lines and game totals |
| NFL injury reports | Weekly practice participation and designations |
""")

        st.markdown("### Ethical note")
        st.markdown("""
This project does not make betting recommendations.
All probability estimates are statistical model outputs with documented uncertainty.
Historical trends are descriptive, not predictive guarantees.
""")

    with col2:
        st.markdown("### Model performance")
        try:
            tools = _load_tools()
            metrics = tools.get_model_metrics()
            win = metrics.get("win_model") or {}
            spread = metrics.get("spread_model") or {}
            if win.get("available"):
                st.metric("Win model accuracy", _pct(win.get("accuracy")))
                st.metric("Win model AUC", _round(win.get("roc_auc"), 3))
            if spread.get("available"):
                st.metric("ATS model accuracy", _pct(spread.get("accuracy")))
            if not win.get("available") and not spread.get("available"):
                st.info("Train models to see live metrics.")
        except Exception:
            st.info("Model metrics unavailable.")

        st.markdown("### Data status")
        try:
            tools = _load_tools()
            s = tools.get_available_data_summary()
            db = s.get("database", {})
            games = db.get("games", {})
            players = db.get("player_game_stats", {})
            injuries = db.get("injuries", {})
            if games.get("count", 0) > 0:
                st.metric("Games in DB", f"{games['count']:,}")
                mn, mx = games.get("min_season"), games.get("max_season")
                if mn and mx:
                    st.caption(f"Seasons {mn}–{mx}")
            if players.get("count", 0) > 0:
                st.metric("Player stat rows", f"{players['count']:,}")
            if injuries.get("count", 0) > 0:
                st.metric("Injury records", f"{injuries['count']:,}")
        except Exception:
            st.info("Connect database to see data status.")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    page = _sidebar()

    if page == "Chatbot":
        render_chatbot()
    elif page == "Matchup Predictor":
        render_matchup()
    elif page == "Player Comparison":
        render_players()
    elif page == "Betting Trends":
        render_trends()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
