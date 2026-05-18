"""
Rule-based response generator for the NFL AI Chatbot.

Flow:
  1. classify_intent(question)   → IntentResult
  2. dispatch to the matching tool
  3. format the tool's JSON output into a plain-English answer

No LLM required.  The response layer owns the text; the tools layer owns the
numbers.  Numbers always come from tools — this module never invents statistics.

Usage
-----
    from nfl_chatbot.chatbot.responder import Responder
    resp = Responder()
    result = resp.respond("Who will win Chiefs vs Bills?")
    print(result.answer)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nfl_chatbot.chatbot.intents import IntentResult, classify_intent
from nfl_chatbot.chatbot.tools import ChatbotTools


# ── Response container ────────────────────────────────────────────────────────

@dataclass
class Response:
    """The chatbot's reply to a single user question."""

    answer: str
    intent: str
    confidence: float
    tool_result: dict | None = None
    follow_ups: list[str] = field(default_factory=list)
    has_caveat: bool = False
    needs_clarification: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "confidence": self.confidence,
            "tool_result": self.tool_result,
            "follow_ups": self.follow_ups,
            "has_caveat": self.has_caveat,
            "needs_clarification": self.needs_clarification,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _none_to_na(v: Any) -> str:
    if v is None:
        return "N/A"
    return str(v)


def _round_or_na(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return str(round(v, decimals))


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"• {i}" for i in items)


# ══════════════════════════════════════════════════════════════════════════════
# Responder
# ══════════════════════════════════════════════════════════════════════════════

class Responder:
    """
    Converts a natural-language question into a Response by:
    1. Classifying the intent.
    2. Calling the appropriate ChatbotTools method.
    3. Formatting the tool output as plain English.

    Parameters
    ----------
    tools : ChatbotTools | None
        Tool container.  Creates a default one if None.
    """

    def __init__(self, tools: ChatbotTools | None = None) -> None:
        self._tools = tools or ChatbotTools()

    # ── Public interface ──────────────────────────────────────────────────────

    def respond(self, question: str) -> Response:
        """
        Classify *question*, call the matching tool, and return a Response.

        Never raises — unexpected errors become a polite error message.
        """
        if not question or not question.strip():
            return Response(
                answer="Please ask me a question about NFL teams, players, or predictions!",
                intent="unknown",
                confidence=0.0,
                needs_clarification=True,
            )

        try:
            ir = classify_intent(question)
            return self._dispatch(ir, question)
        except Exception as exc:
            return Response(
                answer=(
                    f"Sorry, something went wrong while processing your question. "
                    f"({type(exc).__name__}: {exc})"
                ),
                intent="unknown",
                confidence=0.0,
            )

    # ── Intent dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, ir: IntentResult, question: str) -> Response:
        intent = ir.intent
        entities = ir.extracted_entities

        if ir.needs_clarification and intent not in ("general_help", "unknown"):
            return self._clarification_response(ir, question)

        if intent == "matchup_prediction":
            return self._handle_matchup(ir, entities)

        if intent == "player_comparison":
            return self._handle_player_comparison(ir, entities, question)

        if intent == "betting_trend":
            return self._handle_betting_trend(ir, question)

        if intent == "team_summary":
            return self._handle_team_summary(ir, entities)

        if intent == "model_explanation":
            return self._handle_model_explanation(ir)

        if intent == "data_question":
            return self._handle_data_question(ir, entities, question)

        if intent == "general_help":
            return self._handle_general_help(ir)

        # unknown
        return self._handle_unknown(ir)

    # ── Clarification ─────────────────────────────────────────────────────────

    def _clarification_response(self, ir: IntentResult, question: str) -> Response:
        intent = ir.intent
        hints = {
            "matchup_prediction": (
                "I'd love to predict a matchup! Could you name the two teams? "
                "For example: 'Who will win Chiefs vs Bills?'"
            ),
            "player_comparison": (
                "I can compare two players — just tell me their names. "
                "For example: 'Compare Patrick Mahomes and Josh Allen.'"
            ),
            "team_summary": (
                "Which team would you like a summary for? "
                "For example: 'How are the Eagles doing this season?'"
            ),
            "betting_trend": (
                "I can pull up betting trends. Could you be more specific? "
                "For example: 'How often do underdogs cover the spread?'"
            ),
        }
        answer = hints.get(
            intent,
            f"I think you're asking about {intent.replace('_', ' ')}, but I need a bit more detail. "
            "Could you rephrase your question?",
        )
        return Response(
            answer=answer,
            intent=intent,
            confidence=ir.confidence,
            needs_clarification=True,
            follow_ups=self._generic_follow_ups(),
        )

    # ── Tool: predict_matchup ─────────────────────────────────────────────────

    def _handle_matchup(self, ir: IntentResult, entities: dict) -> Response:
        teams = entities.get("teams", [])
        season = entities.get("season")
        week = entities.get("week")

        if len(teams) < 2:
            return self._clarification_response(ir, "")

        home, away = teams[0], teams[1]
        result = self._tools.predict_matchup(home, away, season=season, week=week)

        if result["status"] == "error":
            return Response(
                answer=(
                    f"I wasn't able to generate a prediction for {home} vs {away}. "
                    f"{result['error']}"
                ),
                intent=ir.intent,
                confidence=ir.confidence,
                tool_result=result,
                follow_ups=["Run `python scripts/train_model.py` to train the model first."],
            )

        answer = self._format_matchup_answer(result)
        caveats = result["status"] == "degraded"

        return Response(
            answer=answer,
            intent=ir.intent,
            confidence=ir.confidence,
            tool_result=result,
            follow_ups=[
                f"Ask me to explain why I picked {result.get('home_team', home)}.",
                "Ask for the spread model prediction (ATS).",
                f"Ask 'How are the {home} doing this season?' for team context.",
            ],
            has_caveat=caveats,
        )

    @staticmethod
    def _format_matchup_answer(result: dict) -> str:
        home = result.get("home_team", "Home")
        away = result.get("away_team", "Away")
        pred = result.get("prediction") or {}
        expl = result.get("explanation") or {}

        home_prob = pred.get("home_win_prob")
        away_prob = pred.get("away_win_prob")
        predicted_winner = pred.get("predicted_winner", "Unknown")
        conf = pred.get("confidence", "unknown")

        headline = expl.get("headline") or f"Prediction: {predicted_winner}"
        prob_stmt = expl.get("probability_statement") or (
            f"{home}: {_pct(home_prob)}  |  {away}: {_pct(away_prob)}"
        )
        conf_stmt = expl.get("confidence_statement") or f"Confidence: {conf}"

        lines = [
            f"**{away} @ {home}**",
            "",
            headline,
            "",
            prob_stmt,
            conf_stmt,
        ]

        # Key factors
        key_factors = expl.get("key_factors") or []
        if key_factors:
            lines += ["", "**Key factors:**"]
            for f in key_factors[:3]:
                if isinstance(f, dict):
                    lines.append(f"• {f.get('plain_english', str(f))}")
                else:
                    lines.append(f"• {f}")

        # Warnings
        warnings = expl.get("warnings") or []
        if warnings or result["status"] == "degraded":
            lines.append("")
            if warnings:
                for w in warnings:
                    lines.append(f"⚠ {w}")
            if result["status"] == "degraded":
                lines.append(f"⚠ {result['error']}")

        # Disclaimer
        lines += [
            "",
            "_This is a statistical model prediction, not betting advice._",
        ]

        return "\n".join(lines)

    # ── Tool: compare_players ─────────────────────────────────────────────────

    def _handle_player_comparison(
        self, ir: IntentResult, entities: dict, question: str
    ) -> Response:
        players = entities.get("players", [])
        season = entities.get("season")
        seasons = [season] if season else None

        if len(players) < 2:
            return Response(
                answer=(
                    "I'd need the names of both players to compare them. "
                    "Try something like: 'Compare Patrick Mahomes and Josh Allen this season.'"
                ),
                intent=ir.intent,
                confidence=ir.confidence,
                needs_clarification=True,
            )

        result = self._tools.compare_players(players[0], players[1], seasons=seasons)

        if result["status"] == "error":
            return Response(
                answer=(
                    f"I couldn't compare {players[0]} and {players[1]}. "
                    f"{result['error']}"
                ),
                intent=ir.intent,
                confidence=ir.confidence,
                tool_result=result,
            )

        answer = self._format_comparison_answer(result)
        return Response(
            answer=answer,
            intent=ir.intent,
            confidence=ir.confidence,
            tool_result=result,
            follow_ups=[
                f"Ask about {players[0]}'s team: 'How are the [team] doing?'",
                "Ask for a matchup prediction if these players face each other.",
            ],
        )

    @staticmethod
    def _format_comparison_answer(result: dict) -> str:
        comp = result.get("comparison") or {}
        pa = result.get("player_a", "Player A")
        pb = result.get("player_b", "Player B")

        if not comp or comp.get("status") == "error":
            return (
                f"I found both players but couldn't generate a detailed comparison. "
                f"{comp.get('error', 'Player data may be missing from the database.')}"
            )

        a_summary = comp.get("player_a") or {}
        b_summary = comp.get("player_b") or {}
        a_stats = a_summary.get("aggregate_stats") or {}
        b_stats = b_summary.get("aggregate_stats") or {}
        a_name = a_summary.get("display_name", pa)
        b_name = b_summary.get("display_name", pb)
        position = a_summary.get("position", "")

        lines = [f"**{a_name} vs {b_name}**"]
        if position:
            lines.append(f"Position: {position}")
        seasons_info = comp.get("seasons_compared")
        if seasons_info:
            lines.append(f"Seasons: {seasons_info}")
        lines.append("")

        # Format stats side by side
        if a_stats or b_stats:
            all_keys = list(a_stats.keys() or b_stats.keys())
            lines.append(f"{'Stat':<30} {a_name:<20} {b_name}")
            lines.append("-" * 70)
            for k in all_keys[:15]:
                a_val = a_stats.get(k)
                b_val = b_stats.get(k)
                a_str = _round_or_na(a_val, 1) if isinstance(a_val, float) else _none_to_na(a_val)
                b_str = _round_or_na(b_val, 1) if isinstance(b_val, float) else _none_to_na(b_val)
                lines.append(f"{k:<30} {a_str:<20} {b_str}")
        else:
            lines.append("Detailed stats unavailable — check that player data has been ingested.")

        # Percentile comparison if available
        a_pcts = a_summary.get("percentiles") or {}
        b_pcts = b_summary.get("percentiles") or {}
        if a_pcts or b_pcts:
            lines += ["", "**Percentile ranks vs position peers:**"]
            for k in list(a_pcts.keys())[:5]:
                a_p = a_pcts.get(k)
                b_p = b_pcts.get(k)
                a_str = f"{a_p:.0f}th" if a_p is not None else "N/A"
                b_str = f"{b_p:.0f}th" if b_p is not None else "N/A"
                lines.append(f"  {k}: {a_name} {a_str}  |  {b_name} {b_str}")

        return "\n".join(lines)

    # ── Tool: answer_betting_trend ────────────────────────────────────────────

    def _handle_betting_trend(self, ir: IntentResult, question: str) -> Response:
        result = self._tools.answer_betting_trend(question)

        if result["status"] == "degraded":
            return Response(
                answer=(
                    "I don't have enough data to answer that betting trend question yet. "
                    f"{result['error']}"
                ),
                intent=ir.intent,
                confidence=ir.confidence,
                tool_result=result,
                follow_ups=["Run `python scripts/build_features.py` to build the modeling table."],
            )

        answer = self._format_trend_answer(result)
        return Response(
            answer=answer,
            intent=ir.intent,
            confidence=ir.confidence,
            tool_result=result,
            follow_ups=[
                "Ask about a specific team: 'How often do the Chiefs cover?'",
                "Ask about spread buckets: 'What happens when the spread is over 7 points?'",
                "Ask about over/under: 'Which teams go over most often?'",
            ],
            has_caveat=True,
        )

    @staticmethod
    def _format_trend_answer(result: dict) -> str:
        summary = result.get("plain_summary", "")
        trend = result.get("trend_type", "")
        data = result.get("result") or {}

        lines = []

        if summary:
            lines.append(summary)

        # Show a highlights table for ranked results
        for key in ("table", "by_season", "buckets"):
            rows = data.get(key)
            if isinstance(rows, list) and rows:
                lines.append("")
                cols = list(rows[0].keys())[:5]
                header = "  ".join(f"{c[:16]:<16}" for c in cols)
                lines.append(header)
                lines.append("-" * len(header))
                for row in rows[:8]:
                    vals = "  ".join(
                        f"{str(row.get(c, 'N/A'))[:16]:<16}" for c in cols
                    )
                    lines.append(vals)
                if len(rows) > 8:
                    lines.append(f"  ... and {len(rows) - 8} more rows")
                break

        # Wilson CI details
        s = data.get("summary") or {}
        ci = s.get("wilson_95_ci")
        if ci:
            lo, hi = ci.get("low"), ci.get("high")
            if lo is not None and hi is not None:
                lines.append(f"\n95% confidence interval: {_pct(lo)} – {_pct(hi)}")

        lines.append(
            "\n_Historical trends reflect past data and do not constitute betting advice._"
        )
        return "\n".join(lines)

    # ── Tool: summarize_team ──────────────────────────────────────────────────

    def _handle_team_summary(self, ir: IntentResult, entities: dict) -> Response:
        teams = entities.get("teams", [])
        season = entities.get("season")
        seasons = [season] if season else None

        if not teams:
            return self._clarification_response(ir, "")

        team = teams[0]
        result = self._tools.summarize_team(team, seasons=seasons)

        if result["status"] in ("degraded", "error"):
            return Response(
                answer=(
                    f"I don't have data for {team} yet. {result['error']}"
                ),
                intent=ir.intent,
                confidence=ir.confidence,
                tool_result=result,
                follow_ups=["Run `python scripts/ingest.py` to load game data."],
            )

        answer = self._format_team_summary_answer(result)
        return Response(
            answer=answer,
            intent=ir.intent,
            confidence=ir.confidence,
            tool_result=result,
            follow_ups=[
                f"Ask for a matchup prediction: 'Who will win {team} vs [opponent]?'",
                f"Ask about betting trends: 'How often do the {team} cover?'",
            ],
        )

    @staticmethod
    def _format_team_summary_answer(result: dict) -> str:
        team = result.get("team", "")
        summary = result.get("summary") or {}
        overall = summary.get("overall") or {}
        by_season = summary.get("by_season") or []
        seasons = result.get("seasons") or summary.get("seasons_in_data") or []

        lines = [f"**{team} — Season Summary**"]

        if seasons:
            season_str = (
                str(seasons[0])
                if len(seasons) == 1
                else f"{min(seasons)}–{max(seasons)}"
            )
            lines.append(f"Seasons: {season_str}")

        if overall:
            w = overall.get("wins", 0)
            l = overall.get("losses", 0)
            win_pct = _pct(overall.get("win_pct"))
            scored = _round_or_na(overall.get("avg_pts_scored"))
            allowed = _round_or_na(overall.get("avg_pts_allowed"))
            diff = overall.get("avg_pt_diff")
            diff_str = (f"+{diff}" if (diff or 0) > 0 else str(diff)) if diff is not None else "N/A"

            lines += [
                "",
                f"Record: {w}–{l} ({win_pct} win rate)",
                f"Points scored:  {scored} per game",
                f"Points allowed: {allowed} per game",
                f"Point differential: {diff_str} per game",
            ]

        if by_season:
            lines += ["", "**By season:**"]
            lines.append(f"{'Season':<8} {'W':<4} {'L':<4} {'Win%':<8} {'Pts For':<10} {'Pts Against'}")
            lines.append("-" * 50)
            for row in by_season[:6]:
                lines.append(
                    f"{row.get('season', ''):<8} "
                    f"{row.get('wins', ''):<4} "
                    f"{row.get('losses', ''):<4} "
                    f"{_pct(row.get('win_pct')):<8} "
                    f"{_round_or_na(row.get('avg_pts_scored')):<10} "
                    f"{_round_or_na(row.get('avg_pts_allowed'))}"
                )

        source = summary.get("source")
        if source == "modeling_table_csv":
            lines.append("\n_Stats derived from modeling table. Install DB for season-by-season breakdown._")

        return "\n".join(lines)

    # ── Tool: explain_last_prediction ─────────────────────────────────────────

    def _handle_model_explanation(self, ir: IntentResult) -> Response:
        result = self._tools.explain_last_prediction()

        if result["status"] == "degraded":
            return Response(
                answer=(
                    "I haven't made a prediction yet this session. "
                    "Ask me to predict a matchup first — for example, "
                    "'Who will win Chiefs vs Bills?' — and then I can explain my reasoning."
                ),
                intent=ir.intent,
                confidence=ir.confidence,
                tool_result=result,
                needs_clarification=True,
                follow_ups=["Ask: 'Who will win Chiefs vs Bills?'"],
            )

        answer = self._format_explanation_answer(result)
        return Response(
            answer=answer,
            intent=ir.intent,
            confidence=ir.confidence,
            tool_result=result,
            follow_ups=[
                "Ask about the team's season: 'How are they doing this year?'",
                "Ask about ATS trends: 'How often do they cover?'",
            ],
        )

    @staticmethod
    def _format_explanation_answer(result: dict) -> str:
        home = result.get("home_team", "Home")
        away = result.get("away_team", "Away")
        pred_sum = result.get("prediction_summary") or {}
        expl = result.get("explanation") or {}

        headline = expl.get("headline") or "Prediction explanation"
        prob_stmt = expl.get("probability_statement", "")
        conf_stmt = expl.get("confidence_statement", "")
        model_note = expl.get("model_note", "")

        lines = [f"**Explanation: {away} @ {home}**", "", headline, ""]
        if prob_stmt:
            lines.append(prob_stmt)
        if conf_stmt:
            lines.append(conf_stmt)

        pos_factors = expl.get("positive_factors") or []
        neg_factors = expl.get("negative_factors") or []

        winner = pred_sum.get("predicted_winner", home)

        if pos_factors:
            lines += ["", f"**Why {winner} is favoured:**"]
            for f in pos_factors[:4]:
                if isinstance(f, dict):
                    lines.append(f"• {f.get('plain_english', str(f))}")
                else:
                    lines.append(f"• {f}")

        if neg_factors:
            lines += ["", "**Concerns / factors working against them:**"]
            for f in neg_factors[:3]:
                if isinstance(f, dict):
                    lines.append(f"• {f.get('plain_english', str(f))}")
                else:
                    lines.append(f"• {f}")

        warnings = expl.get("warnings") or []
        if warnings:
            lines.append("")
            for w in warnings:
                lines.append(f"⚠ {w}")

        if model_note:
            lines += ["", f"_{model_note}_"]

        return "\n".join(lines)

    # ── Tool: data_question ───────────────────────────────────────────────────

    def _handle_data_question(
        self, ir: IntentResult, entities: dict, question: str
    ) -> Response:
        q = question.lower()
        teams = entities.get("teams", [])

        # Injury query → show DB status for injuries
        if re.search(r"\b(injur|injured|out|questionable|doubtful|ir)\b", q):
            return self._answer_injury_question(teams)

        # Standings → brief guidance
        if re.search(r"\bstanding\b", q):
            return self._answer_standings_question()

        # Score / result query
        if re.search(r"\b(score|result|final|win|lost|beat)\b", q) and re.search(r"\b(last|week|against|game)\b", q):
            return self._answer_score_question(teams)

        # Schedule / when do they play
        if re.search(r"\b(schedule|when|next game|play next)\b", q):
            return self._answer_schedule_question(teams)

        # General data availability
        data_result = self._tools.get_available_data_summary()
        answer = self._format_data_summary_answer(data_result, question)
        return Response(
            answer=answer,
            intent=ir.intent,
            confidence=ir.confidence,
            tool_result=data_result,
            follow_ups=[
                "Ask for team stats: 'How are the Eagles doing?'",
                "Ask for a prediction: 'Who will win Chiefs vs Bills?'",
            ],
        )

    def _answer_injury_question(self, teams: list[str]) -> Response:
        data_result = self._tools.get_available_data_summary()
        db = data_result.get("database") or {}
        injuries = db.get("injuries") or {}
        count = injuries.get("count", 0)
        min_s = injuries.get("min_season")
        max_s = injuries.get("max_season")

        if count == 0:
            answer = (
                "I don't have injury data loaded yet. "
                "Run `python scripts/ingest.py` to import injury reports."
            )
        else:
            team_str = f" for the {teams[0]}" if teams else ""
            range_str = f" from {min_s} to {max_s}" if min_s else ""
            answer = (
                f"I have {count:,} injury records{range_str} in the database. "
                f"I can show injury counts{team_str} in a team summary — try "
                f"'How are the {teams[0] if teams else 'Chiefs'} doing this season?'"
            )
        return Response(
            answer=answer,
            intent="data_question",
            confidence=0.8,
            tool_result=data_result,
            follow_ups=[
                "Ask 'How are the [team] doing?' to see injury-adjusted team context.",
            ],
        )

    def _answer_standings_question(self) -> Response:
        answer = (
            "I can show you individual team records but don't yet have a full standings table. "
            "Try asking 'How are the Chiefs doing?' or 'How are the Eagles doing?' "
            "for win-loss records and scoring averages for any team."
        )
        return Response(
            answer=answer,
            intent="data_question",
            confidence=0.75,
            follow_ups=[
                "Ask: 'How are the Chiefs doing this season?'",
                "Ask: 'How are the Eagles doing this season?'",
            ],
            needs_clarification=True,
        )

    def _answer_score_question(self, teams: list[str]) -> Response:
        if teams:
            result = self._tools.summarize_team(teams[0])
            answer = self._format_team_summary_answer(result) if result["status"] == "ok" else (
                f"I have game results in the database but couldn't fetch details for {teams[0]}. "
                f"{result.get('error', '')}"
            )
        else:
            result = self._tools.get_available_data_summary()
            db = result.get("database") or {}
            games = db.get("games") or {}
            count = games.get("count", 0)
            min_s = games.get("min_season")
            max_s = games.get("max_season")
            answer = (
                f"I have results for {count:,} games"
                + (f" from {min_s} to {max_s}." if min_s else ".")
                + " Try asking 'How are the [team] doing?' for a team's record."
            )
        return Response(
            answer=answer,
            intent="data_question",
            confidence=0.8,
            follow_ups=["Ask: 'How are the Chiefs doing this season?'"],
        )

    def _answer_schedule_question(self, teams: list[str]) -> Response:
        team_str = teams[0] if teams else "that team"
        answer = (
            f"I have game schedules in the database, but I don't yet have a dedicated "
            f"schedule-lookup tool for upcoming games. "
            f"I can tell you the {team_str}'s past record — try "
            f"'How are the {team_str} doing this season?'"
        )
        return Response(
            answer=answer,
            intent="data_question",
            confidence=0.7,
            needs_clarification=True,
            follow_ups=[f"Ask: 'How are the {team_str} doing this season?'"],
        )

    @staticmethod
    def _format_data_summary_answer(result: dict, question: str) -> str:
        db = result.get("database") or {}
        models = result.get("model_artifacts") or {}
        csvs = result.get("csv_files") or {}
        recs = result.get("recommendations") or []

        if result["status"] == "error":
            return (
                "It looks like the data pipeline hasn't been run yet. "
                + _bullet_list(recs)
            )

        parts = []
        games = db.get("games") or {}
        if games.get("count", 0) > 0:
            parts.append(
                f"• {games['count']:,} completed games"
                + (f" ({games.get('min_season')}–{games.get('max_season')})" if games.get("min_season") else "")
            )

        player_stats = db.get("player_game_stats") or {}
        if player_stats.get("count", 0) > 0:
            parts.append(f"• {player_stats['count']:,} player game stat rows")

        injuries = db.get("injuries") or {}
        if injuries.get("count", 0) > 0:
            parts.append(f"• {injuries['count']:,} injury records")

        csv_rows = csvs.get("modeling_table_rows")
        if csv_rows:
            parts.append(f"• {csv_rows:,} rows in the modeling table")

        win_model = models.get("win_model") or {}
        spread_model = models.get("spread_model") or {}
        model_parts = []
        if win_model.get("available"):
            model_parts.append("win predictor")
        if spread_model.get("available"):
            model_parts.append("spread predictor")
        if model_parts:
            parts.append(f"• Trained models: {', '.join(model_parts)}")

        header = "Here's what data I currently have available:\n"
        body = "\n".join(parts) if parts else "No data loaded yet."
        rec_str = ("\n\nSuggested next steps:\n" + _bullet_list(recs)) if recs and recs[0] != "All systems ready. The chatbot can answer all question types." else ""

        return header + body + rec_str

    # ── general_help ──────────────────────────────────────────────────────────

    def _handle_general_help(self, ir: IntentResult) -> Response:
        answer = (
            "**NFL AI Chatbot — What I can do:**\n\n"
            "• **Matchup predictions** — 'Who will win Chiefs vs Bills?'\n"
            "• **Player comparisons** — 'Compare Patrick Mahomes and Josh Allen'\n"
            "• **Betting trends** — 'How often do underdogs cover the spread?'\n"
            "• **Team summaries** — 'How are the Eagles doing this season?'\n"
            "• **Model explanations** — 'Why did you pick the Chiefs?'\n"
            "• **Data questions** — 'Who is injured on the Ravens?'\n\n"
            "I use a statistical model trained on historical NFL data. "
            "Predictions are for informational purposes only — not betting advice."
        )
        return Response(
            answer=answer,
            intent="general_help",
            confidence=1.0,
            follow_ups=[
                "Ask: 'Who will win Chiefs vs Bills?'",
                "Ask: 'How often do home favorites cover?'",
                "Ask: 'Compare Tyreek Hill and CeeDee Lamb'",
            ],
        )

    # ── unknown ───────────────────────────────────────────────────────────────

    def _handle_unknown(self, ir: IntentResult) -> Response:
        return Response(
            answer=(
                "I'm not sure how to answer that. I'm an NFL chatbot — I can help with:\n\n"
                "• Game predictions ('Who will win Chiefs vs Bills?')\n"
                "• Player comparisons ('Compare Mahomes and Allen')\n"
                "• Betting trends ('How often do underdogs cover?')\n"
                "• Team summaries ('How are the Eagles doing?')\n\n"
                "What would you like to know?"
            ),
            intent="unknown",
            confidence=ir.confidence,
            needs_clarification=True,
            follow_ups=self._generic_follow_ups(),
        )

    @staticmethod
    def _generic_follow_ups() -> list[str]:
        return [
            "Ask: 'Who will win Chiefs vs Bills?'",
            "Ask: 'How often do underdogs cover the spread?'",
            "Ask: 'What can you do?' for a full list of capabilities.",
        ]


# ── Module-level convenience ──────────────────────────────────────────────────

_default_responder: Responder | None = None


def _get_responder() -> Responder:
    global _default_responder
    if _default_responder is None:
        _default_responder = Responder()
    return _default_responder


def respond(question: str, tools: ChatbotTools | None = None) -> Response:
    """
    Classify *question* and return a Response.

    Uses the module-level default Responder if *tools* is not provided.
    """
    if tools is not None:
        return Responder(tools=tools).respond(question)
    return _get_responder().respond(question)
