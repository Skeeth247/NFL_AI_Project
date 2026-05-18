"""
Tests for the intent classification layer.

Coverage:
  - All 8 intents: clear-signal questions, near-ambiguous questions
  - Entity extraction: teams, players, season year, week number, spread
  - needs_clarification flag: low confidence, missing teams, missing players
  - Edge cases: empty string, single word, ALL CAPS, punctuation-heavy input
  - 37 sample questions total (>= 30 required)
"""

from __future__ import annotations

import pytest

from nfl_chatbot.chatbot.intents import (
    IntentResult,
    classify_intent,
    _extract_entities,
    _extract_teams,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _check(question: str, expected_intent: str, *, min_conf: float = 0.45) -> IntentResult:
    """Classify *question*, assert intent, and return the result."""
    result = classify_intent(question)
    assert result.intent == expected_intent, (
        f"Q: {question!r}\n"
        f"  Expected intent : {expected_intent!r}\n"
        f"  Got intent      : {result.intent!r}  (conf={result.confidence})"
    )
    assert result.confidence >= min_conf or result.intent == "unknown", (
        f"Confidence too low for {question!r}: {result.confidence} < {min_conf}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# matchup_prediction  (8 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchupPrediction:
    def test_who_will_win(self):
        r = _check("Who will win Chiefs vs Bills?", "matchup_prediction")
        assert "KC" in r.extracted_entities["teams"]
        assert "BUF" in r.extracted_entities["teams"]

    def test_will_team_beat(self):
        r = _check("Will the Eagles beat the Cowboys on Sunday?", "matchup_prediction")
        assert "PHI" in r.extracted_entities["teams"]
        assert "DAL" in r.extracted_entities["teams"]

    def test_game_prediction(self):
        _check("Give me a game prediction for 49ers vs Rams", "matchup_prediction")

    def test_who_do_you_pick(self):
        r = _check("Who do you pick to win this week: Ravens or Steelers?", "matchup_prediction")
        assert "BAL" in r.extracted_entities["teams"]
        assert "PIT" in r.extracted_entities["teams"]

    def test_moneyline(self):
        _check("What is the moneyline for Chiefs game this week?", "matchup_prediction")

    def test_upset_chance(self):
        _check(
            "What are the chances of an upset when the Browns play the Ravens?",
            "matchup_prediction",
        )

    def test_who_wins_short(self):
        _check("Who wins Eagles vs Giants?", "matchup_prediction")

    def test_win_prediction_capslock(self):
        _check("WHO WILL WIN THE SUPER BOWL?", "matchup_prediction")


# ══════════════════════════════════════════════════════════════════════════════
# player_comparison  (6 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestPlayerComparison:
    def test_compare_two_players(self):
        r = _check(
            "Compare Patrick Mahomes and Josh Allen this season",
            "player_comparison",
        )
        assert len(r.extracted_entities["players"]) >= 1

    def test_who_is_better(self):
        _check("Who is the better QB, Lamar Jackson or Jalen Hurts?", "player_comparison")

    def test_head_to_head(self):
        _check("Give me a head-to-head breakdown of CeeDee Lamb vs Tyreek Hill", "player_comparison")

    def test_who_has_more_yards(self):
        _check(
            "Who has more receiving yards, Cooper Kupp or Davante Adams?",
            "player_comparison",
        )

    def test_compare_without_names(self):
        r = classify_intent("Compare the two quarterbacks")
        assert r.intent == "player_comparison"
        assert r.needs_clarification is True  # no player names extracted

    def test_which_rb_is_better(self):
        _check("Which RB is better right now, Derrick Henry or Christian McCaffrey?", "player_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# betting_trend  (7 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestBettingTrend:
    def test_do_underdogs_cover(self):
        _check("How often do underdogs cover the spread?", "betting_trend")

    def test_home_favorite_cover(self):
        _check("What is the cover rate for home favorites?", "betting_trend")

    def test_eagles_cover_rate(self):
        r = _check("How often do the Eagles cover?", "betting_trend")
        assert "PHI" in r.extracted_entities["teams"]

    def test_spread_greater_than_7(self):
        _check("What happens when the spread is greater than 7 points?", "betting_trend")

    def test_teams_go_over(self):
        _check("What teams go over the total most often?", "betting_trend")

    def test_ats_record(self):
        _check("What is Kansas City's ATS record this season?", "betting_trend")

    def test_road_underdog_trend(self):
        _check("Do road underdogs cover more than home underdogs?", "betting_trend")


# ══════════════════════════════════════════════════════════════════════════════
# team_summary  (4 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestTeamSummary:
    def test_how_is_team_doing(self):
        r = _check("How are the Chiefs doing this season?", "team_summary")
        assert "KC" in r.extracted_entities["teams"]

    def test_team_stats(self):
        r = _check("Show me the 49ers team stats", "team_summary")
        assert "SF" in r.extracted_entities["teams"]

    def test_win_loss_record(self):
        _check("What is the Eagles' win-loss record this year?", "team_summary")

    def test_team_summary_missing_team(self):
        r = classify_intent("How is the team doing this season?")
        assert r.intent == "team_summary"
        assert r.needs_clarification is True  # no team named


# ══════════════════════════════════════════════════════════════════════════════
# model_explanation  (5 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestModelExplanation:
    def test_why_did_you_pick(self):
        r = _check("Why did you pick the Chiefs to win?", "model_explanation")
        assert "KC" in r.extracted_entities["teams"]

    def test_explain_prediction(self):
        _check("Explain your prediction for this game", "model_explanation")

    def test_what_factors(self):
        _check("What factors influenced the model's prediction?", "model_explanation")

    def test_how_confident(self):
        _check("How confident are you in this prediction?", "model_explanation")

    def test_how_does_model_work(self):
        _check("How does the model decide who wins?", "model_explanation")


# ══════════════════════════════════════════════════════════════════════════════
# data_question  (4 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQuestion:
    def test_injury_report(self):
        r = _check("Who is injured on the Eagles this week?", "data_question")
        assert "PHI" in r.extracted_entities["teams"]

    def test_standings(self):
        _check("What are the current NFL standings?", "data_question")

    def test_final_score(self):
        _check("What was the final score of the Chiefs-Ravens game?", "data_question")

    def test_when_do_they_play(self):
        r = _check("When do the Bengals play next?", "data_question")
        assert "CIN" in r.extracted_entities["teams"]


# ══════════════════════════════════════════════════════════════════════════════
# general_help  (3 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneralHelp:
    def test_what_can_you_do(self):
        _check("What can you do?", "general_help")

    def test_help_alone(self):
        _check("help", "general_help")

    def test_what_questions_can_i_ask(self):
        _check("What questions can I ask you about the NFL?", "general_help")


# ══════════════════════════════════════════════════════════════════════════════
# unknown  (3 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestUnknown:
    def test_empty_string(self):
        r = classify_intent("")
        assert r.intent == "unknown"
        assert r.needs_clarification is True

    def test_gibberish(self):
        r = classify_intent("asdfghjkl qwerty zxcv")
        assert r.intent == "unknown"
        assert r.needs_clarification is True

    def test_unrelated_topic(self):
        r = classify_intent("What is the weather like in Denver?")
        assert r.intent == "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Entity extraction  (unit tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestEntityExtraction:
    def test_teams_abbreviations(self):
        teams = _extract_teams("KC vs BUF")
        assert "KC" in teams
        assert "BUF" in teams

    def test_teams_full_names(self):
        teams = _extract_teams("Kansas City Chiefs vs Philadelphia Eagles")
        assert "KC" in teams
        assert "PHI" in teams

    def test_teams_nicknames(self):
        teams = _extract_teams("Do the Patriots cover against the Chiefs?")
        assert "NE" in teams
        assert "KC" in teams

    def test_season_year(self):
        ents = _extract_entities("How did the Chiefs do in the 2023 season?")
        assert ents["season"] == 2023

    def test_week_number(self):
        ents = _extract_entities("Who wins in week 14?")
        assert ents["week"] == 14

    def test_spread_value(self):
        ents = _extract_entities("The Chiefs are 6.5-point favorites")
        assert ents["spread"] == 6.5

    def test_no_entities(self):
        ents = _extract_entities("Who will win?")
        assert ents["season"] is None
        assert ents["week"] is None
        assert ents["spread"] is None

    def test_no_duplicate_teams(self):
        teams = _extract_teams("Will the Chiefs beat the Chiefs?")
        assert teams.count("KC") == 1


# ══════════════════════════════════════════════════════════════════════════════
# IntentResult dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentResult:
    def test_valid_intent(self):
        r = IntentResult(intent="matchup_prediction", confidence=0.8)
        assert r.intent == "matchup_prediction"
        assert r.needs_clarification is False

    def test_invalid_intent_raises(self):
        with pytest.raises(ValueError, match="Unknown intent"):
            IntentResult(intent="nonsense_intent", confidence=0.5)

    def test_default_entities(self):
        r = IntentResult(intent="unknown", confidence=0.1)
        assert r.extracted_entities == {}

    def test_needs_clarification_unknown(self):
        r = classify_intent("")
        assert r.needs_clarification is True

    def test_needs_clarification_matchup_no_teams(self):
        r = classify_intent("Who will win the game?")
        assert r.intent == "matchup_prediction"
        assert r.needs_clarification is True

    def test_confidence_range(self):
        for q in [
            "Who will win Chiefs vs Bills?",
            "How often do underdogs cover?",
            "What can you do?",
            "help",
        ]:
            r = classify_intent(q)
            assert 0.0 <= r.confidence <= 1.0, f"Confidence out of range for {q!r}: {r.confidence}"
