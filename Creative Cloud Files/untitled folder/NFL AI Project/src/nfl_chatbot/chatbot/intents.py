"""
Rule-based intent classification for the NFL AI Chatbot.

Classifies a free-text user question into one of eight intents without
requiring an LLM. Classification is done by weighted keyword/pattern
matching followed by entity extraction. The result is a structured
``IntentResult`` that the chatbot's tool-dispatch layer consumes.

Intents
-------
  matchup_prediction  — predict who wins a game (moneyline / SU)
  player_comparison   — compare two or more players
  betting_trend       — ATS cover rates, over/under trends
  team_summary        — season record, stats, form for a team
  model_explanation   — why the model predicted X; confidence; factors
  data_question       — factual lookups (scores, injuries, standings)
  general_help        — "what can you do?", how-to questions
  unknown             — everything else; triggers a clarification prompt

Classification algorithm
------------------------
1. Normalize the text (lowercase, collapse whitespace).
2. Score every intent by summing the weights of all matching regex patterns.
3. Apply a small context penalty: questions starting with "why" or "explain"
   shift weight toward model_explanation.
4. Pick the intent with the highest score.  Score < 1.0 → ``unknown``.
5. Confidence is calibrated via  score / (score + 2)  (returns 0.33 – 1.0).
6. ``needs_clarification`` is True when:
     - intent is unknown
     - confidence < 0.45
     - matchup_prediction but no teams found
     - player_comparison but fewer than 2 player entities found

Entity extraction
-----------------
``extracted_entities`` always has these keys (values may be empty / None):
  teams     : list[str]   — nflverse 2–3 letter abbreviations (e.g. ["KC","BUF"])
  players   : list[str]   — raw player name strings (title-cased)
  season    : int | None  — four-digit year if mentioned
  week      : int | None  — NFL week number if mentioned
  spread    : float | None — numeric spread value (+3.5, -7, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Intent list
# ══════════════════════════════════════════════════════════════════════════════

INTENTS = (
    "matchup_prediction",
    "player_comparison",
    "betting_trend",
    "team_summary",
    "model_explanation",
    "data_question",
    "general_help",
    "unknown",
)


# ══════════════════════════════════════════════════════════════════════════════
# NFL team registry
# ══════════════════════════════════════════════════════════════════════════════

# Maps every recognisable alias → canonical nflverse abbreviation.
# Keys are lowercase for case-insensitive matching.
_TEAM_ALIASES: dict[str, str] = {
    # AFC East
    "buffalo bills": "BUF", "buffalo": "BUF", "bills": "BUF", "buf": "BUF",
    "miami dolphins": "MIA", "miami": "MIA", "dolphins": "MIA", "mia": "MIA",
    "new england patriots": "NE", "new england": "NE", "patriots": "NE", "pats": "NE", "ne": "NE",
    "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    # AFC North
    "baltimore ravens": "BAL", "baltimore": "BAL", "ravens": "BAL", "bal": "BAL",
    "cleveland browns": "CLE", "cleveland": "CLE", "browns": "CLE", "cle": "CLE",
    "cincinnati bengals": "CIN", "cincinnati": "CIN", "bengals": "CIN", "cin": "CIN",
    "pittsburgh steelers": "PIT", "pittsburgh": "PIT", "steelers": "PIT", "pit": "PIT",
    # AFC South
    "houston texans": "HOU", "houston": "HOU", "texans": "HOU", "hou": "HOU",
    "indianapolis colts": "IND", "indianapolis": "IND", "colts": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jacksonville": "JAX", "jaguars": "JAX", "jags": "JAX", "jax": "JAX",
    "tennessee titans": "TEN", "tennessee": "TEN", "titans": "TEN", "ten": "TEN",
    # AFC West
    "denver broncos": "DEN", "denver": "DEN", "broncos": "DEN", "den": "DEN",
    "kansas city chiefs": "KC", "kansas city": "KC", "chiefs": "KC", "kc": "KC",
    "las vegas raiders": "LV", "las vegas": "LV", "raiders": "LV", "lv": "LV", "oak": "LV",
    "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    # NFC East
    "dallas cowboys": "DAL", "dallas": "DAL", "cowboys": "DAL", "dal": "DAL",
    "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "philadelphia eagles": "PHI", "philadelphia": "PHI", "eagles": "PHI", "phi": "PHI",
    "washington commanders": "WAS", "washington": "WAS", "commanders": "WAS",
    "washington football team": "WAS", "was": "WAS", "wsh": "WAS",
    # NFC North
    "chicago bears": "CHI", "chicago": "CHI", "bears": "CHI", "chi": "CHI",
    "detroit lions": "DET", "detroit": "DET", "lions": "DET", "det": "DET",
    "green bay packers": "GB", "green bay": "GB", "packers": "GB", "gb": "GB",
    "minnesota vikings": "MIN", "minnesota": "MIN", "vikings": "MIN", "min": "MIN",
    # NFC South
    "atlanta falcons": "ATL", "atlanta": "ATL", "falcons": "ATL", "atl": "ATL",
    "carolina panthers": "CAR", "carolina": "CAR", "panthers": "CAR", "car": "CAR",
    "new orleans saints": "NO", "new orleans": "NO", "saints": "NO", "no": "NO",
    "tampa bay buccaneers": "TB", "tampa bay": "TB", "buccaneers": "TB", "bucs": "TB", "tb": "TB",
    # NFC West
    "arizona cardinals": "ARI", "arizona": "ARI", "cardinals": "ARI", "cards": "ARI", "ari": "ARI",
    "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR", "lar": "LAR",
    "seattle seahawks": "SEA", "seattle": "SEA", "seahawks": "SEA", "hawks": "SEA", "sea": "SEA",
    "san francisco 49ers": "SF", "san francisco": "SF", "49ers": "SF", "niners": "SF", "sf": "SF",
}

# Pre-sorted longest first so multi-word aliases match before single words
_TEAM_ALIAS_SORTED: list[tuple[str, str]] = sorted(
    _TEAM_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True
)


# ══════════════════════════════════════════════════════════════════════════════
# Compiled regex patterns per intent
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: (compiled_pattern, weight)

def _p(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(pattern, flags)


_INTENT_PATTERNS: dict[str, list[tuple[re.Pattern, float]]] = {

    "matchup_prediction": [
        (_p(r"who\s+will\s+win"), 3.5),
        (_p(r"will\s+\w+\s+beat\s+\w+"), 3.5),
        (_p(r"who\s+wins"), 3.0),
        (_p(r"\bbeat\b.{0,30}\b(in|this|the|next|on)\b"), 2.0),
        (_p(r"\bpredic\w+\s+\w+\s+(game|matchup|win|victory)"), 2.5),
        (_p(r"\b(chance|probability|odds)\s+(of|that|the)\s+\w+\s+(win|beat|cover|win)"), 2.0),
        (_p(r"\bfavor(ed|ite)?\b.{0,25}\b(win|game|matchup)"), 1.5),
        (_p(r"\bmoneyline\b"), 2.5),
        (_p(r"\bstraight.?up\b"), 2.0),
        (_p(r"\bwho\s+(do\s+you|would\s+you)\s+pick\b"), 3.0),
        (_p(r"\bupset\b"), 2.5),
        (_p(r"\bchances?\s+of\s+an?\s+upset\b"), 3.0),
        (_p(r"\bgame\s+prediction\b"), 3.0),
        (_p(r"\bpredict.*matchup\b"), 2.5),
        (_p(r"\bwin.*this\s+(week|sunday|game)\b"), 2.0),
    ],

    "player_comparison": [
        (_p(r"\bcompar\w+\b.{0,40}\band\b"), 3.5),
        (_p(r"\bwho\s+is\s+better\b"), 3.5),
        (_p(r"\bwho\s+has\s+(more|better|higher|fewer|lower)\b"), 3.0),
        (_p(r"\bhead.?to.?head\b"), 3.0),
        (_p(r"\bversus\b|\bvs\.?\b.{1,40}(stats|performance|season|yards|touchdowns|tds)"), 2.5),
        (_p(r"\bside.?by.?side\b"), 2.5),
        (_p(r"\bcompar\w+\s+\w+\s+and\s+\w+"), 3.0),
        (_p(r"\b(stats|numbers)\s+(for|of)\s+\w+\s+(and|vs\.?|versus)\s+\w+"), 2.5),
        (_p(r"\bwhich\s+(player|qb|rb|wr|te)\s+is\b"), 2.5),
        (_p(r"\bbetter\s+(qb|rb|wr|player)\b"), 2.5),
        (_p(r"\b(passing|rushing|receiving)\s+(yards|stats).{0,30}(than|vs|versus|compared)"), 2.0),
    ],

    "betting_trend": [
        (_p(r"\bcover\b"), 3.5),
        (_p(r"\bats\b"), 3.5),
        (_p(r"\bover.?under\b"), 3.0),
        (_p(r"\bo/?u\b"), 2.0),
        (_p(r"\bspread.{0,15}trend"), 2.5),
        (_p(r"\bunderdog\s+(cover|win|beat|record|rate|trend)"), 3.0),
        (_p(r"\bhow\s+often\s+(do|does|did)\s+\w+\s+(cover|go\s+over|beat)"), 3.5),
        (_p(r"\bfavorite\s+(cover|record|rate|trend)"), 3.0),
        (_p(r"\bcover\s+rate\b"), 3.5),
        (_p(r"\bover\s+rate\b|\bgo\s+over\b"), 3.0),
        (_p(r"\bhome\s+(team|favorite|underdog)\s+(cover|trend)"), 3.0),
        (_p(r"\broad\s+(team|favorite|underdog)\s+(cover|trend)"), 3.0),
        (_p(r"\bbetting\s+(trend|line|edge)"), 3.0),
        (_p(r"\btotal.{0,15}(over|under|hit)"), 2.5),
        (_p(r"\bspread\b.{0,20}(greater|larger|bigger|more)\s+than\b"), 3.0),
        (_p(r"\bspread\s+(bucket|size|greater|larger|bigger|more\s+than)"), 2.5),
        (_p(r"\bpublic\s+betting\b"), 2.5),
    ],

    "team_summary": [
        (_p(r"\bhow\s+(is|are|has|have)\s+(?:\w+\s+){1,2}(doing|been|performed|playing)"), 3.5),
        (_p(r"\bteam\s+stats?\b"), 3.0),
        (_p(r"\b(season|year)\s+(record|stats?|performance|form)"), 3.0),
        (_p(r"\bwin.?loss\s+record\b"), 3.0),
        (_p(r"\boffensive?\s+(stats?|rank|rating|performance)"), 2.5),
        (_p(r"\bdefensive?\s+(stats?|rank|rating|performance)"), 2.5),
        (_p(r"\bhow\s+(good|bad|strong|weak)\s+(is|are)\s+the\b"), 2.5),
        (_p(r"\bteam\s+(performance|overview|summary|report)"), 3.0),
        (_p(r"\b(current|this\s+season|this\s+year)\s+(form|record|standing)"), 2.5),
        (_p(r"\bstreak\b.{0,20}\b(win|los|game)\b"), 2.0),
        (_p(r"\brecent\s+(form|games|results|performance)\b"), 2.0),
        (_p(r"\bpoints?\s+(per\s+game|scored|allowed)\b.{0,20}\b(this|season|year)\b"), 2.0),
    ],

    "model_explanation": [
        (_p(r"\bwhy\s+(did|does|do|would|is|are)\b"), 3.5),
        (_p(r"\bexplain\b"), 3.5),
        (_p(r"\bwhat\s+factors?\b"), 3.5),
        (_p(r"\breason(s|ing)?\s+(for|behind|why)\b"), 3.0),
        (_p(r"\bhow\s+(confident|sure|certain)\b"), 3.0),
        (_p(r"\bconfidence\s+(level|score|in)\b"), 3.0),
        (_p(r"\bwhy\s+\w+\s+(will|would|is)\s+(win|lose|beat|cover)\b"), 3.5),
        (_p(r"\bwhat\s+(drove|drives|caused|makes)\s+(the\s+)?(model|prediction)"), 3.5),
        (_p(r"\bhow\s+(does|did)\s+the\s+(model|ai|system|algorithm)\b"), 3.5),
        (_p(r"\bwhat\s+does\s+the\s+(model|ai)\s+(think|say|predict)"), 3.0),
        (_p(r"\bkey\s+(factor|reason|driver)\b"), 2.5),
        (_p(r"\bprediction\s+explanation\b"), 3.5),
        (_p(r"\bwhy\s+(your|this|that)\s+(pick|prediction|choice|call)\b"), 3.5),
    ],

    "data_question": [
        (_p(r"\binjur(y|ied|ies)\b"), 3.0),
        (_p(r"\binjury\s+report\b"), 3.5),
        (_p(r"\bwho\s+is\s+(out|injured|hurt|questionable|doubtful|listed)\b"), 3.5),
        (_p(r"\bstandings?\b"), 3.5),
        (_p(r"\bscore\b.{0,20}\b(of|was|is|in)\b"), 3.0),
        (_p(r"\bfinal\s+score\b"), 3.5),
        (_p(r"\blast\s+(game|week|season)\s+(score|result)"), 3.0),
        (_p(r"\bwhat\s+(was|is)\s+the\s+(score|result)\b"), 3.5),
        (_p(r"\bscheduled?\b.{0,30}\b(game|matchup|week)\b"), 2.5),
        (_p(r"\bwhen\s+(do|does|did|is|are)\s+(?:\w+\s+){1,2}(play|playing|next)\b"), 3.0),
        (_p(r"\bschedule\b"), 2.5),
        (_p(r"\broster\b"), 2.5),
        (_p(r"\bplaff?\b|\bplayoff\s+(picture|race|seed)\b"), 2.5),
        (_p(r"\bwild\s+card\s+(race|standing)\b"), 2.5),
        (_p(r"\btraded?\b|\bsign(ed|ing)\b.{0,20}\b(player|quarterback|wide\s+receiver)\b"), 2.0),
    ],

    "general_help": [
        (_p(r"\bhelp\b"), 3.0),
        (_p(r"\bwhat\s+can\s+you\b"), 3.5),
        (_p(r"\bhow\s+(do|can|should)\s+i\s+(use|ask|get)\b"), 3.5),
        (_p(r"\bwhat\s+(do\s+you|can\s+you|are\s+you)\s+(do|know|answer|handle)\b"), 3.5),
        (_p(r"\bwhat\s+questions?\s+(can\s+i|should\s+i|do\s+you)\b"), 3.5),
        (_p(r"\btell\s+me\s+about\s+(your|this|the)\s+(system|chatbot|tool|model|ai)\b"), 3.0),
        (_p(r"\bhow\s+does\s+(this|the)\s+(chatbot|system|app|tool)\s+work\b"), 3.5),
        (_p(r"\bwhat\s+topics?\b"), 2.5),
        (_p(r"\bguide\b|\btutorial\b"), 2.5),
        (_p(r"\bget\s+started\b"), 2.5),
        (_p(r"\bexample\s+(question|query)\b"), 2.5),
    ],
}

# ── Compile all patterns once ─────────────────────────────────────────────────
# (They are already compiled above; this is just a check reference.)
_ALL_INTENT_PATTERNS = _INTENT_PATTERNS  # alias for clarity


# ══════════════════════════════════════════════════════════════════════════════
# Entity extraction
# ══════════════════════════════════════════════════════════════════════════════

# Matches "week N" or "week N" (N = 1-22)
_RE_WEEK = re.compile(r"\bweek\s+(\d{1,2})\b", re.IGNORECASE)

# Matches a 4-digit season year (2000–2029)
_RE_SEASON = re.compile(r"\b(20[0-2]\d)\b")

# Matches a spread value like -6.5, +3, -7, 3.5
_RE_SPREAD = re.compile(r"([+-]?\d{1,2}(?:\.\d)?)[-\s]*(point|pt)s?")

# Words that look like names but aren't (common English words, team words, etc.)
_NON_NAME_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "of", "to", "is", "are", "was",
    "will", "can", "could", "would", "should", "did", "do", "does",
    "and", "or", "but", "if", "when", "where", "who", "what", "how",
    "why", "which", "that", "this", "these", "those", "their", "they",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "game", "team", "season", "week", "year", "player",
    "nfl", "afc", "nfc", "super", "bowl", "playoffs",
    "yes", "no", "not", "more", "less", "most", "best", "worst",
    "last", "next", "first", "second", "third",
})

# Potential player name: two or three Title-Case words not in the non-name set
# We look for them after context anchors to reduce false positives.
_RE_PLAYER_ANCHOR = re.compile(
    r"(?:compare|versus|vs\.?|between|about|for|of|than|and)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z']+){1,2})",
    re.IGNORECASE,
)
# Also catch "Player A vs Player B" without explicit anchor
_RE_PLAYER_FREEFORM = re.compile(
    r"([A-Z][a-z]+\s+[A-Z][a-z']+)\s+(?:vs\.?|versus|and|or|compared\s+to)\s+"
    r"([A-Z][a-z]+\s+[A-Z][a-z']+)",
)


def _extract_teams(text: str) -> list[str]:
    """Return a deduplicated list of team abbreviations mentioned in *text*."""
    lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for alias, abbr in _TEAM_ALIAS_SORTED:
        # Use word-boundary-like matching: alias must not be surrounded by letters
        pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
        if re.search(pattern, lower) and abbr not in seen:
            found.append(abbr)
            seen.add(abbr)
    return found


def _extract_players(text: str) -> list[str]:
    """
    Heuristically extract player name strings from *text*.

    Returns raw strings (not normalised to a canonical ID), e.g.
    ["Patrick Mahomes", "Josh Allen"].
    """
    candidates: list[str] = []

    # Anchor-based extraction
    for m in _RE_PLAYER_ANCHOR.finditer(text):
        name = m.group(1).strip()
        words = name.split()
        if all(w.lower() not in _NON_NAME_WORDS for w in words):
            candidates.append(name)

    # Freeform A vs B
    for m in _RE_PLAYER_FREEFORM.finditer(text):
        for g in (m.group(1), m.group(2)):
            name = g.strip()
            words = name.split()
            if all(w.lower() not in _NON_NAME_WORDS for w in words):
                candidates.append(name)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


def _extract_entities(text: str) -> dict[str, Any]:
    """
    Extract structured entities from a raw question string.

    Returns a dict with keys: teams, players, season, week, spread.
    """
    teams = _extract_teams(text)

    week_m = _RE_WEEK.search(text)
    week = int(week_m.group(1)) if week_m else None

    season_matches = _RE_SEASON.findall(text)
    season = int(season_matches[0]) if season_matches else None

    spread_m = _RE_SPREAD.search(text)
    spread: float | None = None
    if spread_m:
        try:
            spread = float(spread_m.group(1))
        except ValueError:
            pass

    players = _extract_players(text)

    return {
        "teams": teams,
        "players": players,
        "season": season,
        "week": week,
        "spread": spread,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Scoring & calibration
# ══════════════════════════════════════════════════════════════════════════════

def _score_intents(text: str) -> dict[str, float]:
    """Return a raw score for each non-unknown intent."""
    scores: dict[str, float] = {intent: 0.0 for intent in INTENTS if intent != "unknown"}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern, weight in patterns:
            if pattern.search(text):
                scores[intent] += weight
    return scores


def _apply_context_boosts(scores: dict[str, float], text: str) -> dict[str, float]:
    """
    Apply lightweight context-aware adjustments.

    Rules:
    - "why" at the start strongly boosts model_explanation.
    - "cover" boosts betting_trend and penalises matchup_prediction.
    - "compare" boosts player_comparison and penalises others.
    - "injury"/"injured" boosts data_question.
    """
    normed = text.strip().lower()

    if re.match(r"^why\b", normed):
        scores["model_explanation"] += 2.0

    if re.search(r"\bcover\b", normed):
        scores["betting_trend"] += 1.5
        scores["matchup_prediction"] = max(0.0, scores["matchup_prediction"] - 1.0)

    if re.search(r"\bcompar\w+\b", normed):
        scores["player_comparison"] += 1.5
        scores["team_summary"] = max(0.0, scores["team_summary"] - 0.5)

    if re.search(r"\binjur\w+\b", normed):
        scores["data_question"] += 1.5

    if re.search(r"\bhelp\b", normed) and len(normed.split()) <= 5:
        scores["general_help"] += 2.0

    return scores


def _calibrate_confidence(raw_score: float) -> float:
    """Map raw score to a 0–1 confidence value using a logistic-like curve."""
    if raw_score <= 0:
        return 0.1
    return round(raw_score / (raw_score + 2.0), 3)


# ══════════════════════════════════════════════════════════════════════════════
# IntentResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentResult:
    """
    Structured output of the intent classifier.

    Attributes
    ----------
    intent : str
        One of the eight intent labels.
    confidence : float
        0.0 – 1.0.  Values below 0.45 indicate genuine uncertainty.
    extracted_entities : dict
        teams     : list[str]    — nflverse abbreviations
        players   : list[str]    — raw player name strings
        season    : int | None   — year if mentioned
        week      : int | None   — NFL week if mentioned
        spread    : float | None — numeric spread if mentioned
    needs_clarification : bool
        True when the chatbot should ask a follow-up before acting.
    """
    intent: str
    confidence: float
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"Unknown intent: {self.intent!r}. Must be one of {INTENTS}")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(question: str) -> IntentResult:
    """
    Classify a user question into one of eight intents.

    Parameters
    ----------
    question : str
        Raw user input (any case, punctuation allowed).

    Returns
    -------
    IntentResult
        intent, confidence, extracted_entities, needs_clarification.

    Examples
    --------
    >>> r = classify_intent("Who will win Chiefs vs Bills?")
    >>> r.intent
    'matchup_prediction'
    >>> r.extracted_entities['teams']
    ['KC', 'BUF']
    """
    if not question or not question.strip():
        return IntentResult(
            intent="unknown",
            confidence=0.0,
            extracted_entities=_extract_entities(""),
            needs_clarification=True,
        )

    text = question.strip()

    # ── Score ──────────────────────────────────────────────────────────────────
    scores = _score_intents(text)
    scores = _apply_context_boosts(scores, text)

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    # Fall through to unknown if no signal
    if best_score < 1.0:
        best_intent = "unknown"
        confidence = 0.2
    else:
        confidence = _calibrate_confidence(best_score)

    # ── Entities ───────────────────────────────────────────────────────────────
    entities = _extract_entities(text)

    # ── needs_clarification ────────────────────────────────────────────────────
    clarify = False

    if best_intent == "unknown":
        clarify = True

    elif confidence < 0.45:
        clarify = True

    elif best_intent == "matchup_prediction" and len(entities["teams"]) < 2:
        # We can predict, but need to know which teams
        clarify = True

    elif best_intent == "player_comparison" and len(entities["players"]) < 2:
        # Asked to compare but didn't name two players
        clarify = True

    elif best_intent == "team_summary" and not entities["teams"]:
        # "How is the team doing?" — which team?
        clarify = True

    return IntentResult(
        intent=best_intent,
        confidence=confidence,
        extracted_entities=entities,
        needs_clarification=clarify,
    )
