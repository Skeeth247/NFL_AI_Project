"""
NFL injury-report HTML parser.

Parses standard NFL weekly injury-report HTML tables into structured records.
This module only parses HTML strings — it never makes HTTP requests directly.
To fetch live data, obtain HTML via NflScraper.fetch() first and pass it here.

Supported table format (column headers are matched case-insensitively):
    Player | Pos | Team | Injury | Wed | Thu | Fri | Status

Any subset of these columns is accepted; missing fields default to None.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class InjuryRecord:
    """One player row from an NFL injury-report table."""

    player_name: str
    team: str
    position: str | None = None
    injury: str | None = None
    practice_wed: str | None = None      # FP | LP | DNP | None
    practice_thu: str | None = None
    practice_fri: str | None = None
    game_status: str | None = None       # Out | Doubtful | Questionable | Probable | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Column-header aliases ──────────────────────────────────────────────────────
# Keys are substrings matched case-insensitively against raw header text.
# Longest / most-specific keys should appear first in _HEADER_MAP so that
# "game status" matches before the bare "status" fallback.

_HEADER_MAP: list[tuple[str, str]] = [
    ("game status", "game_status"),
    ("designation", "game_status"),
    ("status", "game_status"),
    ("player", "player_name"),
    ("name", "player_name"),
    ("pos", "position"),
    ("team", "team"),
    ("injury", "injury"),
    ("type", "injury"),
    ("wed", "practice_wed"),
    ("thu", "practice_thu"),
    ("fri", "practice_fri"),
]

# Canonical practice-participation codes
_PRACTICE_FULL = frozenset({"fp", "full", "full participation"})
_PRACTICE_LIMITED = frozenset({"lp", "limited", "limited participation"})
_PRACTICE_DNP = frozenset({"dnp", "did not participate", "no practice"})
_EMPTY_VALUES = frozenset({"", "-", "—", "n/a", "none"})


# ── Public API ─────────────────────────────────────────────────────────────────


def parse_injury_table(html: str) -> list[InjuryRecord]:
    """
    Parse every injury-report table found in *html*.

    Scans all <table> elements and returns a flat list of InjuryRecords from
    any table whose headers include a player column and a team column.
    Returns an empty list (never raises) if no matching table is found.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[InjuryRecord] = []

    for table in soup.find_all("table"):
        if isinstance(table, Tag):
            records.extend(_parse_single_table(table))

    logger.info("parse_injury_table: extracted %d records", len(records))
    return records


def parse_injury_html_file(path: Path) -> list[InjuryRecord]:
    """Load HTML from a local file and return parsed InjuryRecords."""
    return parse_injury_table(path.read_text(encoding="utf-8"))


def records_to_dataframe(records: list[InjuryRecord]) -> pd.DataFrame:
    """Convert a list of InjuryRecords to a tidy DataFrame."""
    _COLS = [
        "player_name", "team", "position", "injury",
        "practice_wed", "practice_thu", "practice_fri", "game_status",
    ]
    if not records:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame([r.to_dict() for r in records])[_COLS]


# ── Internal helpers ───────────────────────────────────────────────────────────


def _parse_single_table(table: Tag) -> list[InjuryRecord]:
    """Return records from *table* if it looks like an injury report, else []."""
    col_map = _extract_header_map(table)
    if not col_map:
        return []

    field_names = set(col_map.values())
    if "player_name" not in field_names or "team" not in field_names:
        return []

    records: list[InjuryRecord] = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        # Skip empty rows and header rows (all-<th>)
        if not cells or all(c.name == "th" for c in cells):
            continue
        rec = _parse_row(cells, col_map)
        if rec is not None:
            records.append(rec)

    return records


def _extract_header_map(table: Tag) -> dict[int, str]:
    """
    Map column index → normalised field name from the table's first <tr>.
    Returns {} if no recognisable injury-report headers are found.
    """
    header_row = table.find("tr")
    if not isinstance(header_row, Tag):
        return {}

    result: dict[int, str] = {}
    for idx, cell in enumerate(header_row.find_all(["th", "td"])):
        text = cell.get_text(strip=True).lower()
        for alias, field in _HEADER_MAP:
            if alias in text:
                result[idx] = field
                break
    return result


def _parse_row(cells: list[Tag], col_map: dict[int, str]) -> InjuryRecord | None:
    """Build one InjuryRecord from a row's cells using the column map."""
    raw: dict[str, str] = {}
    for idx, cell in enumerate(cells):
        field = col_map.get(idx)
        if field:
            raw[field] = cell.get_text(strip=True)

    player = raw.get("player_name", "").strip()
    team = raw.get("team", "").strip().upper()
    if not player or not team:
        return None

    return InjuryRecord(
        player_name=player,
        team=team,
        position=_clean_optional(raw.get("position")),
        injury=_clean_optional(raw.get("injury")),
        practice_wed=_normalise_practice(raw.get("practice_wed")),
        practice_thu=_normalise_practice(raw.get("practice_thu")),
        practice_fri=_normalise_practice(raw.get("practice_fri")),
        game_status=_normalise_status(raw.get("game_status")),
    )


def _clean_optional(value: str | None) -> str | None:
    """Return stripped value or None for blank / dash strings."""
    if not value:
        return None
    v = value.strip()
    return None if v.lower() in _EMPTY_VALUES else v


def _normalise_practice(value: str | None) -> str | None:
    """Normalise raw practice text to FP / LP / DNP / None."""
    if not value:
        return None
    v = value.strip()
    lower = v.lower()
    if lower in _EMPTY_VALUES:
        return None
    if lower in _PRACTICE_FULL:
        return "FP"
    if lower in _PRACTICE_LIMITED:
        return "LP"
    if lower in _PRACTICE_DNP:
        return "DNP"
    # Return as-is for unrecognised values (e.g. "IR", future formats)
    return v.upper()


def _normalise_status(value: str | None) -> str | None:
    """Normalise raw game-status text; return None for 'cleared / no designation'."""
    if not value:
        return None
    v = value.strip()
    if v.lower() in _EMPTY_VALUES:
        return None
    return v
