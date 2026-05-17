"""
Tests for the scraping sub-package.

Coverage:
  - RobotsChecker: allow / disallow / 404 / network error / crawl-delay
  - NflScraper: robots enforcement, caching, 403/429 blocking, force_refresh
  - parse_injury_table: full sample, empty HTML, missing columns, status norms
  - records_to_dataframe: column presence, empty-list edge case
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nfl_chatbot.scraping.injury_scraper import (
    InjuryRecord,
    parse_injury_html_file,
    parse_injury_table,
    records_to_dataframe,
)
from nfl_chatbot.scraping.robots import RobotsChecker
from nfl_chatbot.scraping.scraper import (
    NflScraper,
    ScraperConfig,
    ScrapingBlockedError,
    ScrapingNotAllowedError,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_HTML = FIXTURES / "injury_report_sample.html"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def _robots_txt(disallowed_paths: list[str] | None = None, crawl_delay: int | None = None) -> str:
    lines = ["User-agent: *"]
    for path in (disallowed_paths or []):
        lines.append(f"Disallow: {path}")
    if crawl_delay is not None:
        lines.append(f"Crawl-delay: {crawl_delay}")
    return "\n".join(lines)


# ── RobotsChecker ──────────────────────────────────────────────────────────────


class TestRobotsChecker:
    def test_allows_when_robots_permits(self):
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _robots_txt())
            assert checker.can_fetch("https://example.com/page") is True

    def test_disallows_when_robots_blocks_path(self):
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _robots_txt(["/blocked/"]))
            assert checker.can_fetch("https://example.com/blocked/data") is False

    def test_allows_when_robots_404(self):
        """No robots.txt means no restrictions."""
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404)
            assert checker.can_fetch("https://example.com/anything") is True

    def test_allows_when_network_error(self):
        """Unreachable robots.txt → fail open per RFC 9309."""
        import requests as req_lib
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.side_effect = req_lib.ConnectionError("timeout")
            assert checker.can_fetch("https://example.com/page") is True

    def test_robots_fetched_once_per_domain(self):
        """robots.txt should be fetched only once per domain across multiple calls."""
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _robots_txt())
            checker.can_fetch("https://example.com/a")
            checker.can_fetch("https://example.com/b")
            checker.can_fetch("https://example.com/c")
            assert mock_get.call_count == 1

    def test_crawl_delay_returned(self):
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _robots_txt(crawl_delay=5))
            delay = checker.crawl_delay("https://example.com/page")
            assert delay == 5.0

    def test_crawl_delay_zero_when_not_set(self):
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _robots_txt())
            assert checker.crawl_delay("https://example.com/page") == 0.0

    def test_clear_cache_forces_refetch(self):
        checker = RobotsChecker()
        with patch("nfl_chatbot.scraping.robots.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, _robots_txt())
            checker.can_fetch("https://example.com/page")
            checker.clear_cache()
            checker.can_fetch("https://example.com/page")
            assert mock_get.call_count == 2


# ── NflScraper ─────────────────────────────────────────────────────────────────


class TestNflScraper:
    def _scraper(self, tmp_path: Path) -> NflScraper:
        cfg = ScraperConfig(
            cache_dir=tmp_path / "cache",
            rate_limit_seconds=0.0,
            max_retries=2,
        )
        return NflScraper(config=cfg)

    def test_fetch_returns_html(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = True
        robots_mock.crawl_delay.return_value = 0.0
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(200, "<html>ok</html>")
            html = scraper.fetch("https://example.com/page")

        assert html == "<html>ok</html>"

    def test_robots_disallow_raises(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = False
        scraper._robots = robots_mock

        with pytest.raises(ScrapingNotAllowedError, match="robots.txt"):
            scraper.fetch("https://example.com/blocked")

    def test_http_403_raises_blocked_error(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = True
        robots_mock.crawl_delay.return_value = 0.0
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(403)
            with pytest.raises(ScrapingBlockedError, match="403"):
                scraper.fetch("https://example.com/page")

    def test_http_429_raises_blocked_error(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = True
        robots_mock.crawl_delay.return_value = 0.0
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(429)
            with pytest.raises(ScrapingBlockedError, match="429"):
                scraper.fetch("https://example.com/page")

    def test_response_cached_on_disk(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = True
        robots_mock.crawl_delay.return_value = 0.0
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(200, "<html>data</html>")
            scraper.fetch("https://example.com/page")

        assert any((tmp_path / "cache").glob("*.html"))

    def test_second_fetch_uses_cache(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = True
        robots_mock.crawl_delay.return_value = 0.0
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(200, "<html>data</html>")
            scraper.fetch("https://example.com/page")
            scraper.fetch("https://example.com/page")
            # HTTP should only be called once — second call hits cache
            assert mock_get.call_count == 1

    def test_force_refresh_bypasses_cache(self, tmp_path):
        scraper = self._scraper(tmp_path)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = True
        robots_mock.crawl_delay.return_value = 0.0
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(200, "<html>data</html>")
            scraper.fetch("https://example.com/page")
            scraper.fetch("https://example.com/page", force_refresh=True)
            assert mock_get.call_count == 2

    def test_context_manager_closes_session(self, tmp_path):
        cfg = ScraperConfig(cache_dir=tmp_path / "cache", rate_limit_seconds=0.0)
        with NflScraper(config=cfg) as scraper:
            pass  # just verify __exit__ doesn't raise

    def test_respect_robots_false_skips_check(self, tmp_path):
        cfg = ScraperConfig(
            cache_dir=tmp_path / "cache",
            rate_limit_seconds=0.0,
            respect_robots=False,
        )
        scraper = NflScraper(config=cfg)
        robots_mock = MagicMock()
        robots_mock.can_fetch.return_value = False  # would block if checked
        scraper._robots = robots_mock

        with patch.object(scraper._session, "get") as mock_get:
            mock_get.return_value = _mock_response(200, "<html>ok</html>")
            html = scraper.fetch("https://example.com/page")

        robots_mock.can_fetch.assert_not_called()
        assert html == "<html>ok</html>"


# ── parse_injury_table ─────────────────────────────────────────────────────────


class TestParseInjuryTable:
    def test_parses_sample_fixture(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        assert len(records) == 14  # 7 KC/BUF + 7 PHI/DAL rows

    def test_returns_injury_record_instances(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        assert all(isinstance(r, InjuryRecord) for r in records)

    def test_player_names_parsed(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        names = {r.player_name for r in records}
        assert "Patrick Mahomes" in names
        assert "CeeDee Lamb" in names
        assert "Jalen Hurts" in names

    def test_teams_uppercased(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        teams = {r.team for r in records}
        assert teams == {"KC", "BUF", "PHI", "DAL"}

    def test_game_status_out(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        out_players = [r for r in records if r.game_status == "Out"]
        names = {r.player_name for r in out_players}
        assert "Stefon Diggs" in names
        assert "CeeDee Lamb" in names

    def test_game_status_questionable(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        q_players = [r for r in records if r.game_status == "Questionable"]
        names = {r.player_name for r in q_players}
        assert "Travis Kelce" in names
        assert "Dak Prescott" in names

    def test_game_status_none_for_cleared_players(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        cleared = [r for r in records if r.game_status is None]
        names = {r.player_name for r in cleared}
        assert "Josh Allen" in names
        assert "Micah Parsons" in names

    def test_practice_status_normalised_fp(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        mahomes = next(r for r in records if r.player_name == "Patrick Mahomes")
        assert mahomes.practice_fri == "FP"

    def test_practice_status_normalised_dnp(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        diggs = next(r for r in records if r.player_name == "Stefon Diggs")
        assert diggs.practice_wed == "DNP"
        assert diggs.practice_thu == "DNP"
        assert diggs.practice_fri == "DNP"

    def test_injury_type_parsed(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        kelce = next(r for r in records if r.player_name == "Travis Kelce")
        assert kelce.injury == "Knee"

    def test_position_parsed(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        mahomes = next(r for r in records if r.player_name == "Patrick Mahomes")
        assert mahomes.position == "QB"

    def test_irrelevant_table_ignored(self):
        """The 'Schedule Summary' table has no player/team columns and should be skipped."""
        records = parse_injury_html_file(SAMPLE_HTML)
        # All records should be players, not schedule rows
        for r in records:
            assert r.player_name not in ("14", "16")

    def test_empty_html_returns_empty_list(self):
        assert parse_injury_table("") == []

    def test_no_table_returns_empty_list(self):
        assert parse_injury_table("<html><body><p>No tables here</p></body></html>") == []

    def test_table_without_player_column_ignored(self):
        html = """
        <table>
          <tr><th>Date</th><th>Venue</th><th>Attendance</th></tr>
          <tr><td>2024-12-06</td><td>Arrowhead</td><td>76,000</td></tr>
        </table>
        """
        assert parse_injury_table(html) == []

    def test_handles_missing_optional_columns(self):
        """Table with only Player, Team, Status — no practice columns."""
        html = """
        <table>
          <tr><th>Player</th><th>Team</th><th>Injury</th><th>Status</th></tr>
          <tr><td>Test Player</td><td>KC</td><td>Back</td><td>Questionable</td></tr>
        </table>
        """
        records = parse_injury_table(html)
        assert len(records) == 1
        assert records[0].player_name == "Test Player"
        assert records[0].team == "KC"
        assert records[0].game_status == "Questionable"
        assert records[0].practice_wed is None
        assert records[0].practice_thu is None
        assert records[0].practice_fri is None

    def test_blank_status_stored_as_none(self):
        html = """
        <table>
          <tr><th>Player</th><th>Team</th><th>Status</th></tr>
          <tr><td>Josh Allen</td><td>BUF</td><td></td></tr>
        </table>
        """
        records = parse_injury_table(html)
        assert records[0].game_status is None

    def test_dash_status_stored_as_none(self):
        html = """
        <table>
          <tr><th>Player</th><th>Team</th><th>Status</th></tr>
          <tr><td>Josh Allen</td><td>BUF</td><td>-</td></tr>
        </table>
        """
        records = parse_injury_table(html)
        assert records[0].game_status is None

    def test_multiple_tables_aggregated(self):
        """parse_injury_table should collect rows from all matching tables."""
        html = """
        <table>
          <tr><th>Player</th><th>Team</th><th>Status</th></tr>
          <tr><td>Player A</td><td>KC</td><td>Out</td></tr>
        </table>
        <table>
          <tr><th>Player</th><th>Team</th><th>Status</th></tr>
          <tr><td>Player B</td><td>BUF</td><td>Questionable</td></tr>
        </table>
        """
        records = parse_injury_table(html)
        assert len(records) == 2
        assert {r.player_name for r in records} == {"Player A", "Player B"}

    def test_row_missing_player_skipped(self):
        html = """
        <table>
          <tr><th>Player</th><th>Team</th><th>Status</th></tr>
          <tr><td></td><td>KC</td><td>Out</td></tr>
          <tr><td>Valid Player</td><td>KC</td><td>Questionable</td></tr>
        </table>
        """
        records = parse_injury_table(html)
        assert len(records) == 1
        assert records[0].player_name == "Valid Player"


# ── Practice status normalisation ─────────────────────────────────────────────


class TestNormalisePracticeStatus:
    """Verify _normalise_practice handles all expected raw values."""

    def _parse_practice(self, raw: str) -> str | None:
        html = f"""
        <table>
          <tr><th>Player</th><th>Team</th><th>Wed</th></tr>
          <tr><td>P</td><td>KC</td><td>{raw}</td></tr>
        </table>
        """
        return parse_injury_table(html)[0].practice_wed

    def test_fp_variants(self):
        for val in ("FP", "fp", "Full", "Full Participation", "FULL"):
            assert self._parse_practice(val) == "FP", f"Failed for {val!r}"

    def test_lp_variants(self):
        for val in ("LP", "lp", "Limited", "Limited Participation", "LIMITED"):
            assert self._parse_practice(val) == "LP", f"Failed for {val!r}"

    def test_dnp_variants(self):
        for val in ("DNP", "dnp", "Did Not Participate", "No Practice"):
            assert self._parse_practice(val) == "DNP", f"Failed for {val!r}"

    def test_empty_becomes_none(self):
        assert self._parse_practice("") is None
        assert self._parse_practice("-") is None


# ── records_to_dataframe ───────────────────────────────────────────────────────


class TestRecordsToDataframe:
    def test_empty_list_returns_empty_dataframe(self):
        df = records_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "player_name" in df.columns
        assert "game_status" in df.columns

    def test_dataframe_has_all_expected_columns(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        df = records_to_dataframe(records)
        expected = {
            "player_name", "team", "position", "injury",
            "practice_wed", "practice_thu", "practice_fri", "game_status",
        }
        assert expected.issubset(set(df.columns))

    def test_dataframe_row_count_matches_records(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        df = records_to_dataframe(records)
        assert len(df) == len(records)

    def test_dataframe_values_correct(self):
        records = parse_injury_html_file(SAMPLE_HTML)
        df = records_to_dataframe(records)
        mahomes_row = df[df["player_name"] == "Patrick Mahomes"].iloc[0]
        assert mahomes_row["team"] == "KC"
        assert mahomes_row["position"] == "QB"
        assert mahomes_row["injury"] == "Ankle"
        assert mahomes_row["game_status"] == "Probable"
