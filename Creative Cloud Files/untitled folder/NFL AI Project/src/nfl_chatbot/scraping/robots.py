"""
robots.txt compliance checker.

Fetches and caches robots.txt for each domain, then answers can_fetch()
queries before any HTTP request is made. Follows RFC 9309:
- 404 → no restrictions (all paths allowed)
- Network error → fail open (assume allowed), log warning
- Never silently bypass a Disallow directive
"""

from __future__ import annotations

import logging
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "NFL-AI-Research-Bot/1.0 (educational project; not for commercial use)"


class RobotsChecker:
    """
    Per-domain robots.txt checker.  One instance is shared across a scraping session
    so each domain's robots.txt is fetched at most once.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 10,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def can_fetch(self, url: str) -> bool:
        """
        Return True if this scraper's user-agent is allowed to fetch *url*.

        Always returns True when robots.txt is unreachable (fail-open per RFC 9309).
        Logs a WARNING when a Disallow rule blocks the URL.
        """
        parser = self._get_parser(url)
        allowed: bool = parser.can_fetch(self.user_agent, url)
        if not allowed:
            logger.warning(
                "robots.txt disallows %r for user-agent %r — request will be blocked",
                url,
                self.user_agent,
            )
        return allowed

    def crawl_delay(self, url: str) -> float:
        """
        Return the Crawl-delay (seconds) for this user-agent on *url*'s domain.
        Returns 0.0 when no directive is set.
        """
        parser = self._get_parser(url)
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay is not None else 0.0

    def clear_cache(self) -> None:
        """Force re-fetch of all cached robots.txt on next access."""
        self._parsers.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _origin(self, url: str) -> str:
        """Return scheme://host for *url*."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        """Return a cached (or freshly fetched) RobotFileParser for *url*'s domain."""
        key = self._origin(url)
        if key not in self._parsers:
            self._parsers[key] = self._fetch_robots(key)
        return self._parsers[key]

    def _fetch_robots(self, origin: str) -> urllib.robotparser.RobotFileParser:
        """Fetch robots.txt from *origin* and return a populated parser."""
        robots_url = urljoin(origin + "/", "robots.txt")
        parser = urllib.robotparser.RobotFileParser(robots_url)

        try:
            resp = requests.get(
                robots_url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
                logger.info("Loaded robots.txt from %s", robots_url)
            elif resp.status_code == 404:
                # RFC 9309 §2.3.1.3: treat missing robots.txt as no restrictions
                parser.parse([])
                logger.debug("No robots.txt at %s (404) — all paths assumed allowed", robots_url)
            else:
                # Non-200/404: treat as inaccessible → fail open
                parser.parse([])
                logger.warning(
                    "robots.txt at %s returned HTTP %d — assuming all paths allowed",
                    robots_url,
                    resp.status_code,
                )
        except requests.RequestException as exc:
            # Network failure: fail open per RFC 9309 §2.3.1.3
            parser.parse([])
            logger.warning(
                "Could not fetch robots.txt from %s (%s) — assuming all paths allowed",
                robots_url,
                exc,
            )

        return parser
