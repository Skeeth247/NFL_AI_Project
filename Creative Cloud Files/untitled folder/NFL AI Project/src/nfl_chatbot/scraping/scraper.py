"""
Base HTTP scraper with robots.txt compliance, rate limiting, and HTML caching.

Usage::

    from nfl_chatbot.scraping.scraper import NflScraper, ScraperConfig

    with NflScraper() as scraper:
        html = scraper.fetch("https://example.com/injury-report")

Design guarantees:
- robots.txt is checked before every fetch (first request per domain only).
- Requests are rate-limited to ≥ rate_limit_seconds between calls.
- HTML responses are cached on disk; stale entries (> cache_ttl_hours) are refreshed.
- 403 / 429 responses raise ScrapingBlockedError immediately (no retry).
- 5xx responses are retried with exponential backoff up to max_retries.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from nfl_chatbot.scraping.robots import DEFAULT_USER_AGENT, RobotsChecker

logger = logging.getLogger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────────────────


class ScrapingNotAllowedError(Exception):
    """
    Raised when robots.txt explicitly disallows the requested URL.
    The caller should log this and skip the URL — never retry or bypass.
    """


class ScrapingBlockedError(Exception):
    """
    Raised when the server actively blocks the request (403, 429, or repeated 5xx).
    Indicates the site does not permit automated access at this time.
    """


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class ScraperConfig:
    """Tunable parameters for NflScraper.  All fields have sensible defaults."""

    user_agent: str = DEFAULT_USER_AGENT
    rate_limit_seconds: float = 3.0          # minimum gap between HTTP requests
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/html"))
    cache_ttl_hours: int = 24                # hours before cached HTML is re-fetched
    timeout_seconds: int = 30
    max_retries: int = 3
    respect_robots: bool = True


# ── Scraper ────────────────────────────────────────────────────────────────────


class NflScraper:
    """
    Thread-safe (single-threaded use) scraper for public NFL data sources.

    Implements a robots.txt check → cache lookup → rate-limited HTTP fetch pipeline.
    """

    def __init__(
        self,
        config: ScraperConfig | None = None,
        robots_checker: RobotsChecker | None = None,
    ) -> None:
        self.cfg = config or ScraperConfig()
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._robots = robots_checker or RobotsChecker(user_agent=self.cfg.user_agent)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = self.cfg.user_agent
        self._last_fetch_time: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(self, url: str, *, force_refresh: bool = False) -> str:
        """
        Return the HTML for *url*, using the local cache when valid.

        Args:
            url:           Full URL to fetch.
            force_refresh: If True, bypass the cache and always make an HTTP request.

        Raises:
            ScrapingNotAllowedError: robots.txt disallows this URL.
            ScrapingBlockedError:    Server returned 403/429 or exhausted retries on 5xx.
        """
        if self.cfg.respect_robots and not self._robots.can_fetch(url):
            raise ScrapingNotAllowedError(
                f"robots.txt disallows {url!r}.  "
                "This path is not available for automated access — "
                "respect the site's terms and do not attempt to bypass this restriction."
            )

        if not force_refresh:
            cached = self._load_cache(url)
            if cached is not None:
                logger.debug("Cache hit for %s", url)
                return cached

        self._apply_rate_limit(url)
        html = self._fetch_with_retries(url)
        self._save_cache(url, html)
        return html

    def close(self) -> None:
        """Close the underlying requests.Session."""
        self._session.close()

    def __enter__(self) -> "NflScraper":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:20]

    def _cache_path(self, url: str) -> Path:
        return self.cfg.cache_dir / f"{self._cache_key(url)}.html"

    def _load_cache(self, url: str) -> str | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cfg.cache_ttl_hours:
            logger.debug("Cache expired (%.1f h old) for %s", age_hours, url)
            return None
        return path.read_text(encoding="utf-8")

    def _save_cache(self, url: str, html: str) -> None:
        path = self._cache_path(url)
        path.write_text(html, encoding="utf-8")
        logger.debug("Cached %s → %s", url, path.name)

    # ── Rate limiting ──────────────────────────────────────────────────────────

    def _apply_rate_limit(self, url: str) -> None:
        """Sleep until the minimum inter-request interval has elapsed."""
        crawl_delay = self._robots.crawl_delay(url) if self.cfg.respect_robots else 0.0
        min_interval = max(self.cfg.rate_limit_seconds, crawl_delay)
        elapsed = time.monotonic() - self._last_fetch_time
        remaining = min_interval - elapsed
        if remaining > 0:
            logger.debug("Rate-limiting: sleeping %.2f s", remaining)
            time.sleep(remaining)

    # ── HTTP fetch ─────────────────────────────────────────────────────────────

    def _fetch_with_retries(self, url: str) -> str:
        last_exc: Exception | None = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                logger.info("GET %s (attempt %d/%d)", url, attempt, self.cfg.max_retries)
                resp = self._session.get(url, timeout=self.cfg.timeout_seconds)
                self._last_fetch_time = time.monotonic()

                if resp.status_code == 200:
                    return resp.text

                if resp.status_code == 403:
                    raise ScrapingBlockedError(
                        f"Access forbidden (HTTP 403) for {url!r}.  "
                        "The site does not permit automated access to this resource."
                    )

                if resp.status_code == 429:
                    raise ScrapingBlockedError(
                        f"Rate limited by server (HTTP 429) for {url!r}.  "
                        "Too many requests — back off and try again later."
                    )

                if 400 <= resp.status_code < 500:
                    raise ScrapingBlockedError(
                        f"Client error HTTP {resp.status_code} for {url!r}."
                    )

                # 5xx: log and retry
                logger.warning("  Server error %d for %s", resp.status_code, url)
                last_exc = ScrapingBlockedError(f"Server error {resp.status_code}")

            except ScrapingBlockedError:
                raise  # Never retry on explicit blocks
            except requests.RequestException as exc:
                logger.warning("  Request error (attempt %d): %s", attempt, exc)
                last_exc = exc

            if attempt < self.cfg.max_retries:
                backoff = 2.0 ** attempt
                logger.debug("  Retrying in %.0f s", backoff)
                time.sleep(backoff)

        raise ScrapingBlockedError(
            f"Failed to fetch {url!r} after {self.cfg.max_retries} attempts.  "
            f"Last error: {last_exc}"
        )
