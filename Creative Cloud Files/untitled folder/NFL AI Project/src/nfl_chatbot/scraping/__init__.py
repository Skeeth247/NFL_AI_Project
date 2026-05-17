"""Compliant web scraping with robots.txt checking and rate limiting."""

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

__all__ = [
    "InjuryRecord",
    "NflScraper",
    "RobotsChecker",
    "ScraperConfig",
    "ScrapingBlockedError",
    "ScrapingNotAllowedError",
    "parse_injury_html_file",
    "parse_injury_table",
    "records_to_dataframe",
]
