"""Model evaluation and betting trend analysis."""

from nfl_chatbot.evaluation.betting_trends import (
    BettingTrends,
    favorite_cover_rate,
    underdog_cover_rate,
    home_favorite_cover_rate,
    road_underdog_cover_rate,
    over_under_hit_rate,
    team_cover_rate,
    team_over_rate,
    spread_bucket_analysis,
    model_edge_analysis,
)

__all__ = [
    "BettingTrends",
    "favorite_cover_rate",
    "underdog_cover_rate",
    "home_favorite_cover_rate",
    "road_underdog_cover_rate",
    "over_under_hit_rate",
    "team_cover_rate",
    "team_over_rate",
    "spread_bucket_analysis",
    "model_edge_analysis",
]
