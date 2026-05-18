"""Chatbot layer: intent classification, LLM providers, tool registry, and agent loop."""

from nfl_chatbot.chatbot.intents import classify_intent, IntentResult, INTENTS
from nfl_chatbot.chatbot.tools import (
    ChatbotTools,
    predict_matchup,
    compare_players,
    answer_betting_trend,
    summarize_team,
    explain_last_prediction,
    get_model_metrics,
    get_available_data_summary,
)
from nfl_chatbot.chatbot.responder import Responder, Response, respond
from nfl_chatbot.chatbot.llm import (
    LLMClient,
    LLMResponder,
    AnthropicClient,
    OpenAIClient,
    LocalLLMClient,
    FakeLLMClient,
    auto_client,
    respond_with_llm,
    SYSTEM_PROMPT,
)

__all__ = [
    "classify_intent",
    "IntentResult",
    "INTENTS",
    "ChatbotTools",
    "predict_matchup",
    "compare_players",
    "answer_betting_trend",
    "summarize_team",
    "explain_last_prediction",
    "get_model_metrics",
    "get_available_data_summary",
    "Responder",
    "Response",
    "respond",
    "LLMClient",
    "LLMResponder",
    "AnthropicClient",
    "OpenAIClient",
    "LocalLLMClient",
    "FakeLLMClient",
    "auto_client",
    "respond_with_llm",
    "SYSTEM_PROMPT",
]
