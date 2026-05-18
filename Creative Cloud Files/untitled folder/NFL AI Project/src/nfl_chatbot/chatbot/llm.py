"""
Optional LLM integration for the NFL AI Chatbot.

Architecture
------------
- ``LLMClient`` (ABC) — defines the interface every provider must implement
- ``AnthropicClient``  — uses the Anthropic API (ANTHROPIC_API_KEY)
- ``OpenAIClient``     — uses the OpenAI API (OPENAI_API_KEY)
- ``LocalLLMClient``   — calls any OpenAI-compatible local server (LOCAL_LLM_URL)
- ``FakeLLMClient``    — deterministic stub for unit tests
- ``LLMResponder``     — wraps ``Responder``:
    1. classify intent + call tool (existing rule-based layer)
    2. ask the LLM to rewrite the tool JSON into natural language
    3. fall back to the rule-based answer on any failure

The LLM is only allowed to *rewrite* tool outputs.  It is explicitly forbidden
from inventing stats, teams, players, odds, or predictions.

Environment variables
---------------------
ANTHROPIC_API_KEY   — enables AnthropicClient
OPENAI_API_KEY      — enables OpenAIClient
LOCAL_LLM_URL       — enables LocalLLMClient  (e.g. http://localhost:11434/v1)
LOCAL_LLM_MODEL     — model name for the local server (default: "llama3")

Usage
-----
    from nfl_chatbot.chatbot.llm import LLMResponder
    resp = LLMResponder().respond("Who wins Chiefs vs Bills?")
    print(resp.answer)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from nfl_chatbot.chatbot.responder import Response, Responder
from nfl_chatbot.chatbot.tools import ChatbotTools

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an NFL analytics assistant. "
    "You may only answer using the provided tool outputs. "
    "If the tool output does not contain the answer, say the data is unavailable. "
    "Do not invent statistics, predictions, team names, player names, odds, or scores. "
    "Do not make gambling recommendations. "
    "Keep answers concise and factual. "
    "When citing a probability or percentage, take it directly from the tool output."
)

# ── Abstract base ─────────────────────────────────────────────────────────────


class LLMClient(ABC):
    """
    Minimal interface that every LLM provider must implement.

    ``complete`` receives a list of messages in OpenAI chat format::

        [
          {"role": "system", "content": "..."},
          {"role": "user",   "content": "..."},
        ]

    and returns the model's reply as a plain string.
    """

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* and return the text reply."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when the client has credentials and can make API calls."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(available={self.is_available})"


# ── Anthropic ─────────────────────────────────────────────────────────────────


class AnthropicClient(LLMClient):
    """
    Calls the Anthropic Messages API.

    Requires ``ANTHROPIC_API_KEY`` in the environment.
    Uses the ``anthropic`` SDK when installed; falls back to ``requests``.

    Parameters
    ----------
    model : str
        Anthropic model ID.  Defaults to ``claude-haiku-4-5-20251001`` for speed.
    max_tokens : int
        Upper limit on response tokens.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 512) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.is_available:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")

        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        # Prefer the official SDK when available.
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=self._api_key)
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=user_messages,
            )
            return response.content[0].text

        except ImportError:
            pass  # fall through to requests

        import requests  # stdlib on all platforms

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": user_messages,
        }
        resp = requests.post(self.API_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


# ── OpenAI ────────────────────────────────────────────────────────────────────


class OpenAIClient(LLMClient):
    """
    Calls the OpenAI Chat Completions API.

    Requires ``OPENAI_API_KEY`` in the environment.
    Uses the ``openai`` SDK when installed; falls back to ``requests``.

    Parameters
    ----------
    model : str
        OpenAI model ID (default ``gpt-4o-mini`` for cost efficiency).
    max_tokens : int
        Upper limit on response tokens.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 512) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key: str | None = os.getenv("OPENAI_API_KEY")

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.is_available:
            raise RuntimeError("OPENAI_API_KEY not set.")

        try:
            import openai  # type: ignore

            client = openai.OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""

        except ImportError:
            pass

        import requests

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        resp = requests.post(self.API_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Local LLM ─────────────────────────────────────────────────────────────────


class LocalLLMClient(LLMClient):
    """
    Calls a locally-running OpenAI-compatible server (e.g. Ollama, LM Studio,
    llama.cpp server).

    Requires ``LOCAL_LLM_URL`` in the environment, e.g.::

        LOCAL_LLM_URL=http://localhost:11434/v1
        LOCAL_LLM_MODEL=llama3        # optional, default "llama3"

    Parameters
    ----------
    base_url : str | None
        Override for the server URL (ignores env var when set explicitly).
    model : str | None
        Model name to request.  Falls back to ``LOCAL_LLM_MODEL`` env var.
    max_tokens : int
        Upper limit on response tokens.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self._base_url = (base_url or os.getenv("LOCAL_LLM_URL", "")).rstrip("/")
        self._model = model or os.getenv("LOCAL_LLM_MODEL", "llama3")
        self._max_tokens = max_tokens

    @property
    def is_available(self) -> bool:
        return bool(self._base_url)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.is_available:
            raise RuntimeError("LOCAL_LLM_URL not set.")

        import requests

        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Fake client (testing) ─────────────────────────────────────────────────────


class FakeLLMClient(LLMClient):
    """
    Deterministic stub for unit tests.

    Returns ``reply`` for every call and records all ``messages`` passed to it.

    Parameters
    ----------
    reply : str
        The fixed text to return from ``complete``.
    available : bool
        Whether ``is_available`` returns True (default True).
    raise_on_complete : Exception | None
        If set, ``complete`` raises this exception instead of returning *reply*.
    """

    def __init__(
        self,
        reply: str = "Fake LLM response.",
        available: bool = True,
        raise_on_complete: Exception | None = None,
    ) -> None:
        self.reply = reply
        self._available = available
        self.raise_on_complete = raise_on_complete
        self.calls: list[list[dict[str, str]]] = []

    @property
    def is_available(self) -> bool:
        return self._available

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.raise_on_complete is not None:
            raise self.raise_on_complete
        return self.reply


# ── auto_client ───────────────────────────────────────────────────────────────


def auto_client() -> LLMClient | None:
    """
    Return the first available LLM client, probing in priority order:
    Anthropic → OpenAI → LocalLLM.

    Returns None if no credentials are configured.
    """
    for cls in (AnthropicClient, OpenAIClient, LocalLLMClient):
        client = cls()
        if client.is_available:
            logger.info("LLM client selected: %s", cls.__name__)
            return client
    logger.debug("No LLM credentials found; rule-based responder will be used.")
    return None


# ── Message builder ───────────────────────────────────────────────────────────


def _build_user_message(
    question: str,
    tool_result: dict[str, Any],
    rule_based_answer: str,
) -> str:
    """
    Compose the user-turn message sent to the LLM.

    Includes the original question, full tool JSON, and the rule-based answer
    as a formatting reference.  Instructs the model explicitly not to add facts.
    """
    tool_json = json.dumps(tool_result, indent=2, default=str)
    return (
        f"User question: {question}\n\n"
        f"Tool output (JSON — use ONLY these facts):\n```json\n{tool_json}\n```\n\n"
        f"Rule-based answer (for reference only — do not copy wholesale):\n"
        f"{rule_based_answer}\n\n"
        "Rewrite the answer in clear, natural language. "
        "Use only numbers, team names, and facts present in the tool output above. "
        "Do not add statistics, predictions, or claims not in the tool output. "
        "End with a one-sentence disclaimer that this is not betting advice."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLMResponder
# ══════════════════════════════════════════════════════════════════════════════


class LLMResponder:
    """
    Drop-in replacement for ``Responder`` that optionally rewrites answers
    through an LLM.

    Flow
    ----
    1. ``Responder.respond(question)`` → rule-based ``Response``
    2. If client is None or unavailable → return rule-based response as-is
    3. If response needs clarification or has no tool_result → return as-is
    4. Build a prompt from (question, tool_result, rule_based_answer)
    5. Call ``client.complete(messages)`` → LLM text
    6. Return a new ``Response`` with ``answer=llm_text`` and all other
       fields preserved from the rule-based response
    7. On any LLM error → log warning, return rule-based response

    Parameters
    ----------
    client : LLMClient | None
        LLM backend.  If None, ``auto_client()`` is called; if that also
        returns None the responder behaves identically to the rule-based one.
    tools : ChatbotTools | None
        Tool container forwarded to the underlying ``Responder``.
    system_prompt : str | None
        Override the default ``SYSTEM_PROMPT``.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        tools: ChatbotTools | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client: LLMClient | None = client if client is not None else auto_client()
        self._base = Responder(tools=tools)
        self._system_prompt = system_prompt or SYSTEM_PROMPT

    @property
    def client(self) -> LLMClient | None:
        return self._client

    @property
    def using_llm(self) -> bool:
        """True when an LLM client is active and available."""
        return self._client is not None and self._client.is_available

    # ── Public interface ──────────────────────────────────────────────────────

    def respond(self, question: str) -> Response:
        """
        Classify *question*, call the correct tool, and return a Response.

        The answer is rewritten by the LLM when one is available; otherwise
        the rule-based answer is returned unchanged.
        """
        base = self._base.respond(question)

        # Skip LLM for these cases — the rule-based text is already the right reply.
        if (
            not self.using_llm
            or base.needs_clarification
            or base.tool_result is None
            or base.intent == "general_help"
        ):
            return base

        try:
            llm_text = self._rewrite(question, base)
        except Exception as exc:
            logger.warning(
                "LLM rewrite failed for intent=%s — falling back to rule-based. (%s: %s)",
                base.intent,
                type(exc).__name__,
                exc,
            )
            return base

        return Response(
            answer=llm_text,
            intent=base.intent,
            confidence=base.confidence,
            tool_result=base.tool_result,
            follow_ups=base.follow_ups,
            has_caveat=base.has_caveat,
            needs_clarification=base.needs_clarification,
        )

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _rewrite(self, question: str, base: Response) -> str:
        """Build the prompt and call the LLM.  Returns the model's text reply."""
        assert self._client is not None  # guarded by using_llm check above

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": _build_user_message(
                    question=question,
                    tool_result=base.tool_result,  # type: ignore[arg-type]
                    rule_based_answer=base.answer,
                ),
            },
        ]
        return self._client.complete(messages)


# ── Module-level convenience ──────────────────────────────────────────────────

_default_llm_responder: LLMResponder | None = None


def _get_llm_responder() -> LLMResponder:
    global _default_llm_responder
    if _default_llm_responder is None:
        _default_llm_responder = LLMResponder()
    return _default_llm_responder


def respond_with_llm(question: str) -> Response:
    """
    Respond to *question* using the LLM when available, rule-based otherwise.

    Uses the module-level shared ``LLMResponder`` instance.
    """
    return _get_llm_responder().respond(question)
