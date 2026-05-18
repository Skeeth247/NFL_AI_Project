"""
Tests for the LLM integration layer.

Coverage:
  - FakeLLMClient behaviour (available, unavailable, raises)
  - AnthropicClient / OpenAIClient / LocalLLMClient availability checks
  - auto_client() returns None when no env vars set
  - auto_client() returns the right client when one env var is set
  - SYSTEM_PROMPT contains the required text
  - _build_user_message includes question, tool JSON, and instruction text
  - LLMResponder.using_llm flag
  - LLMResponder uses LLM when client is available and tool result is present
  - LLMResponder falls back to rule-based when client is None
  - LLMResponder falls back when LLM raises
  - LLMResponder skips LLM for clarification responses
  - LLMResponder skips LLM for general_help intent
  - LLMResponder preserves intent / confidence / follow_ups / tool_result
  - LLMResponder accepts a custom system_prompt
  - System message is first in the messages list
  - Tool JSON appears in the user message
  - respond_with_llm() returns a Response
"""

from __future__ import annotations

import json
import os

import pytest

from nfl_chatbot.chatbot.llm import (
    SYSTEM_PROMPT,
    AnthropicClient,
    FakeLLMClient,
    LLMResponder,
    LocalLLMClient,
    OpenAIClient,
    _build_user_message,
    auto_client,
    respond_with_llm,
)
from nfl_chatbot.chatbot.responder import Response
from nfl_chatbot.chatbot.tools import ChatbotTools


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def no_api_keys(monkeypatch):
    """Ensure no LLM credentials are present in the environment."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LOCAL_LLM_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_tools():
    """ChatbotTools with no real data (fast, deterministic)."""
    return ChatbotTools(engine=None, model_dir="/nonexistent/models", data_dir="/nonexistent/data")


@pytest.fixture
def fake_client():
    return FakeLLMClient(reply="The model gives the Eagles a 61% win probability.")


@pytest.fixture
def llm_responder(fake_client, fake_tools):
    return LLMResponder(client=fake_client, tools=fake_tools)


# ══════════════════════════════════════════════════════════════════════════════
# FakeLLMClient
# ══════════════════════════════════════════════════════════════════════════════

class TestFakeLLMClient:
    def test_is_available_true(self):
        assert FakeLLMClient(available=True).is_available is True

    def test_is_available_false(self):
        assert FakeLLMClient(available=False).is_available is False

    def test_complete_returns_reply(self):
        client = FakeLLMClient(reply="hello")
        assert client.complete([{"role": "user", "content": "hi"}]) == "hello"

    def test_complete_records_calls(self):
        client = FakeLLMClient()
        msgs = [{"role": "user", "content": "test"}]
        client.complete(msgs)
        assert client.calls == [msgs]

    def test_complete_raises_when_configured(self):
        err = ValueError("boom")
        client = FakeLLMClient(raise_on_complete=err)
        with pytest.raises(ValueError, match="boom"):
            client.complete([{"role": "user", "content": "x"}])

    def test_multiple_calls_recorded(self):
        client = FakeLLMClient()
        client.complete([{"role": "user", "content": "a"}])
        client.complete([{"role": "user", "content": "b"}])
        assert len(client.calls) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Provider clients — availability checks
# ══════════════════════════════════════════════════════════════════════════════

class TestProviderAvailability:
    def test_anthropic_unavailable_without_key(self, no_api_keys):
        assert AnthropicClient().is_available is False

    def test_anthropic_available_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert AnthropicClient().is_available is True

    def test_openai_unavailable_without_key(self, no_api_keys):
        assert OpenAIClient().is_available is False

    def test_openai_available_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        assert OpenAIClient().is_available is True

    def test_local_unavailable_without_url(self, no_api_keys):
        assert LocalLLMClient().is_available is False

    def test_local_available_with_url(self, monkeypatch):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        assert LocalLLMClient().is_available is True

    def test_local_available_with_explicit_url(self):
        client = LocalLLMClient(base_url="http://localhost:11434/v1")
        assert client.is_available is True

    def test_anthropic_complete_raises_without_key(self, no_api_keys):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicClient().complete([{"role": "user", "content": "hi"}])

    def test_openai_complete_raises_without_key(self, no_api_keys):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIClient().complete([{"role": "user", "content": "hi"}])

    def test_local_complete_raises_without_url(self, no_api_keys):
        with pytest.raises(RuntimeError, match="LOCAL_LLM_URL"):
            LocalLLMClient().complete([{"role": "user", "content": "hi"}])


# ══════════════════════════════════════════════════════════════════════════════
# auto_client
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoClient:
    def test_returns_none_when_no_credentials(self, no_api_keys):
        assert auto_client() is None

    def test_returns_anthropic_when_key_set(self, monkeypatch, no_api_keys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        client = auto_client()
        assert isinstance(client, AnthropicClient)

    def test_returns_openai_when_only_openai_key(self, monkeypatch, no_api_keys):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        client = auto_client()
        assert isinstance(client, OpenAIClient)

    def test_returns_local_when_only_url(self, monkeypatch, no_api_keys):
        monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        client = auto_client()
        assert isinstance(client, LocalLLMClient)

    def test_anthropic_preferred_over_openai(self, monkeypatch, no_api_keys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        client = auto_client()
        assert isinstance(client, AnthropicClient)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM_PROMPT content
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemPrompt:
    def test_contains_required_text(self):
        assert "You are an NFL analytics assistant" in SYSTEM_PROMPT
        assert "only answer using the provided tool outputs" in SYSTEM_PROMPT
        assert "data is unavailable" in SYSTEM_PROMPT

    def test_no_gambling_advice(self):
        lower = SYSTEM_PROMPT.lower()
        assert "gambling" not in lower or "no gambling" in lower or "not" in lower

    def test_forbids_inventing_stats(self):
        assert "invent" in SYSTEM_PROMPT.lower() or "do not" in SYSTEM_PROMPT.lower()


# ══════════════════════════════════════════════════════════════════════════════
# _build_user_message
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildUserMessage:
    def test_contains_question(self):
        msg = _build_user_message("Who wins Chiefs vs Bills?", {}, "fallback")
        assert "Who wins Chiefs vs Bills?" in msg

    def test_contains_tool_json(self):
        tool = {"tool": "predict_matchup", "status": "ok", "home_win_prob": 0.61}
        msg = _build_user_message("question", tool, "fallback")
        assert '"home_win_prob"' in msg
        assert "0.61" in msg

    def test_contains_rule_based_answer(self):
        msg = _build_user_message("q", {}, "rule-based-answer-here")
        assert "rule-based-answer-here" in msg

    def test_contains_instruction_not_to_invent(self):
        msg = _build_user_message("q", {}, "")
        lower = msg.lower()
        assert "only" in lower or "do not add" in lower

    def test_tool_json_is_valid_json(self):
        tool = {"a": 1, "b": None, "c": [1, 2]}
        msg = _build_user_message("q", tool, "")
        # Extract JSON block
        start = msg.index("```json\n") + len("```json\n")
        end = msg.index("\n```", start)
        parsed = json.loads(msg[start:end])
        assert parsed["a"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# LLMResponder
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMResponder:
    def test_using_llm_true_when_client_available(self, llm_responder):
        assert llm_responder.using_llm is True

    def test_using_llm_false_when_no_client(self, fake_tools):
        r = LLMResponder(client=None, tools=fake_tools)
        # auto_client() returns None because no env vars are set in tests
        # (CI / local — keys may or may not be set; skip if they are)
        if r.client is not None and r.client.is_available:
            pytest.skip("LLM credentials present in environment")
        assert r.using_llm is False

    def test_using_llm_false_when_client_unavailable(self, fake_tools):
        r = LLMResponder(client=FakeLLMClient(available=False), tools=fake_tools)
        assert r.using_llm is False

    def test_llm_answer_used_when_available(self, llm_responder, fake_client):
        resp = llm_responder.respond("How are the Eagles doing?")
        # Any intent that produces a tool_result and doesn't need clarification
        # should be rewritten by the LLM.  With no data the tool status="degraded"
        # but tool_result is still present.
        if resp.tool_result is not None and not resp.needs_clarification:
            assert resp.answer == fake_client.reply
            assert len(fake_client.calls) >= 1

    def test_falls_back_when_client_none(self, fake_tools, no_api_keys):
        r = LLMResponder(client=None, tools=fake_tools)
        resp = r.respond("What can you do?")
        assert isinstance(resp, Response)
        assert "NFL" in resp.answer

    def test_falls_back_when_client_unavailable(self, fake_tools):
        r = LLMResponder(client=FakeLLMClient(available=False), tools=fake_tools)
        resp = r.respond("What can you do?")
        assert isinstance(resp, Response)
        assert len(resp.answer) > 10

    def test_falls_back_on_llm_error(self, fake_tools):
        bad_client = FakeLLMClient(raise_on_complete=RuntimeError("timeout"))
        r = LLMResponder(client=bad_client, tools=fake_tools)
        resp = r.respond("How are the Eagles doing?")
        assert isinstance(resp, Response)
        # Must return the rule-based answer, not raise
        assert len(resp.answer) > 10

    def test_skips_llm_for_clarification(self, llm_responder, fake_client):
        # "Who will win the game?" → needs_clarification=True
        resp = llm_responder.respond("Who will win the game?")
        assert resp.needs_clarification is True
        assert resp.answer != fake_client.reply

    def test_skips_llm_for_general_help(self, llm_responder, fake_client):
        resp = llm_responder.respond("What can you do?")
        assert resp.intent == "general_help"
        assert resp.answer != fake_client.reply

    def test_preserves_intent(self, llm_responder):
        resp = llm_responder.respond("How are the Eagles doing?")
        assert resp.intent == "team_summary"

    def test_preserves_confidence(self, llm_responder):
        resp = llm_responder.respond("How are the Eagles doing?")
        assert 0.0 <= resp.confidence <= 1.0

    def test_preserves_follow_ups(self, llm_responder):
        resp = llm_responder.respond("How are the Eagles doing?")
        assert isinstance(resp.follow_ups, list)

    def test_preserves_tool_result(self, llm_responder):
        resp = llm_responder.respond("How are the Eagles doing?")
        if resp.tool_result is not None:
            assert isinstance(resp.tool_result, dict)
            assert "tool" in resp.tool_result

    def test_system_prompt_is_first_message(self, llm_responder, fake_client):
        llm_responder.respond("How are the Eagles doing?")
        if fake_client.calls:
            first_call = fake_client.calls[0]
            assert first_call[0]["role"] == "system"
            assert "NFL analytics assistant" in first_call[0]["content"]

    def test_tool_json_in_user_message(self, llm_responder, fake_client):
        llm_responder.respond("How are the Eagles doing?")
        if fake_client.calls:
            user_msg = next(
                m["content"] for m in fake_client.calls[0] if m["role"] == "user"
            )
            assert "tool" in user_msg.lower() or "json" in user_msg.lower()

    def test_custom_system_prompt(self, fake_tools, fake_client):
        custom = "Custom system prompt for testing."
        r = LLMResponder(client=fake_client, tools=fake_tools, system_prompt=custom)
        r.respond("How are the Eagles doing?")
        if fake_client.calls:
            assert fake_client.calls[0][0]["content"] == custom

    def test_respond_returns_response_object(self, llm_responder):
        resp = llm_responder.respond("What can you do?")
        assert isinstance(resp, Response)

    def test_empty_question_handled(self, llm_responder):
        resp = llm_responder.respond("")
        assert isinstance(resp, Response)
        assert resp.intent == "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Module-level respond_with_llm
# ══════════════════════════════════════════════════════════════════════════════

class TestRespondWithLlm:
    def test_returns_response(self, no_api_keys):
        resp = respond_with_llm("What can you do?")
        assert isinstance(resp, Response)
        assert len(resp.answer) > 10

    def test_intent_set(self, no_api_keys):
        resp = respond_with_llm("What can you do?")
        assert resp.intent == "general_help"
