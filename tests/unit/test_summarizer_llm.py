"""Summarizer LLM path: tool call, temperature retry, number check, fallbacks."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from careagent.agents import summarizer
from careagent.graph.state import initial_state
from careagent.scoring.peer import DISCLAIMER
from tests.unit.test_agents import TestSummarizerTemplate


@contextmanager
def fake_db():
    yield MagicMock()


def completion(narrative: str, tokens: int = 321):
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"narrative": narrative})))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))],
        usage=SimpleNamespace(total_tokens=tokens),
    )


@pytest.fixture
def state():
    return TestSummarizerTemplate().comparable_state(initial_state("1234567890", "r"))


@pytest.fixture
def with_key():
    with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
         patch.object(summarizer, "get_db", fake_db), \
         patch.object(summarizer, "update_provider_narrative") as save:
        yield save


class TestNumbersGrounded:
    facts = ["Specialty peers compared: 891", "cost: 3.1x the specialty median (99th percentile)"]

    def test_numbers_from_facts_pass(self):
        assert summarizer.numbers_grounded("Cost is 3.1x the median of 891 peers.", self.facts)

    def test_invented_number_fails(self):
        assert not summarizer.numbers_grounded("Cost is 4.7x the median.", self.facts)

    def test_small_counting_words_allowed(self):
        assert summarizer.numbers_grounded("2 metrics stand out; cost is 3.1x.", self.facts)

    def test_thousands_separator(self):
        assert summarizer.numbers_grounded("Compared with 891 peers.", self.facts)
        assert summarizer.numbers_grounded("Among 1,200 peers", ["peers: 1200"])


class TestLLMPath:
    def test_grounded_narrative_is_used(self, state, with_key):
        good = "Billing is 3.1x the Cardiology median among 891 peers."
        with patch.object(summarizer, "_call_gpt", return_value=(json.dumps({"narrative": good}), 321)):
            out = summarizer.summarizer_agent(state)
        assert out["narrative_source"] == "llm"
        assert out["performance_narrative"].startswith(good)
        assert out["performance_narrative"].endswith(DISCLAIMER)
        assert out["llm_tokens_used"] == 321
        with_key.assert_called_once()

    def test_hallucinated_number_falls_back_to_template(self, state, with_key):
        bad = "Billing is 7.4x the median."
        with patch.object(summarizer, "_call_gpt", return_value=(json.dumps({"narrative": bad}), 300)):
            out = summarizer.summarizer_agent(state)
        assert out["narrative_source"] == "template"
        assert "7.4" not in out["performance_narrative"]

    def test_api_error_falls_back(self, state, with_key):
        err = openai.APIConnectionError(request=httpx.Request("POST", "https://x"))
        with patch.object(summarizer, "_call_gpt", side_effect=err):
            out = summarizer.summarizer_agent(state)
        assert out["narrative_source"] == "template" and out["llm_tokens_used"] == 0

    def test_no_llm_call_when_nothing_to_compare(self, with_key):
        s = {**initial_state("1234567890", "r"), "billing_pattern_status": "insufficient_peers",
             "provider_type": "Rare", "peer_group_size": 3}
        with patch.object(summarizer, "_call_gpt") as call:
            out = summarizer.summarizer_agent(s)
        call.assert_not_called()
        assert out["narrative_source"] == "template"

    def test_no_key_uses_template(self, state):
        with patch.object(summarizer.settings, "openai_api_key", ""), \
             patch.object(summarizer, "get_db", fake_db), \
             patch.object(summarizer, "update_provider_narrative"), \
             patch.object(summarizer, "_call_gpt") as call:
            out = summarizer.summarizer_agent(state)
        call.assert_not_called()
        assert out["narrative_source"] == "template"


class TestCallGpt:
    def test_uses_tools_timeout_and_completion_tokens(self):
        client = MagicMock()
        client.chat.completions.create.return_value = completion("x")
        with patch.object(summarizer, "OpenAI", return_value=client) as ctor:
            args, tokens = summarizer._call_gpt("prompt")
        assert json.loads(args) == {"narrative": "x"} and tokens == 321
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == [summarizer.NARRATIVE_TOOL]
        assert "max_completion_tokens" in kwargs and "max_tokens" not in kwargs
        assert ctor.call_args.kwargs["timeout"] == summarizer.settings.openai_timeout_seconds

    def test_model_that_rejects_temperature_is_retried_without_it(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
        err = openai.BadRequestError("Unsupported value: 'temperature'", response=resp, body=None)
        client = MagicMock()
        client.chat.completions.create.side_effect = [err, completion("y")]
        with patch.object(summarizer, "OpenAI", return_value=client):
            args, _ = summarizer._call_gpt("prompt")
        assert json.loads(args)["narrative"] == "y"
        second = client.chat.completions.create.call_args_list[1].kwargs
        assert "temperature" not in second

    def test_other_bad_request_is_raised(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
        err = openai.BadRequestError("context length exceeded", response=resp, body=None)
        client = MagicMock()
        client.chat.completions.create.side_effect = err
        with patch.object(summarizer, "OpenAI", return_value=client), \
             pytest.raises(openai.BadRequestError):
            summarizer._call_gpt("prompt")

    def test_missing_usage_counts_zero_tokens(self):
        client = MagicMock()
        resp = completion("z")
        resp.usage = None
        client.chat.completions.create.return_value = resp
        with patch.object(summarizer, "OpenAI", return_value=client):
            assert summarizer._call_gpt("p")[1] == 0
