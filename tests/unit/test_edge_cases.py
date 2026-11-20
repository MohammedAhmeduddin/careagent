"""Edge cases: NPI validation, LLM time budget, narrative cache, rate limiting."""

import json

import pandas as pd
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from careagent.agents import summarizer
from careagent.api import ratelimit
from careagent.api.main import app
from careagent.api.ratelimit import SlidingWindowLimiter, client_key
from careagent.graph.state import initial_state
from careagent.scoring.peer import prepare, score_providers
from tests.unit.test_agents import TestSummarizerTemplate
from tests.unit.test_api import final_state
from tests.unit.test_summarizer_llm import bad_request


@contextmanager
def fake_db():
    yield MagicMock()


@pytest.fixture
def state():
    return TestSummarizerTemplate().comparable_state(initial_state("1234567890", "r"))


# ── NPI validation in the loader ──────────────────────────────────────────────

class TestNpiValidation:
    def test_duplicate_npi_is_a_clear_error(self, cms_frame):
        dup = cms_frame.iloc[[0]].copy()
        with pytest.raises(ValueError, match="appear more than once.*1000000001"):
            prepare(pd.concat([cms_frame, dup], ignore_index=True))

    @pytest.mark.parametrize("bad", ["12345", "12345678901", "abcdefghij", "", "nan"])
    def test_malformed_npi_is_a_clear_error(self, cms_frame, bad):
        cms_frame.loc[0, "Rndrng_NPI"] = bad
        with pytest.raises(ValueError, match="not 10 digits"):
            score_providers(cms_frame)

    def test_missing_npi_value_is_rejected(self, cms_frame):
        cms_frame.loc[0, "Rndrng_NPI"] = None
        with pytest.raises(ValueError, match="not 10 digits"):
            prepare(cms_frame)

    def test_valid_frame_passes(self, cms_frame):
        assert len(prepare(cms_frame)) == len(cms_frame)


# ── LLM time budget ───────────────────────────────────────────────────────────

def completion(narrative="ok"):
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"narrative": narrative})))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))],
                           usage=SimpleNamespace(total_tokens=10))


class TestTimeBudget:
    def test_defaults_fit_inside_cloud_run_limit(self):
        s = summarizer.settings
        assert s.openai_total_budget_seconds < 120
        assert s.openai_timeout_seconds <= s.openai_total_budget_seconds
        assert s.openai_max_retries == 0

    def test_each_attempt_gets_the_remaining_budget_at_most(self):
        client = MagicMock()
        client.chat.completions.create.return_value = completion()
        with patch.object(summarizer, "OpenAI", return_value=client):
            summarizer._call_gpt("p")
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert 0 < timeout <= summarizer.settings.openai_timeout_seconds

    def test_budget_exhausted_raises_timeout_without_calling(self):
        client = MagicMock()
        with patch.object(summarizer.settings, "openai_total_budget_seconds", 0.5), \
             patch.object(summarizer, "OpenAI", return_value=client), \
             pytest.raises(TimeoutError, match="budget"):
            summarizer._call_gpt("p")
        client.chat.completions.create.assert_not_called()

    def test_retries_stop_when_budget_runs_out(self):
        """A retry after a rejected parameter is skipped if the budget is gone."""
        clock = iter([0.0, 0.0, 100.0])  # deadline set at 0; second attempt at t=100
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            bad_request("Unsupported value: 'temperature'", "temperature")]
        with patch.object(summarizer.time, "monotonic", lambda: next(clock)), \
             patch.object(summarizer, "OpenAI", return_value=client), \
             pytest.raises(TimeoutError):
            summarizer._call_gpt("p")
        assert client.chat.completions.create.call_count == 1

    def test_timeout_falls_back_to_template(self, state):
        with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
             patch.object(summarizer, "get_db", fake_db), \
             patch.object(summarizer, "get_cached_narrative", return_value=None), \
             patch.object(summarizer, "update_provider_narrative"), \
             patch.object(summarizer, "_call_gpt", side_effect=TimeoutError("budget")):
            out = summarizer.summarizer_agent(state)
        assert out["narrative_source"] == "template"


# ── Narrative cache ───────────────────────────────────────────────────────────

class TestNarrativeCache:
    def test_fingerprint_is_stable(self, state):
        assert summarizer.narrative_fingerprint(state) == summarizer.narrative_fingerprint(dict(state))

    def test_fingerprint_changes_with_facts_model_and_prompt(self, state):
        base = summarizer.narrative_fingerprint(state)
        changed = {**state, "peer_group_size": 892}
        assert summarizer.narrative_fingerprint(changed) != base
        with patch.object(summarizer.settings, "openai_model", "other-model"):
            assert summarizer.narrative_fingerprint(state) != base
        with patch.object(summarizer, "PROMPT_VERSION", "99"):
            assert summarizer.narrative_fingerprint(state) != base

    def test_cached_narrative_is_reused_without_llm_call(self, state):
        with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
             patch.object(summarizer, "get_db", fake_db), \
             patch.object(summarizer, "get_cached_narrative", return_value="Stored text."), \
             patch.object(summarizer, "update_provider_narrative") as save, \
             patch.object(summarizer, "_call_gpt") as call:
            out = summarizer.summarizer_agent(state)
        call.assert_not_called()
        save.assert_not_called()
        assert out["performance_narrative"] == "Stored text."
        assert out["narrative_source"] == "llm" and out["narrative_cached"] is True
        assert out["llm_tokens_used"] == 0

    def test_fresh_llm_narrative_is_stored_with_fingerprint(self, state):
        good = "Billing is 3.1x the Cardiology median among 891 peers."
        with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
             patch.object(summarizer, "get_db", fake_db), \
             patch.object(summarizer, "get_cached_narrative", return_value=None), \
             patch.object(summarizer, "update_provider_narrative") as save, \
             patch.object(summarizer, "_call_gpt", return_value=(json.dumps({"narrative": good}), 50)):
            out = summarizer.summarizer_agent(state)
        assert out["narrative_cached"] is False
        assert save.call_args.kwargs["fingerprint"] == summarizer.narrative_fingerprint(state)

    def test_template_is_stored_without_fingerprint(self, state):
        with patch.object(summarizer.settings, "openai_api_key", "sk-test"), \
             patch.object(summarizer, "get_db", fake_db), \
             patch.object(summarizer, "get_cached_narrative", return_value=None), \
             patch.object(summarizer, "update_provider_narrative") as save, \
             patch.object(summarizer, "_call_gpt", side_effect=RuntimeError("down")):
            summarizer.summarizer_agent(state)
        assert save.call_args.kwargs["fingerprint"] is None

    def test_no_cache_lookup_without_key(self, state):
        with patch.object(summarizer.settings, "openai_api_key", ""), \
             patch.object(summarizer, "get_db", fake_db), \
             patch.object(summarizer, "get_cached_narrative") as lookup, \
             patch.object(summarizer, "update_provider_narrative"):
            summarizer.summarizer_agent(state)
        lookup.assert_not_called()


# ── Rate limiting ─────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestSlidingWindowLimiter:
    def test_allows_up_to_limit_then_blocks(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(60, clock)
        assert all(lim.check("a", 3) == 0 for _ in range(3))
        wait = lim.check("a", 3)
        assert 59 < wait <= 60

    def test_window_slides(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(60, clock)
        for _ in range(3):
            lim.check("a", 3)
        clock.t += 60.1
        assert lim.check("a", 3) == 0

    def test_clients_are_independent(self):
        lim = SlidingWindowLimiter(60, FakeClock())
        for _ in range(3):
            lim.check("a", 3)
        assert lim.check("b", 3) == 0

    def test_zero_limit_disables(self):
        lim = SlidingWindowLimiter(60, FakeClock())
        assert all(lim.check("a", 0) == 0 for _ in range(100))

    def test_blocked_requests_do_not_extend_the_wait(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(60, clock)
        for _ in range(2):
            lim.check("a", 2)
        for _ in range(10):
            lim.check("a", 2)  # rejected, not recorded
        clock.t += 60.1
        assert lim.check("a", 2) == 0

    def test_idle_clients_are_evicted(self):
        clock = FakeClock()
        lim = SlidingWindowLimiter(60, clock)
        with patch.object(ratelimit, "MAX_TRACKED_CLIENTS", 5):
            for i in range(5):
                lim.check(f"old{i}", 10)
            clock.t += 61
            lim.check("new", 10)
        assert set(lim._hits) == {"new"}


class TestClientKey:
    def req(self, xff=None, host="9.9.9.9"):
        headers = {"x-forwarded-for": xff} if xff else {}
        return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))

    def test_uses_last_forwarded_entry(self):
        assert client_key(self.req("1.1.1.1, 2.2.2.2")) == "2.2.2.2"

    def test_spoofed_first_entry_is_ignored(self):
        assert client_key(self.req("6.6.6.6, 3.3.3.3")) == client_key(self.req("7.7.7.7, 3.3.3.3"))

    def test_falls_back_to_socket_address(self):
        assert client_key(self.req()) == "9.9.9.9"

    def test_no_client_info(self):
        assert client_key(SimpleNamespace(headers={}, client=None)) == "unknown"


class TestAnalyzeRateLimit:
    @pytest.fixture
    def pipeline(self):
        with patch("careagent.api.routes.analyze.get_db", fake_db), \
             patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=object()), \
             patch("careagent.api.routes.analyze.create_agent_run"), \
             patch("careagent.api.routes.analyze.complete_agent_run") as complete, \
             patch("careagent.api.routes.analyze.fail_agent_run"), \
             patch("careagent.api.routes.analyze.careagent_graph") as graph:
            graph.invoke.return_value = final_state()
            yield SimpleNamespace(graph=graph, complete=complete)

    def test_429_with_retry_after_once_over_limit(self, pipeline):
        client = TestClient(app)
        with patch.object(ratelimit.get_settings(), "rate_limit_per_minute", 2):
            codes = [client.post("/analyze", json={"npi": "1234567890"}).status_code for _ in range(3)]
            last = client.post("/analyze", json={"npi": "1234567890"})
        assert codes == [200, 200, 429]
        assert last.status_code == 429 and int(last.headers["Retry-After"]) > 0
        assert pipeline.graph.invoke.call_count == 2  # blocked requests never run the pipeline

    def test_health_is_not_rate_limited(self):
        client = TestClient(app)
        with patch.object(ratelimit.get_settings(), "rate_limit_per_minute", 1), \
             patch("careagent.api.routes.health.check_connection", return_value=False):
            assert all(client.get("/health").status_code == 200 for _ in range(5))

    def test_cached_narrative_is_not_counted_as_llm_call(self, pipeline):
        pipeline.graph.invoke.return_value = final_state(narrative_cached=True)
        resp = TestClient(app).post("/analyze", json={"npi": "1234567890"})
        assert resp.json()["narrative_cached"] is True
        assert pipeline.complete.call_args.kwargs["llm_calls"] == 0
