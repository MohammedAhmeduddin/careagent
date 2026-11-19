"""
tests/unit/test_llm_and_errors.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regression tests for issues found in the November 2026 audit:

1. The narrative call used the deprecated functions= parameter, a fixed
   temperature and no timeout (the OpenAI client default is 600 s).
2. When the LLM failed, the template fallback was silent: the API response
   did not say which one produced the narrative.
3. agent_runs always stored tokens_used=0 and counted a template as an LLM call.
4. Pipeline exceptions were returned to callers verbatim.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
from fastapi.testclient import TestClient

from careagent.agents import summarizer
from careagent.api.main import app
from tests.unit.test_api_routes import mock_final_state, mock_provider

client = TestClient(app)

NARRATIVE_JSON = json.dumps(
    {"narrative": "Solid performer.", "key_strengths": ["volume"], "key_concerns": []}
)


def _fake_response(arguments=NARRATIVE_JSON, tokens=321):
    call = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=tokens),
    )


def _bad_request(message):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


# ── 1. Request format: tools, timeout, temperature fallback ──────────────────


def test_call_uses_tools_and_a_timeout():
    with patch.object(summarizer, "OpenAI") as mock_openai:
        create = mock_openai.return_value.chat.completions.create
        create.return_value = _fake_response()
        args, tokens = summarizer._call_gpt("prompt")

    _, client_kwargs = mock_openai.call_args
    assert client_kwargs["timeout"] == summarizer.settings.openai_timeout_seconds
    _, call_kwargs = create.call_args
    assert "tools" in call_kwargs and "functions" not in call_kwargs
    assert "max_completion_tokens" in call_kwargs and "max_tokens" not in call_kwargs
    assert json.loads(args)["narrative"] == "Solid performer."
    assert tokens == 321


def test_model_that_rejects_temperature_is_retried_without_it():
    with patch.object(summarizer, "OpenAI") as mock_openai:
        create = mock_openai.return_value.chat.completions.create
        create.side_effect = [
            _bad_request("Unsupported value: 'temperature' does not support 0.1"),
            _fake_response(),
        ]
        args, _ = summarizer._call_gpt("prompt")

    assert create.call_count == 2
    assert "temperature" in create.call_args_list[0].kwargs
    assert "temperature" not in create.call_args_list[1].kwargs
    assert json.loads(args)["narrative"] == "Solid performer."


# ── 2-3. Narrative source and token accounting ───────────────────────────────


def _run_summarizer(call_gpt):
    state = {"npi": "1234567890", "agents_executed": []}
    with (
        patch.object(summarizer, "get_db") as mock_db,
        patch.object(summarizer, "get_provider_by_npi", return_value=MagicMock()),
        patch.object(summarizer, "update_provider_narrative"),
        patch.object(summarizer, "_build_prompt", return_value="prompt"),
        patch.object(summarizer, "_template_narrative", return_value="Template text."),
        patch.object(summarizer, "_call_gpt", side_effect=call_gpt),
        patch.object(summarizer.settings, "openai_api_key", "sk-test"),
    ):
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        return summarizer.summarizer_agent(state)


def test_llm_narrative_is_labelled_and_tokens_recorded():
    result = _run_summarizer(lambda prompt: (NARRATIVE_JSON, 321))
    assert result["narrative_source"] == "llm"
    assert result["llm_tokens_used"] == 321


def test_llm_failure_falls_back_visibly():
    def boom(prompt):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://x"))

    result = _run_summarizer(boom)
    assert result["narrative_source"] == "template"
    assert result["performance_narrative"] == "Template text."
    assert result["llm_tokens_used"] == 0


# ── API: narrative_source exposed, accounting, safe errors ───────────────────


def _analyze(final_state=None, invoke_error=None):
    with (
        patch("careagent.api.routes.analyze.get_provider_by_npi", return_value=mock_provider()),
        patch("careagent.api.routes.analyze.get_db") as mock_db,
        patch("careagent.api.routes.analyze.create_agent_run"),
        patch("careagent.api.routes.analyze.careagent_graph") as mock_graph,
        patch("careagent.api.routes.analyze.complete_agent_run") as complete,
        patch("careagent.api.routes.analyze.fail_agent_run") as fail,
    ):
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        if invoke_error:
            mock_graph.invoke.side_effect = invoke_error
        else:
            mock_graph.invoke.return_value = final_state
        resp = client.post("/analyze", json={"npi": "1234567890"})
    return resp, complete, fail


def test_response_says_where_the_narrative_came_from():
    state = {**mock_final_state(), "narrative_source": "template", "llm_tokens_used": 0}
    resp, _, _ = _analyze(final_state=state)
    assert resp.status_code == 200
    assert resp.json()["narrative_source"] == "template"


def test_agent_run_records_real_tokens_and_llm_calls():
    state = {**mock_final_state(), "narrative_source": "llm", "llm_tokens_used": 321}
    _, complete, _ = _analyze(final_state=state)
    kwargs = complete.call_args.kwargs
    assert kwargs["tokens_used"] == 321
    assert kwargs["llm_calls"] == 1


def test_template_narrative_is_not_counted_as_llm_call():
    state = {**mock_final_state(), "narrative_source": "template", "llm_tokens_used": 0}
    _, complete, _ = _analyze(final_state=state)
    assert complete.call_args.kwargs["llm_calls"] == 0


def test_pipeline_exception_details_are_not_returned_to_caller():
    secret = "could not connect to db.internal.example:5432 password=hunter2"
    resp, _, fail = _analyze(invoke_error=Exception(secret))
    assert resp.status_code == 500
    assert "hunter2" not in resp.text and "db.internal" not in resp.text
    assert "run_id=" in resp.json()["detail"]
    assert secret in fail.call_args.kwargs["error_message"]  # kept for debugging
