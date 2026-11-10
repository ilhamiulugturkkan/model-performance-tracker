"""Tests for the call wrappers. The REAL SDKs run; only the network is faked (httpx2.MockTransport)."""

import json
from pathlib import Path

import anthropic
import httpx2
import openai
import pytest

from src.providers.clients import MAX_ATTEMPTS, call_model, error_kind, make_client
from src.providers.models import MODELS, parameters_record

FIXTURES = Path(__file__).parent / "fixtures"
ANTHROPIC_OK = json.loads((FIXTURES / "anthropic.json").read_text())
OPENAI_OK = json.loads((FIXTURES / "openai.json").read_text())
PROMPT = "What is 17 * 23? Reply with only the number."

HAIKU = MODELS["haiku"]
LUNA = MODELS["luna"]


class FixedRandom:
    def uniform(self, low, high):
        return 1.0


def fake_network(*replies):
    """A transport that answers with `replies` in order. A reply is an httpx2.Response or an exception."""
    seen = []
    queue = list(replies)

    def handler(request):
        seen.append(request)
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    return httpx2.MockTransport(handler), seen


def anthropic_client(*replies):
    transport, seen = fake_network(*replies)
    client = anthropic.Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=transport)
    )
    return client, seen


def openai_client(*replies):
    transport, seen = fake_network(*replies)
    client = openai.OpenAI(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=transport)
    )
    return client, seen


def ok(body):
    return httpx2.Response(200, json=body)


def call(client, config, sleeps=None, **kwargs):
    sleeps = [] if sleeps is None else sleeps
    return call_model(
        client, config["provider"], config["model"], PROMPT, config["settings"],
        sleep=sleeps.append, rng=FixedRandom(), **kwargs,
    )  # fmt: skip


# ----------------------------------------------------------------- success, through the real SDKs
def test_anthropic_success_and_exactly_what_was_sent():
    client, seen = anthropic_client(ok(ANTHROPIC_OK))
    result = call(client, HAIKU)
    assert (result["text"], result["input_tokens"], result["output_tokens"]) == (
        "391",
        25,
        3,
    )
    assert (
        result["finish_reason"],
        result["returned_model"],
        result["http_status"],
    ) == ("end_turn", "claude-haiku-5-5", 200)
    assert (result["requested_model"], result["attempts"], result["error"]) == (
        "claude-haiku-5-5",
        1,
        None,
    )
    assert result["latency"] is not None and result["raw_response"]
    sent = json.loads(seen[0].content)
    assert sent["model"] == "claude-haiku-5-5" and sent["max_tokens"] == 4096
    assert sent["messages"] == [{"role": "user", "content": PROMPT}]
    assert "temperature" not in sent  # Claude 5.5 models reject one


def test_openai_success_and_exactly_what_was_sent():
    client, seen = openai_client(ok(OPENAI_OK))
    result = call(client, LUNA)
    assert (
        result["text"],
        result["input_tokens"],
        result["output_tokens"],
        result["reasoning_tokens"],
    ) == ("391", 20, 24, 17)
    assert (result["finish_reason"], result["returned_model"], result["attempts"]) == (
        "completed",
        "gpt-6-luna",
        1,
    )
    assert (
        "<removed" in result["raw_response"]
    )  # the opaque reasoning blob is not stored
    sent = json.loads(seen[0].content)
    assert sent["model"] == "gpt-6-luna" and sent["input"] == PROMPT
    assert sent["max_output_tokens"] == 4096
    assert sent["temperature"] == 0
    assert sent["reasoning"] == {"effort": "none"}


def test_a_system_prompt_is_sent_when_one_is_set():
    client, seen = anthropic_client(ok(ANTHROPIC_OK))
    call(client, dict(HAIKU, settings={"max_tokens": 100, "system": "Be brief."}))
    assert json.loads(seen[0].content)["system"] == "Be brief."


# ----------------------------------------------------------------- retries
def test_server_errors_are_retried_until_it_works():
    client, seen = anthropic_client(
        httpx2.Response(500, json={}), httpx2.Response(529, json={}), ok(ANTHROPIC_OK)
    )
    sleeps = []
    result = call(client, HAIKU, sleeps)
    assert (result["text"], result["attempts"], result["error"]) == ("391", 3, None)
    assert len(seen) == 3
    assert sleeps == [1.0, 2.0]  # exponential: 1s, then 2s


def test_the_providers_retry_after_hint_is_honoured():
    client, _ = openai_client(
        httpx2.Response(
            429, json={"error": {"message": "slow down"}}, headers={"retry-after": "7"}
        ),
        ok(OPENAI_OK),
    )
    sleeps = []
    result = call(client, LUNA, sleeps)
    assert result["text"] == "391"
    assert sleeps == [7.0]


def test_after_the_last_attempt_the_failure_is_recorded_not_raised():
    client, seen = anthropic_client(
        httpx2.Response(500, json={"error": {"message": "boom"}})
    )
    result = call(client, HAIKU)
    assert result["text"] is None
    assert result["attempts"] == MAX_ATTEMPTS == len(seen)
    assert result["error"].startswith("server_error:")
    assert result["http_status"] == 500


def test_a_dropped_connection_is_retried_then_recorded():
    client, _seen = openai_client(httpx2.ConnectError("no route"))
    result = call(client, LUNA)
    assert result["error"].startswith("connection:")
    assert result["attempts"] == MAX_ATTEMPTS


def test_a_timeout_is_retried_then_recorded():
    client, _ = anthropic_client(httpx2.ReadTimeout("too slow"))
    result = call(client, HAIKU)
    assert result["error"].startswith("timeout:")
    assert result["attempts"] == MAX_ATTEMPTS


# ----------------------------------------------------------------- failures that must NOT be retried
def test_a_rate_limit_is_retried():
    client, _ = openai_client(
        httpx2.Response(
            429, json={"error": {"message": "Rate limit reached for tokens per min"}}
        ),
        ok(OPENAI_OK),
    )
    assert call(client, LUNA)["attempts"] == 2


def test_out_of_money_is_not_retried_and_is_named_openai():
    body = {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                      "type": "insufficient_quota", "param": None, "code": "insufficient_quota"}}  # fmt: skip
    client, seen = openai_client(httpx2.Response(429, json=body))
    result = call(client, LUNA)
    assert result["error"].startswith("quota_exhausted:")
    assert (result["attempts"], len(seen)) == (1, 1)


def test_out_of_money_is_not_retried_and_is_named_anthropic():
    # ASSUMED wording: confirm it against a real low-balance error before relying on it.
    body = {"type": "error", "error": {"type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API."}}  # fmt: skip
    client, seen = anthropic_client(httpx2.Response(400, json=body))
    result = call(client, HAIKU)
    assert result["error"].startswith("quota_exhausted:")
    assert len(seen) == 1


@pytest.mark.parametrize(
    "status,kind",
    [(400, "bad_request"), (401, "auth"), (403, "auth"), (404, "bad_request")],
)
def test_client_errors_are_recorded_after_one_attempt(status, kind):
    client, seen = openai_client(
        httpx2.Response(status, json={"error": {"message": "nope"}})
    )
    result = call(client, LUNA)
    assert result["error"].startswith(f"{kind}:")
    assert (result["attempts"], result["http_status"], len(seen)) == (1, status, 1)


def test_our_own_bugs_are_not_recorded_as_failed_calls():
    class Broken:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise TypeError("a bug in OUR code")

    with pytest.raises(TypeError):
        call(Broken(), HAIKU)


# ----------------------------------------------------------------- small pieces
def test_the_sdks_own_retries_are_switched_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert make_client("anthropic").max_retries == 0
    assert make_client("openai").max_retries == 0
    with pytest.raises(ValueError):
        make_client("nobody")


def test_error_kind_on_an_unknown_error_is_other():
    assert error_kind(RuntimeError("???")) == "other"


def test_parameters_record_says_so_when_a_temperature_cannot_be_set():
    assert parameters_record(HAIKU)["temperature"] == "provider default (not settable)"
    assert parameters_record(LUNA)["temperature"] == 0
    assert parameters_record(LUNA)["reasoning_effort"] == "none"
