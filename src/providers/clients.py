import random
import time

import anthropic
import openai

from src.providers.parser import parse_anthropic, parse_openai

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 30.0
RETRYABLE_KINDS = {"rate_limited", "server_error", "timeout", "connection"}

API_ERRORS = (anthropic.APIError, openai.APIError)

QUOTA_MARKERS = ("insufficient_quota", "exceeded your current quota", "credit balance")

PARSERS = {"anthropic": parse_anthropic, "openai": parse_openai}


def make_client(provider, timeout=120):
    if provider == "anthropic":
        return anthropic.Anthropic(max_retries=0, timeout=timeout)
    if provider == "openai":
        return openai.OpenAI(max_retries=0, timeout=timeout)
    raise ValueError(f"unknown provider {provider!r}")


# Provider-specific request builders.
def _send_anthropic(client, model, prompt, settings):
    kwargs = {
        "model": model,
        "max_tokens": settings["max_tokens"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if settings.get("system"):
        kwargs["system"] = settings["system"]
    return client.messages.create(**kwargs).model_dump(mode="json")


def _send_openai(client, model, prompt, settings):
    kwargs = {
        "model": model,
        "input": prompt,
        "max_output_tokens": settings["max_tokens"],
    }
    if "temperature" in settings:
        kwargs["temperature"] = settings["temperature"]
    if settings.get("reasoning_effort"):
        kwargs["reasoning"] = {"effort": settings["reasoning_effort"]}
    if settings.get("system"):
        kwargs["instructions"] = settings["system"]
    return client.responses.create(**kwargs).model_dump(mode="json")


SENDERS = {"anthropic": _send_anthropic, "openai": _send_openai}


def error_kind(error):
    """One word for what went wrong, so the right thing can be done about it."""
    text = f"{error} {getattr(error, 'body', '')}".lower()
    status = getattr(error, "status_code", None)
    if any(marker in text for marker in QUOTA_MARKERS):
        return "quota_exhausted"  # retrying cannot help
    if isinstance(error, (anthropic.APITimeoutError, openai.APITimeoutError)):
        return "timeout"
    if isinstance(error, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return "connection"
    if status == 429:
        return "rate_limited"
    if status is not None and status >= 500:
        return "server_error"
    if status in (401, 403):
        return "auth"
    if status is not None and 400 <= status < 500:
        return "bad_request"
    return "other"


def call_model(
    client,
    provider,
    model,
    prompt,
    settings,
    sleep=time.sleep,
    rng=random,
    max_attempts=MAX_ATTEMPTS,
):
    # Make one call, with retries. Always returns a result dict, even when every attempt failed.

    # All the things to store in db
    result = {
        "provider": provider,
        "requested_model": model,
        "returned_model": None,
        "text": None,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "finish_reason": None,
        "request_id": None,
        "http_status": None,
        "error": None,
        "latency": None,
        "attempts": 0,
        "raw_response": None,
    }

    send = SENDERS[
        provider
    ]  # chose the request creator func with provider = "anthropic" or "openai"

    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        started = time.monotonic()
        try:
            raw = send(client, model, prompt, settings)  # create the request and send
        except API_ERRORS as error:
            result["latency"] = time.monotonic() - started
            kind = error_kind(error)
            result["http_status"] = getattr(error, "status_code", None)
            result["error"] = f"{kind}: {str(error)[:300]}"
            if kind in RETRYABLE_KINDS and attempt < max_attempts:
                sleep_time = random.uniform(55, 65)
                time.sleep(sleep_time)
                continue
            elif attempt == max_attempts:
                # SEND A WARNING THROUGH TELEGRAM!!! (TODO #1)
                continue
        result["latency"] = time.monotonic() - started
        result.update(PARSERS[provider](raw))
        result["http_status"] = 200
        result["error"] = None
        return result
    return result
