import copy
import json
from pathlib import Path

import pytest

from src.providers.parser import parse_anthropic, parse_openai

FIXTURES = Path(__file__).parent / "fixtures"
KEYS = {
    "text", "input_tokens", "output_tokens", "reasoning_tokens",
    "finish_reason", "returned_model", "request_id", "raw_response",
}  # fmt: skip


def load(name):
    return json.loads((FIXTURES / name).read_text())


# ------------------------------------------------------------------ the two real responses
def test_anthropic_real_response():
    raw = load("anthropic.json")
    got = parse_anthropic(raw)
    assert set(got) == KEYS
    assert got["text"] == "391"
    assert (got["input_tokens"], got["output_tokens"], got["reasoning_tokens"]) == (
        25,
        3,
        0,
    )
    assert got["finish_reason"] == "end_turn"
    assert got["returned_model"] == "claude-haiku-5-5"
    assert got["request_id"] == "msg_011Cgm1zyd2bNYGoTffkevdf"
    assert json.loads(got["raw_response"]) == raw  # nothing to strip in this one


def test_openai_real_response():
    raw = load("openai.json")
    got = parse_openai(raw)
    assert set(got) == KEYS
    assert got["text"] == "391"
    assert (got["input_tokens"], got["output_tokens"], got["reasoning_tokens"]) == (
        20,
        24,
        17,
    )
    assert got["finish_reason"] == "completed"
    assert got["returned_model"] == "gpt-6-luna"
    assert (
        got["request_id"] == "resp_07d821ed5f2105d4006aed52e5ac8887d28d87fdc8556d2e74"
    )


def test_openai_raw_response_drops_the_opaque_blob_but_keeps_everything_else():
    raw = load("openai.json")
    blob = raw["output"][0]["encrypted_content"]
    stored = json.loads(parse_openai(raw)["raw_response"])
    assert stored["output"][0]["encrypted_content"] == f"<removed {len(blob)} chars>"
    assert stored["usage"] == raw["usage"]
    assert stored["output"][1] == raw["output"][1]
    assert stored["reasoning"]["effort"] == "medium"


def test_parsing_does_not_change_the_dict_it_was_given():
    raw = load("openai.json")
    before = copy.deepcopy(raw)
    parse_openai(raw)
    assert raw == before


# ------------------------------------------------------------------ the text must be found by type, not position
def test_openai_text_is_found_even_when_there_is_no_reasoning_item():
    raw = load("openai.json")
    raw["output"] = [raw["output"][1]]  # the message is now index 0
    assert parse_openai(raw)["text"] == "391"


def test_openai_text_parts_are_joined_in_order():
    raw = load("openai.json")
    raw["output"][1]["content"] = [
        {"type": "output_text", "text": "39", "annotations": [], "logprobs": []},
        {"type": "output_text", "text": "1", "annotations": [], "logprobs": []},
    ]
    assert parse_openai(raw)["text"] == "391"


def test_openai_without_a_message_has_no_text():
    raw = load("openai.json")
    raw["output"] = [raw["output"][0]]  # only the reasoning item
    assert parse_openai(raw)["text"] is None


def test_anthropic_text_comes_only_from_text_blocks():
    raw = load("anthropic.json")
    raw["content"] = [
        {"type": "thinking", "thinking": "let me work it out", "signature": "abc"},
        {"type": "text", "text": "391", "citations": None},
    ]
    assert parse_anthropic(raw)["text"] == "391"


def test_anthropic_text_blocks_are_joined_in_order():
    raw = load("anthropic.json")
    raw["content"] = [{"type": "text", "text": "39"}, {"type": "text", "text": "1"}]
    assert parse_anthropic(raw)["text"] == "391"


# ------------------------------------------------------------------ cut-off replies
# ASSUMED shapes (from the docs, not yet seen for real). After the cut-off experiment, replace
# these with real fixtures.
def test_anthropic_cut_off_is_reported_as_max_tokens():
    raw = load("anthropic.json")
    raw["stop_reason"] = "max_tokens"
    assert parse_anthropic(raw)["finish_reason"] == "max_tokens"


def test_openai_cut_off_names_the_reason():
    raw = load("openai.json")
    raw["status"] = "incomplete"
    raw["incomplete_details"] = {"reason": "max_output_tokens"}
    assert parse_openai(raw)["finish_reason"] == "incomplete:max_output_tokens"


# ------------------------------------------------------------------ missing fields must not crash
@pytest.mark.parametrize("parse", [parse_anthropic, parse_openai])
def test_an_empty_response_gives_nones_and_does_not_crash(parse):
    got = parse({})
    assert set(got) == KEYS
    for key in KEYS - {"raw_response"}:
        assert got[key] is None, key
    assert json.loads(got["raw_response"]) == {}


def test_openai_without_reasoning_details_reports_none():
    raw = load("openai.json")
    del raw["usage"]["output_tokens_details"]
    got = parse_openai(raw)
    assert got["reasoning_tokens"] is None
    assert got["output_tokens"] == 24
