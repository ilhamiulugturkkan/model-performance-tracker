"""Scorer tests. Every case is a reply that must PASS (True) or FAIL (False).

The probes are defined inline so these tests do not depend on the probe files.
A scorer returns (passed, details); a wrong reply is a normal result, never an exception.
"""

import json

import pytest

from src.scorer import get_scorer


def score(scorer, probe, reply):
    passed, _details = get_scorer(scorer)(probe, reply)
    return bool(passed)


# ---------------------------------------------------------------- numeric_match
INT = {"expected": 14, "tolerance": 0}
NEG = {"expected": -392, "tolerance": 0}
DEC = {"expected": -1.0888, "tolerance": 0.0001}

NUMERIC_CASES = [
    (INT, "14", True),
    (INT, "14.", True),  # trailing period
    (INT, " 14\n", True),  # whitespace
    (INT, "14.0", True),
    (INT, "15", False),
    (INT, "The answer is 14", False),  # prompt said: reply with only the number
    (NEG, "-392", True),
    (NEG, "\u2212392", True),  # unicode minus sign
    (NEG, "392", False),  # sign lost
    (DEC, "-1.0888", True),
    (DEC, "-1.0889", True),  # exactly on the tolerance edge
    (DEC, "-1.0887", True),  # exactly on the tolerance edge
    (DEC, "-1.0886", False),  # just outside
    (DEC, "-1.09", False),
]


@pytest.mark.parametrize("probe,reply,want", NUMERIC_CASES)
def test_numeric_match(probe, reply, want):
    assert score("numeric_match", probe, reply) is want


# ------------------------------------------------------------------- json_match
PERSON = {
    "expected": {"name": "Mara", "age": 34, "tracking_number": None},
    "tolerance": 0,
    "extra_keys_allowed": False,
}
MONEY = {"expected": {"total": 30.69}, "tolerance": 0.005, "extra_keys_allowed": False}
GOOD = {"name": "Mara", "age": 34, "tracking_number": None}


def js(d):
    return json.dumps(d)


JSON_CASES = [
    (PERSON, js(GOOD), True),
    (
        PERSON,
        "```json\n" + js(GOOD) + "\n```",
        True,
    ),  # DESIGN CHOICE: accept a code fence (and log it)
    (
        PERSON,
        js({"name": "Mara", "age": 34}),
        False,
    ),  # missing key whose expected value is null
    (PERSON, js({**GOOD, "age": "34"}), False),  # wrong type
    (PERSON, js({**GOOD, "zip": "0150"}), False),  # extra key
    (PERSON, "[1, 2]", False),  # reply is a list
    (PERSON, "null", False),  # reply is null
    (PERSON, "not json at all", False),
    (MONEY, js({"total": 30.69}), True),
    (MONEY, js({"total": 30.7}), False),  # outside tolerance
    (MONEY, js({"subtotal": 27.9}), False),  # missing key, tolerance branch
]


@pytest.mark.parametrize("probe,reply,want", JSON_CASES)
def test_json_match(probe, reply, want):
    assert score("json_match", probe, reply) is want


# ------------------------------------------------------------------- rule_check
# One passing and one failing reply per rule key, plus the traps that bit us.
RULE_CASES = [
    ({"sentences": 2}, "One. Two.", True),
    ({"sentences": 2}, "One. Two.\n", True),  # trailing newline
    ({"sentences": 2}, "One. Two. Three.", False),
    ({"words": 3}, "a b c", True),
    ({"words": 3}, "a b", False),
    ({"unique_words": True}, "red green blue", True),
    ({"unique_words": True}, "Red red blue", False),  # ignoring case
    ({"letters_only": True}, "It equals twenty one", True),
    ({"letters_only": True}, "It equals 21", False),
    ({"each_word_contains": "x"}, "fox six", True),
    ({"each_word_contains": "x"}, "fox cat", False),
    ({"each_word_starts_with": "a"}, "Amber apple", True),  # capital first letter
    ({"each_word_starts_with": "a"}, "amber blue", False),
    ({"min_word_length": 4}, "tree frog", True),
    ({"min_word_length": 4}, "tree cat", False),
    ({"forbidden_letters": ["a"]}, "Why do hens sit", True),
    ({"forbidden_letters": ["a"]}, "Lost An egg", False),  # capital letter
    ({"forbidden_words": ["the"]}, "to keep eggs", True),
    ({"forbidden_words": ["the"]}, "lost The.", False),  # capital + punctuation
    ({"ends_with": "tomorrow"}, "Think of tomorrow.", True),
    ({"ends_with": "tomorrow"}, "Think of today.", False),
    ({"starts_with": "Weird"}, "Weird noises woke me.", True),
    ({"starts_with": "Weird"}, "Strange noises woke me.", False),
    ({"uppercase_words": ["night"]}, "A NIGHT sky.", True),
    ({"uppercase_words": ["night"]}, "A night sky.", False),
    ({"uppercase_words": ["night"]}, "A sky.", False),  # must appear at least once
    ({"list_items": 3}, "lion, fox, deer", True),
    ({"list_items": 3}, "lion, fox, deer, owl, wolf", False),
    ({"digit_count": 3}, "Room 12 and 4.", True),
    ({"digit_count": 3}, "Room 12.", False),
    ({"required_words": ["twenty", "one"]}, "It equals twenty one", True),
    ({"required_words": ["twenty", "one"]}, "It equals twenty two", False),
    ({"words_from_list": ["one", "two"]}, "one two", True),
    ({"words_from_list": ["one", "two"]}, "one tree", False),
    # --- added after finding that the first versions of these cases passed by coincidence ---
    (
        {"words_from_list": ["one", "two", "three"]},
        "one two",
        True,
    ),  # reply is a SUBSET of the list
    (
        {"words_from_list": ["one", "two", "three"]},
        "one four",
        False,
    ),  # a word that is not in the list
    (
        {"uppercase_words": ["night"]},
        "Under the NIGHT.",
        True,
    ),  # punctuation after the word
    (
        {"uppercase_words": ["night"]},
        "A NIGHT sky and a night wind.",
        False,
    ),  # EVERY occurrence must be capitals
    ({"ends_with": "tomorrow"}, "Think of tomorrow.\n", True),  # trailing newline
    (
        {"ends_with": "tomorrow"},
        "Think of Tomorrow.",
        True,
    ),  # DESIGN CHOICE: case-insensitive
    (
        {"ends_with": "tomorrow"},
        "Think of untomorrow.",
        False,
    ),  # end of a longer word is not the last word
    ({"starts_with": "Weird"}, "\nWeird noises woke me.", True),  # leading newline
    (
        {"starts_with": "Weird"},
        "Weirdly quiet today.",
        False,
    ),  # prefix of a longer word
    ({"digit_count": 0}, "No digits here.", True),
    (
        {"digit_count": 0},
        "It has 1 digit",
        False,
    ),  # zero is a real value, not 'rule absent'
]


@pytest.mark.parametrize("rules,reply,want", RULE_CASES)
def test_rule_check(rules, reply, want):
    assert score("rule_check", {"rules": rules}, reply) is want


def test_unknown_rule_key_is_a_loud_error():
    # A typo'd rule must never be silently ignored (the validator should stop it earlier).
    with pytest.raises(ValueError):
        get_scorer("rule_check")({"rules": {"forbidden_latters": ["a"]}}, "anything")


NO_REPLY_PROBES = [
    ("numeric_match", {"expected": 1, "tolerance": 0}),
    ("json_match", {"expected": {"a": 1}, "tolerance": 0, "extra_keys_allowed": False}),
    ("rule_check", {"rules": {"words": 1}}),
]


@pytest.mark.parametrize("scorer,probe", NO_REPLY_PROBES)
def test_missing_reply_is_not_scored(scorer, probe):
    # D11: a call with no reply is not scored. passed is None (never False), so it cannot be
    # stored as a 0 by accident; the caller records calls.error and writes no scores row.
    passed, details = get_scorer(scorer)(probe, None)
    assert passed is None
    assert details == {"reason": "no reply"}


@pytest.mark.parametrize("scorer,probe", NO_REPLY_PROBES)
def test_empty_reply_is_scored_as_a_fail(scorer, probe):
    # An empty string is something the model returned, so it counts in the pass rate.
    passed, details = get_scorer(scorer)(probe, "")
    assert passed is False
    assert details != {"reason": "no reply"}


# ---------------------------------------------------------------- details shape
def test_numeric_details_carry_the_parsed_value():
    passed, details = get_scorer("numeric_match")(
        {"expected": 391, "tolerance": 0}, "391"
    )
    assert passed is True
    assert details["value"] == 391.0 and isinstance(details["reason"], str)


@pytest.mark.parametrize("reply", ["abc", "nan", "inf", "-inf", "1e999"])
def test_numeric_non_answers_fail_with_null_value(reply):
    # nan/inf parse as floats but are not answers, and they are not valid JSON for the details column
    passed, details = get_scorer("numeric_match")(
        {"expected": 1, "tolerance": 0}, reply
    )
    assert passed is False
    assert details["value"] is None


@pytest.mark.parametrize("fenced", [False, True])
def test_json_details_have_a_queryable_fence_flag(fenced):
    reply = js(GOOD)
    if fenced:
        reply = "```json\n" + reply + "\n```"
    passed, details = get_scorer("json_match")(PERSON, reply)
    assert passed is True
    assert details["fenced"] is fenced
    assert not details["reason"].endswith(
        "fence)"
    )  # the flag is a field, not text in the sentence


def test_json_fence_flag_is_kept_on_failures_too():
    _, details = get_scorer("json_match")(PERSON, "```json\n[1, 2]\n```")
    assert details["fenced"] is True


ALL_CALLS = (
    [("numeric_match", p, r) for p, r, _ in NUMERIC_CASES]
    + [("json_match", p, r) for p, r, _ in JSON_CASES]
    + [("rule_check", {"rules": rules}, r) for rules, r, _ in RULE_CASES]
    + [(s, p, None) for s, p in NO_REPLY_PROBES]
    + [(s, p, "") for s, p in NO_REPLY_PROBES]
)


@pytest.mark.parametrize("scorer,probe,reply", ALL_CALLS)
def test_details_is_always_a_json_dict(scorer, probe, reply):
    # The details column is JSON and queried, so every path returns one shape: a dict that serialises.
    _passed, details = get_scorer(scorer)(probe, reply)
    assert isinstance(details, dict) and details
    json.dumps(details, allow_nan=False)
