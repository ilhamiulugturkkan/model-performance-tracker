"""Power test of the text detector: known shifts are injected into no-change burst windows.

The injection tests use made-up rows. The power tests need the burst database
(mpt_burst.db, git-ignored) and skip without it. All are seeded, so they do not flake.
"""

import random

import pytest

from experiments.calibrate_length import BURST_DB, burst_rows
from experiments.power_text import (
    PREAMBLE,
    SHIFTS,
    add_bullets,
    add_preamble,
    inject,
    run_trials,
    smallest_reliable_fraction,
)
from src.detectors.output_dist import extract_features
from src.detectors.pass_rate import load_thresholds

# ---- the injected shifts and inject() (no database) ----


def test_preamble_adds_one_short_line_and_no_markdown():
    before, after = extract_features("42"), extract_features(add_preamble("42"))
    assert after["lines"] == before["lines"] + 1
    assert after["markdown"] == before["markdown"] == 0
    assert after["short_start"] == 1.0
    assert add_preamble("42").startswith(PREAMBLE)


def test_bullets_add_two_bullet_lines_and_two_markers():
    before, after = extract_features("42"), extract_features(add_bullets("42"))
    assert after["lines"] == before["lines"] + 2
    assert after["markdown"] == before["markdown"] + 2
    assert add_bullets("42\n\n") == "42\n- point\n- point"


def rows(n=20):
    return [{"probe_id": i % 4, "raw_output": f"reply {i}"} for i in range(n)]


@pytest.mark.parametrize(
    "fraction,expected", [(0.0, 0), (0.1, 2), (0.25, 5), (1.0, 20)]
)
def test_inject_changes_exactly_the_requested_share(fraction, expected):
    original = rows()
    out = inject(original, add_preamble, fraction, random.Random(0))
    changed = [o for o, n in zip(original, out) if o["raw_output"] != n["raw_output"]]
    assert len(changed) == expected
    assert original == rows()  # the input is not modified


def test_inject_never_touches_rows_without_a_reply():
    data = [{"probe_id": 1, "raw_output": None}, {"probe_id": 1, "raw_output": "x"}]
    out = inject(data, add_preamble, 1.0, random.Random(0))
    assert out[0]["raw_output"] is None and out[1]["raw_output"].startswith(PREAMBLE)


def test_smallest_reliable_fraction():
    curve = {
        f: {"power": p}
        for f, p in [(0.0, 0.0), (0.1, 0.2), (0.2, 0.9), (0.5, 0.7), (1.0, 1.0)]
    }
    assert (
        smallest_reliable_fraction(curve) == 1.0
    )  # dips to 0.7 at 0.5, so only 1.0 holds
    curve[0.5]["power"] = 0.95
    assert smallest_reliable_fraction(curve) == 0.2
    assert (
        smallest_reliable_fraction({0.0: {"power": 0.0}, 1.0: {"power": 0.5}}) is None
    )


# ---- power on burst data ----

needs_burst = pytest.mark.skipif(
    not BURST_DB.exists(), reason="mpt_burst.db not available"
)
THRESHOLDS = None
N = 100


def trials(model, category, shift, fraction, n=N):
    global THRESHOLDS
    THRESHOLDS = THRESHOLDS or load_thresholds()
    return run_trials(
        model, category, burst_rows(model, category), shift, fraction, n, 0, THRESHOLDS
    )


@needs_burst
@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gpt-6-sol"])
def test_preamble_on_every_reply_alarms_and_moves_lines_and_word_len(model):
    r = trials(model, "all", "preamble", 1.0)
    assert r["power"] == 1.0
    assert r["by_feature"]["lines"] >= 0.95 and r["by_feature"]["word_len"] >= 0.95
    # the preamble has no markdown, so that feature must not move at all
    assert r["by_feature"].get("markdown", 0) == 0


@needs_burst
def test_bullets_on_every_reply_alarm_and_move_markdown_and_lines():
    r = trials("claude-sonnet-5-5", "all", "bullets", 1.0)
    assert r["power"] == 1.0
    assert r["by_feature"]["markdown"] >= 0.95 and r["by_feature"]["lines"] >= 0.95


@needs_burst
def test_bullets_are_still_caught_where_markdown_has_no_threshold():
    # GPT markdown never varies in the burst, so it is not checked (D22): the alarm
    # then has to come from the other features.
    r = trials("gpt-6-sol", "all", "bullets", 1.0)
    assert "markdown" not in r["by_feature"]
    assert r["power"] == 1.0 and r["by_feature"]["lines"] >= 0.95


@needs_burst
@pytest.mark.parametrize("shift", sorted(SHIFTS))
def test_no_injection_gives_few_false_alarms_even_on_the_lumpy_keys(shift):
    # Regression: threshold rounding used to turn ties at the limit into alarms
    # (11% on sonnet|structured). The calibrated rate is 1 in 500.
    for model, category in [
        ("claude-sonnet-5-5", "structured"),
        ("claude-haiku-5-5", "structured"),
        ("claude-sonnet-5-5", "all"),
    ]:
        assert trials(model, category, shift, 0.0, n=200)["power"] <= 0.02


@needs_burst
def test_power_falls_as_fewer_replies_are_changed():
    # haiku coding: bullets on 10% of replies are mostly missed, on 100% always caught
    low = trials("claude-haiku-5-5", "coding", "bullets", 0.1)["power"]
    mid = trials("claude-haiku-5-5", "coding", "bullets", 0.35)["power"]
    full = trials("claude-haiku-5-5", "coding", "bullets", 1.0)["power"]
    assert low < 0.2 < mid < 0.9 < full == 1.0
