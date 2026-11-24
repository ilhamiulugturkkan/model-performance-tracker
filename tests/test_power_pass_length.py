"""Power test of the pass-rate and length detectors on burst splits (needs mpt_burst.db for the power part).

The injections are checked on made-up rows. The power tests run the real detect_windows on burst splits
with a known shift injected, seeded so they do not flake. Numbers in the assertions come from
`python -m experiments.power_pass_length` and leave a wide margin.
"""

import random

import pytest

from experiments.calibrate_length import BURST_DB, burst_rows
from experiments.power_pass_length import drop_pass_rate, lengthen, power
from src.detectors.pass_rate import load_thresholds

# ---- the injections (no database) ----


def length_rows():
    return [{"probe_id": p, "output_tokens": 100} for p in range(10) for _ in range(5)]


def test_lengthen_scales_exactly_the_chosen_share_of_probes():
    rows = length_rows()
    out = lengthen(rows, 20.0, 0.3, random.Random(0))
    grown = {r["probe_id"] for r in out if r["output_tokens"] == pytest.approx(120.0)}
    assert len(grown) == 3
    assert all(r["output_tokens"] in (100, pytest.approx(120.0)) for r in out)
    assert rows == length_rows()  # the input is not modified


def test_lengthen_leaves_missing_token_counts_alone_and_share_zero_changes_nothing():
    rows = [
        {"probe_id": 1, "output_tokens": None},
        {"probe_id": 2, "output_tokens": 50},
    ]
    assert lengthen(rows, 10.0, 1.0, random.Random(0))[0]["output_tokens"] is None
    assert lengthen(rows, 10.0, 0.0, random.Random(0)) == rows


def score_rows(rate, n=10000, probe=0):
    rng = random.Random(1)
    return [{"probe_id": probe, "score": float(rng.random() < rate)} for _ in range(n)]


def mean_score(rows, probe=None):
    s = [r["score"] for r in rows if probe is None or r["probe_id"] == probe]
    return sum(s) / len(s)


def test_drop_pass_rate_lowers_the_chosen_probes_by_about_the_drop_in_points():
    rows = score_rows(0.8)
    out = drop_pass_rate(rows, 5.0, 1.0, random.Random(0))
    assert mean_score(out) == pytest.approx(mean_score(rows) - 0.05, abs=0.01)
    assert all(
        o["score"] <= r["score"] for o, r in zip(out, rows)
    )  # scores only go down


def test_drop_pass_rate_only_touches_the_chosen_probes():
    rows = score_rows(0.8, 2000, probe=0) + score_rows(0.8, 2000, probe=1)
    out = drop_pass_rate(rows, 10.0, 0.5, random.Random(0))
    moved = [p for p in (0, 1) if mean_score(out, p) < mean_score(rows, p) - 0.03]
    assert len(moved) == 1
    assert drop_pass_rate(rows, 10.0, 0.0, random.Random(0)) == rows
    assert mean_score(drop_pass_rate(rows, 0.0, 1.0, random.Random(0))) == mean_score(
        rows
    )


def test_a_drop_larger_than_the_pass_rate_takes_it_to_zero_not_below():
    out = drop_pass_rate(score_rows(0.05), 8.0, 1.0, random.Random(0))
    assert mean_score(out) == 0.0


# ---- power on burst splits ----

needs_burst = pytest.mark.skipif(
    not BURST_DB.exists(), reason="mpt_burst.db not available"
)
N = 100


def trials(name, model, category, size, share, n=N):
    thresholds = load_thresholds()
    return power(
        name,
        model,
        category,
        burst_rows(model, category),
        size,
        share,
        n,
        0,
        thresholds,
    )


@needs_burst
def test_pass_rate_catches_a_big_drop_and_misses_a_small_or_narrow_one():
    key = ("pass_rate", "claude-sonnet-5-5", "all")
    assert trials(*key, 8.0, 1.0) >= 0.9
    assert trials(*key, 3.0, 1.0) < 0.5
    assert (
        trials(*key, 5.0, 0.1) <= 0.1
    )  # a 5-point drop on 10% of the probes is a 0.5 point change
    sizes = [trials(*key, d, 1.0) for d in (3.0, 5.0, 8.0)]
    assert sizes == sorted(sizes)
    shares = [trials(*key, 5.0, s) for s in (0.1, 0.25, 0.5, 1.0)]
    assert shares == sorted(shares)


@needs_burst
def test_length_catches_a_20_percent_change_and_a_5_percent_change_everywhere():
    key = ("length", "claude-sonnet-5-5", "all")
    assert trials(*key, 20.0, 1.0) == 1.0
    assert trials(*key, 5.0, 1.0) >= 0.6
    assert trials(*key, 20.0, 0.25) >= 0.6
    assert trials(*key, 5.0, 0.1) <= 0.1
    sizes = [trials(*key, c, 1.0) for c in (5.0, 10.0, 20.0)]
    assert sizes == sorted(sizes)


@needs_burst
def test_a_loose_limit_misses_what_a_tight_one_catches():
    # sonnet instructions length limit is about 22%, gpt-6-sol math is about 0.3%
    assert trials("length", "claude-sonnet-5-5", "instructions", 5.0, 1.0) <= 0.1
    assert trials("length", "gpt-6-sol", "math", 5.0, 0.1) >= 0.8


@needs_burst
@pytest.mark.parametrize(
    "name,model,category",
    [
        (
            "pass_rate",
            "claude-sonnet-5-5",
            "structured",
        ),  # was 47% before the tie tolerance
        ("pass_rate", "gpt-6-sol", "instructions"),
        ("pass_rate", "claude-sonnet-5-5", "all"),
        ("length", "claude-sonnet-5-5", "structured"),
        ("length", "gpt-6-sol", "math"),
        ("length", "claude-haiku-5-5", "all"),
    ],
)
def test_no_shift_gives_few_false_alarms(name, model, category):
    size = 5.0
    assert trials(name, model, category, size, 0.0, n=300) <= 0.02
