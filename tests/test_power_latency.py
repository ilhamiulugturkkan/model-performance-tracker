"""Power test of the latency detector on burst splits, at the BURST limits (needs mpt_burst.db)."""

import random

import pytest

from experiments.calibrate_length import BURST_DB, burst_rows, split_burst
from experiments.power_latency import (
    WINDOW_RUNS,
    alarms_fast,
    burst_thresholds,
    power,
    slow_down,
)
from src.detectors.output_latency import detect_windows

needs_burst = pytest.mark.skipif(
    not BURST_DB.exists(), reason="mpt_burst.db not available"
)
KEY = ("claude-sonnet-5-5", "all")


def test_slow_down_scales_exactly_the_chosen_share_of_probes():
    rows = [{"probe_id": p, "latency": 1.0} for p in range(10) for _ in range(3)]
    out = slow_down(rows, 0.5, 0.3, random.Random(0))
    assert len({r["probe_id"] for r in out if r["latency"] == 1.5}) == 3
    assert all(r["latency"] in (1.0, 1.5) for r in out) and rows[0]["latency"] == 1.0


@needs_burst
def test_fast_alarm_rule_matches_detect_windows():
    th, rows, rng = burst_thresholds(), burst_rows(*KEY), random.Random(3)
    for shift in (0.0, 0.01, 0.02, 0.03, 0.05, 0.1):
        a, b = split_burst(rows, rng)
        b = slow_down(b, shift, 1.0, rng)
        slow = (
            detect_windows(*KEY, a, b, WINDOW_RUNS, WINDOW_RUNS, th).status == "alarm"
        )
        assert alarms_fast(*KEY, a, b, th) == slow


@needs_burst
def test_power_rises_with_the_size_of_the_slowdown_and_false_alarms_are_rare():
    th, rows = burst_thresholds(), burst_rows(*KEY)
    curve = [
        power(*KEY, rows, s, 1.0, 100, 0, th) for s in (0.0, 0.01, 0.02, 0.05, 0.2)
    ]
    assert curve[0] <= 0.02  # nothing changed: about 1 in 500
    assert curve == sorted(curve)  # more slowdown, more alarms
    assert (
        curve[-1] == 1.0 and curve[2] > 0.3
    )  # +20% always caught; +2% often caught here


@needs_burst
def test_a_slowdown_on_a_quarter_of_the_probes_is_harder_to_catch_than_on_all():
    th, rows = burst_thresholds(), burst_rows(*KEY)
    everywhere = power(*KEY, rows, 0.05, 1.0, 100, 0, th)
    some = power(*KEY, rows, 0.05, 0.25, 100, 0, th)
    assert some < everywhere


# ---- power at the FINAL limits, on the real windows (needs mpt.db) ----

from pathlib import Path

from experiments.power_latency import SLOWDOWNS, real_power, smallest_caught
from src.db.connection import DB_PATH, connect
from src.detectors.pass_rate import load_thresholds

needs_real = pytest.mark.skipif(
    not Path(DB_PATH).exists(), reason="mpt.db not available"
)


def test_smallest_caught_needs_the_share_to_stay_up():
    assert (
        smallest_caught({0.0: 0.0, 0.1: 0.9, 0.2: 0.5, 0.5: 1.0}) == 0.5
    )  # dips at 0.2
    assert smallest_caught({0.0: 0.0, 0.1: 0.9, 0.2: 1.0}) == 0.1
    assert smallest_caught({0.0: 0.0, 0.1: 0.2}) is None


@needs_real
def test_final_limits_never_alarm_on_unchanged_real_windows_and_catch_a_big_slowdown():
    th, conn = load_thresholds(), connect()
    checked = 0
    for model in ("claude-haiku-5-5", "gpt-6-sol"):
        for slot in (None, "night", "day"):
            curve = real_power(model, "all", th, conn, slot)
            assert curve is not None
            checked += 1
            assert (
                curve[0.0] == 0.0
            )  # no slowdown: no alarms (the limits are fitted on these windows)
            shares = [curve[s] for s in sorted(curve)]
            assert shares == sorted(shares)  # a bigger slowdown never alarms less
            assert curve[max(SLOWDOWNS)] == 1.0  # +150% is always caught
    assert checked == 6


@needs_real
def test_a_null_limit_has_no_power_curve():
    assert (
        real_power("gpt-6.1-terra", "all", load_thresholds(), connect(), "day") is None
    )
