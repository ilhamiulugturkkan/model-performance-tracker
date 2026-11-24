"""Regression: a pass-rate effect exactly at the limit is not an alarm, even after threshold rounding."""

import pytest

from src.detectors.pass_rate import detect_windows


def window(n_fail, probes=10, repeats=10):
    """Rows with n_fail failed calls spread over the probes, the rest passing."""
    rows = [
        {"probe_id": p, "score": 1.0} for p in range(probes) for _ in range(repeats)
    ]
    for i in range(n_fail):
        rows[i * repeats]["score"] = 0.0  # one failure per probe, first probes first
    return rows


def run(n_fail):
    # stored limit 1.8257 points at 12 repeats; 10-repeat windows (10/3 runs) scale it to 1.99998
    thresholds = {"thresholds": {"m|structured": 1.8257}}
    return detect_windows(
        "m",
        "structured",
        window(0),
        window(n_fail),
        10 / 3,
        10 / 3,
        thresholds,
        n_perm=1,
        n_boot=1,
    )


def test_an_effect_of_exactly_the_limit_does_not_alarm():
    r = run(2)  # 2 of 100 calls fail: effect -2.0 points, limit 1.99998
    assert r.effect_size == pytest.approx(-2.0)
    assert r.context.threshold == pytest.approx(1.99998, abs=1e-4)
    assert r.status == "ok"


def test_a_clearly_larger_effect_still_alarms():
    assert run(3).status == "alarm"
