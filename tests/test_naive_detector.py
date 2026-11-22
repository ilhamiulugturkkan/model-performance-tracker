import inspect
import math

import pytest

from src.baselines.naive import detect_naive
from src.detectors.pass_rate import detect_windows


def window(pass_fractions, repeats=10):
    """Rows where probe i passes in pass_fractions[i] of its repeats (exact)."""
    rows = []
    for i, frac in enumerate(pass_fractions):
        n_pass = round(frac * repeats)
        rows += [{"probe_id": f"p{i}", "score": 1.0} for _ in range(n_pass)]
        rows += [{"probe_id": f"p{i}", "score": 0.0} for _ in range(repeats - n_pass)]
    return rows


def run(ref, cur, drop_points=5.0, drops_only=False):
    return detect_naive("m", "math", ref, cur, drop_points, drops_only)


def test_drop_points_has_no_default():
    sig = inspect.signature(detect_naive)
    assert sig.parameters["drop_points"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        detect_naive("m", "math", window([1.0]), window([1.0]))


def test_fields_have_no_statistics():
    r = run(window([1.0] * 20), window([1.0] * 19 + [0.0]))
    assert r.detector == "naive" and r.metric_unit == "points"
    assert r.p_value is None and r.ci_low is None and r.ci_high is None
    assert r.context.threshold == 5.0
    assert math.isclose(r.effect_size, -5.0) and r.statistic == r.effect_size


def test_no_change_is_ok():
    r = run(window([0.5] * 20), window([0.5] * 20))
    assert r.status == "ok" and r.effect_size == 0


def test_big_drop_alarms_with_reason():
    r = run(window([1.0] * 20), window([1.0] * 18 + [0.0] * 2))  # -10 points
    assert r.status == "alarm"
    assert r.reason == "Pass rate fell 10.0 points, at least the fixed 5.0."


def test_small_drop_is_ok():
    r = run(window([1.0] * 20), window([1.0] * 19 + [0.9]))  # -0.5 points
    assert r.status == "ok" and "less than the fixed 5.0" in r.reason


def test_boundary_exact_drop_alarms_both_modes():
    ref, cur = window([1.0] * 20), window([1.0] * 19 + [0.0])  # exactly -5 points
    assert run(ref, cur, 5.0, drops_only=False).status == "alarm"
    assert run(ref, cur, 5.0, drops_only=True).status == "alarm"
    assert run(ref, cur, 5.01).status == "ok"


def test_rise_two_sided_alarms_but_drops_only_does_not():
    ref, cur = window([0.5] * 10 + [0.0] * 10), window([0.5] * 10 + [1.0] * 10)  # +25
    two_sided = run(ref, cur, drops_only=False)
    drops = run(ref, cur, drops_only=True)
    assert two_sided.status == "alarm" and "rose" in two_sided.reason
    assert drops.status == "ok" and drops.effect_size > 0


def test_drop_alarms_in_drops_only_mode():
    r = run(window([1.0] * 20), window([0.0] * 20), drops_only=True)
    assert r.status == "alarm" and r.effect_size == -100


def test_empty_window_is_no_data():
    assert run([], window([1.0])).status == "no_data"
    assert run(window([1.0]), []).status == "no_data"
    assert run([], []).status == "no_data"


def test_no_shared_probes_is_no_data():
    ref = [{"probe_id": "a", "score": 1.0}]
    cur = [{"probe_id": "b", "score": 0.0}]
    assert run(ref, cur).status == "no_data"


def test_only_common_probes_are_compared():
    # Probe "extra" exists only in the current window and fails; it must be ignored.
    ref = window([1.0] * 5)
    cur = window([1.0] * 5) + [{"probe_id": "extra", "score": 0.0}] * 10
    assert run(ref, cur).effect_size == 0


def test_same_effect_as_real_detector():
    ref = window([1.0] * 20)
    cur = window([1.0] * 17 + [0.0] * 3)
    thresholds = {"thresholds": {"m|math": 4.0}}
    real = detect_windows("m", "math", ref, cur, 4, 4, thresholds)
    assert math.isclose(run(ref, cur).effect_size, real.effect_size)


def test_never_returns_not_enough_power_or_no_threshold():
    cases = [(window([1.0]), window([0.0])), ([], []), (window([1.0]), window([1.0]))]
    for ref, cur in cases:
        for drops_only in (True, False):
            assert run(ref, cur, 5.0, drops_only).status in {"ok", "alarm", "no_data"}


@pytest.mark.parametrize("bad", [0, -1.0])
def test_drop_points_must_be_positive(bad):
    with pytest.raises(ValueError):
        run(window([1.0]), window([1.0]), drop_points=bad)
