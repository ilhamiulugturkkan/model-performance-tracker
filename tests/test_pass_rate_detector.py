import math
import random

from src.detectors.pass_rate import detect_windows, scaled_threshold

THRESHOLDS = {
    "approximate": ["m|rough"],
    "thresholds": {"m|math": 4.0, "m|all": 3.0, "m|rough": 5.0, "m|coding": None},
}


def window(rate, n_probes=20, repeats=12, seed=0):
    """Synthetic rows: each probe's pass fraction is `rate` (exact, no noise)."""
    rng = random.Random(seed)
    rows = []
    for p in range(n_probes):
        scores = [1.0] * round(rate * repeats) + [0.0] * (
            repeats - round(rate * repeats)
        )
        rng.shuffle(scores)
        rows += [{"probe_id": f"p{p}", "score": s} for s in scores]
    return rows


def run(category, ref, cur, n_ref=4, n_cur=4):
    return detect_windows("m", category, ref, cur, n_ref, n_cur, THRESHOLDS)


def test_scaled_threshold_matches_formula():
    for n in (1, 2, 4, 8):
        assert math.isclose(scaled_threshold(4.0, n, n), 4.0 * math.sqrt(12 / (3 * n)))


def test_no_change_is_ok():
    r = run("math", window(0.75), window(0.75))
    assert r.status == "ok" and r.effect_size == 0 and r.context.threshold == 4.0


def test_shift_beyond_threshold_alarms_both_directions():
    down = run("math", window(0.75), window(0.5))  # -25 points
    up = run("math", window(0.5), window(0.75))
    assert down.status == "alarm" and down.effect_size < 0 and up.status == "alarm"
    assert down.ci_high < 0 and down.p_value < 0.05


def test_injected_10_point_drop_alarms():
    # 2 of 20 probes go from always-pass to always-fail: exactly -10 points, threshold is 4.
    ref = window(1.0)
    cur = [
        dict(r, score=0.0) if r["probe_id"] in ("p0", "p1") else r for r in window(1.0)
    ]
    r = run("math", ref, cur)
    assert math.isclose(r.effect_size, -10.0)
    assert r.status == "alarm"
    assert r.ci_high < 0 and r.p_value < 0.05


def test_same_shift_alarms_with_4_runs_but_not_with_1_run():
    # One probe in 20 goes from always-pass to always-fail: -5 points.
    ref = window(1.0)
    cur = [dict(r, score=0.0) if r["probe_id"] == "p0" else r for r in window(1.0)]
    four = run("math", ref, cur, 4, 4)  # threshold 4.0
    one = run("math", ref, cur, 1, 1)  # threshold 4.0 * sqrt(4) = 8.0
    assert math.isclose(four.effect_size, -5.0) and four.status == "alarm"
    assert one.status == "ok"


def test_fewer_runs_widen_threshold():
    r2 = run("math", window(0.75), window(0.75), 2, 2)
    assert math.isclose(r2.context.threshold, 4.0 * math.sqrt(2))


def test_null_threshold_and_missing_key_are_no_threshold():
    assert run("coding", window(1.0), window(0.5)).status == "no_threshold"
    assert run("nonexistent", window(1.0), window(0.5)).status == "no_threshold"


def test_approximate_threshold_is_noted():
    assert "approximate" in run("rough", window(0.75), window(0.75)).reason
    assert "approximate" not in run("math", window(0.75), window(0.75)).reason


def test_no_data():
    assert run("math", [], window(0.5)).status == "no_data"
    assert run("math", window(0.5), window(0.5), 0, 4).status == "no_data"


def test_probe_in_one_window_only_is_ignored():
    ref = window(0.75)
    cur = window(0.75) + [{"probe_id": "extra", "score": 0.0}] * 12
    assert run("math", ref, cur).effect_size == 0
