import math

import pytest

from src.detectors.output_length import detect_windows, known_threshold

THRESHOLDS = {
    "length": {
        "approximate": ["m|rough"],
        "thresholds": {
            "m|math": 4.0,
            "m|all": 3.0,
            "m|rough": 5.0,
            "m|coding": None,
            "m|zero": 0.0,
            "m|negative": -1.0,
            "m|text": "4.0",
        },
    }
}


def window(mean=100, n_probes=20, repeats=12, scale=1.0, scale_probes=None):
    """Rows with a small fixed spread around `mean`; probes in scale_probes (or all) get `scale`."""
    rows = []
    for p in range(n_probes):
        factor = scale if scale_probes is None or p in scale_probes else 1.0
        for j in range(repeats):
            rows.append(
                {"probe_id": f"p{p}", "output_tokens": (mean + (j % 3) - 1) * factor}
            )
    return rows


def run(category, ref, cur, n_ref=4, n_cur=4, thresholds=THRESHOLDS):
    return detect_windows("m", category, ref, cur, n_ref, n_cur, thresholds)


def test_no_change_is_ok():
    r = run("math", window(), window())
    assert r.status == "ok" and r.effect_size == 0 and r.context.threshold == 4.0
    assert r.detector == "length" and r.metric_unit == "percent"


def test_big_shift_alarms_in_both_directions():
    longer = run("math", window(), window(scale=1.3))
    shorter = run("math", window(), window(scale=0.7))
    assert longer.status == "alarm" and longer.effect_size == pytest.approx(30.0)
    assert shorter.status == "alarm" and shorter.effect_size == pytest.approx(-30.0)
    assert longer.ci_low > 0 and longer.p_value < 0.05


def test_small_shift_within_threshold_is_ok():
    r = run("math", window(), window(scale=1.02))
    assert r.status == "ok" and r.effect_size == pytest.approx(2.0)


def test_concentrated_change_is_named_in_the_message():
    # one probe doubles: mean over 20 probes is +5%, beyond 4.0
    r = run("math", window(), window(scale=2.0, scale_probes={7}))
    assert r.status == "alarm" and r.effect_size == pytest.approx(5.0)
    assert "p7 +100.0%" in r.reason


def test_fewer_runs_widen_threshold():
    r = run("math", window(), window(), 2, 2)
    assert math.isclose(r.context.threshold, 4.0 * math.sqrt(2))
    assert run("math", window(), window(scale=1.06), 1, 1).status == "ok"  # 4*2 = 8
    assert run("math", window(), window(scale=1.06), 4, 4).status == "alarm"


# --- unknown threshold: do nothing, never alarm ---------------------------


@pytest.mark.parametrize(
    "category", ["coding", "nonexistent", "zero", "negative", "text"]
)
def test_unknown_threshold_never_alarms_even_on_a_huge_change(category):
    r = run(category, window(), window(scale=5.0))
    assert r.status == "no_threshold"
    assert r.context.threshold is None
    assert r.p_value is None and r.ci_low is None


def test_missing_length_section_is_no_threshold():
    r = run(
        "math", window(), window(scale=5.0), thresholds={"thresholds": {"m|math": 4}}
    )
    assert r.status == "no_threshold"
    assert (
        run("math", window(), window(scale=5.0), thresholds={}).status == "no_threshold"
    )


def test_known_threshold_rejects_bad_values():
    assert known_threshold(THRESHOLDS, "m|math") == 4.0
    for key in ("m|coding", "m|zero", "m|negative", "m|text", "m|missing"):
        assert known_threshold(THRESHOLDS, key) is None
    nan = {"length": {"thresholds": {"k": float("nan"), "j": float("inf"), "b": True}}}
    assert [known_threshold(nan, k) for k in "kjb"] == [None, None, None]


# --- no data -------------------------------------------------------------


def test_no_data():
    assert run("math", [], window()).status == "no_data"
    assert run("math", window(), window(), 0, 4).status == "no_data"


def test_all_tiny_means_or_missing_tokens_is_no_data_not_an_alarm():
    tiny = window(mean=2)  # reference mean under 5 tokens -> every probe skipped
    assert run("math", tiny, window(mean=50)).status == "no_data"
    none_rows = [dict(r, output_tokens=None) for r in window()]
    assert run("math", none_rows, window()).status == "no_data"


def test_probe_in_one_window_only_is_ignored():
    cur = window() + [{"probe_id": "extra", "output_tokens": 9999}] * 12
    assert run("math", window(), cur).effect_size == 0


# --- notes in the reason ---------------------------------------------------


def test_approximate_threshold_is_noted():
    assert "approximate" in run("rough", window(), window()).reason
    assert "approximate" not in run("math", window(), window()).reason


def test_skipped_probes_are_reported():
    ref = window() + [{"probe_id": "tiny", "output_tokens": 1}] * 12
    cur = window() + [{"probe_id": "tiny", "output_tokens": 50}] * 12
    r = run("math", ref, cur)
    assert r.status == "ok" and "1 probe(s) skipped" in r.reason
    assert "skipped" not in run("math", window(), window()).reason
