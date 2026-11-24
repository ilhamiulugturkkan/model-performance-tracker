import pytest

from src.detectors.output_latency import (
    bootstrap_ci_latency,
    compare_window_latencies,
    detect_windows,
    known_threshold,
    window_latency,
)

THRESHOLDS = {
    "latency": {
        "approximate": ["m|rough"],
        "thresholds": {
            "m|math": 10.0,
            "m|all": 5.0,
            "m|rough": 8.0,
            "m|coding": None,
            "m|zero": 0.0,
            "m|negative": -1.0,
            "m|text": "4.0",
        },
    }
}


def window(
    median=1.0, n_probes=20, repeats=12, scale=1.0, scale_probes=None, spike=None
):
    """Rows with a small fixed spread around `median` seconds.

    Probes in scale_probes (or all) get `scale`. `spike`, if set, replaces one repeat of every
    probe with that latency (a slow outlier).
    """
    rows = []
    for p in range(n_probes):
        factor = scale if scale_probes is None or p in scale_probes else 1.0
        for j in range(repeats):
            latency = (median + 0.01 * ((j % 3) - 1)) * factor
            if spike is not None and j == 0:
                latency = spike
            rows.append({"probe_id": f"p{p}", "latency": latency})
    return rows


def run(category, ref, cur, n_ref=4, n_cur=4, thresholds=THRESHOLDS):
    return detect_windows("m", category, ref, cur, n_ref, n_cur, thresholds)


# ---- the statistic ----


def test_effect_is_the_mean_of_per_probe_percent_changes_in_the_median():
    ref = [{"probe_id": "a", "latency": v} for v in (1.0, 1.0, 1.0)] + [
        {"probe_id": "b", "latency": v} for v in (2.0, 2.0, 2.0)
    ]
    cur = [{"probe_id": "a", "latency": v} for v in (1.5, 1.5, 1.5)] + [
        {"probe_id": "b", "latency": v} for v in (2.0, 2.0, 2.0)
    ]
    r = compare_window_latencies(ref, cur)
    assert r["changes"] == {"a": pytest.approx(50.0), "b": pytest.approx(0.0)}
    assert r["effect"] == pytest.approx(25.0)
    assert (r["slower"], r["faster"], r["same"], r["skipped"]) == (1, 0, 1, 0)


def test_one_slow_call_does_not_move_the_median():
    # a 30 s spike in 1 of 12 repeats of every probe: the mean would move by ~200%, the median by 0
    r = compare_window_latencies(window(), window(spike=30.0))
    assert abs(r["effect"]) < 1.5


def test_null_latencies_are_skipped():
    cur = window()
    for row in cur[:6]:
        row["latency"] = None
    assert window_latency(cur)["p0"] == [row["latency"] for row in cur[6:12]]
    assert compare_window_latencies(window(), cur)["effect"] == pytest.approx(
        0.0, abs=2
    )


def test_a_probe_with_only_null_latencies_is_left_out():
    cur = window(n_probes=3)
    for row in cur:
        if row["probe_id"] == "p0":
            row["latency"] = None
    assert set(compare_window_latencies(window(n_probes=3), cur)["changes"]) == {
        "p1",
        "p2",
    }


def test_tiny_reference_medians_are_skipped_and_counted():
    ref = [{"probe_id": "tiny", "latency": 0.01}] * 3 + [
        {"probe_id": "ok", "latency": 1.0}
    ] * 3
    cur = [{"probe_id": "tiny", "latency": 0.05}] * 3 + [
        {"probe_id": "ok", "latency": 1.1}
    ] * 3
    r = compare_window_latencies(ref, cur)
    assert list(r["changes"]) == ["ok"] and r["skipped"] == 1
    assert r["effect"] == pytest.approx(10.0)


def test_no_usable_probes_gives_none():
    r = compare_window_latencies([], [])
    assert r["effect"] is None and r["median"] is None and r["top_changes"] == []


def test_top_changes_are_the_biggest_by_absolute_size():
    r = compare_window_latencies(window(), window(scale=1.2, scale_probes={3}), top_n=1)
    assert r["top_changes"][0][0] == "p3" and r["top_changes"][0][1] == pytest.approx(
        20.0, abs=1
    )


# ---- the detector ----


def test_no_change_is_ok():
    r = run("math", window(), window())
    assert r.status == "ok" and r.effect_size == pytest.approx(0)
    assert (
        r.detector == "latency"
        and r.metric_unit == "percent"
        and r.context.threshold == 10.0
    )


def test_slowdown_and_speedup_both_alarm():
    slower = run("math", window(), window(scale=1.3))
    faster = run("math", window(), window(scale=0.7))
    assert slower.status == "alarm" and slower.effect_size == pytest.approx(30.0)
    assert faster.status == "alarm" and faster.effect_size == pytest.approx(-30.0)
    assert slower.ci_low > 0 and slower.p_value < 0.05
    assert "slower" in slower.reason and "faster" in faster.reason


def test_the_alarm_says_it_is_not_proof_the_model_changed():
    r = run("math", window(), window(scale=1.3))
    assert "not that the model changed" in r.reason
    assert "not that the model changed" not in run("math", window(), window()).reason


def test_small_change_within_the_limit_is_ok():
    assert run("math", window(), window(scale=1.04)).status == "ok"


def test_threshold_scales_with_window_size():
    cur = window(scale=1.12)  # +12%
    assert run("math", window(), cur, n_ref=4, n_cur=4).status == "alarm"  # limit 10
    assert run("math", window(), cur, n_ref=1, n_cur=1).status == "ok"  # limit 20


def test_concentrated_change_is_named_in_the_message():
    r = run(
        "math", window(), window(scale=3.5, scale_probes={7})
    )  # +250% on one of 20 probes = +12.5% mean
    assert r.status == "alarm" and "p7" in r.reason


def test_approximate_threshold_is_noted():
    assert "approximate" in run("rough", window(), window()).reason
    assert "approximate" not in run("math", window(), window()).reason


def test_a_single_slow_spike_does_not_cause_an_alarm():
    assert run("math", window(), window(spike=30.0)).status == "ok"


# ---- no threshold / no data ----


def test_no_threshold_does_nothing_and_never_alarms():
    for cat in ("coding", "zero", "negative", "text", "never-seen"):
        r = run(cat, window(), window(scale=5.0))
        assert r.status == "no_threshold" and r.context.threshold is None, cat


def test_known_threshold_only_accepts_positive_finite_numbers():
    assert known_threshold(THRESHOLDS, "m|math") == 10.0
    for key in ("m|coding", "m|zero", "m|negative", "m|text", "m|missing"):
        assert known_threshold(THRESHOLDS, key) is None
    assert known_threshold({}, "m|math") is None
    assert (
        known_threshold({"latency": {"thresholds": {"k": float("inf")}}}, "k") is None
    )
    assert known_threshold({"latency": {"thresholds": {"k": True}}}, "k") is None


def test_no_threshold_wording_separates_null_keys_from_uncalibrated_keys():
    null_key = run("coding", window(), window())
    unseen = run("never-seen", window(), window())
    assert "never varied enough" in null_key.reason
    assert "never calibrated" not in null_key.reason
    assert "never calibrated" in unseen.reason


def test_no_data_cases():
    assert run("math", [], [], n_ref=0, n_cur=0).status == "no_data"
    assert run("math", window(), [], n_cur=1).status == "no_data"
    no_latency = [{"probe_id": "p", "latency": None}] * 3
    assert run("math", no_latency, no_latency).status == "no_data"
    assert (
        run("math", window(n_probes=2), [{"probe_id": "other", "latency": 1.0}]).status
        == "no_data"
    )


def test_only_common_probes_are_compared():
    cur = window(n_probes=5) + [
        {"probe_id": "extra", "latency": 9.0} for _ in range(12)
    ]
    assert run("math", window(n_probes=5), cur).status == "ok"


def test_bootstrap_interval_covers_a_known_shift_and_none_without_probes():
    low, high = bootstrap_ci_latency(window(), window(scale=1.2))
    assert low < 20.0 < high and 15 < low and high < 25
    assert bootstrap_ci_latency([], []) is None


# ---- day and night separately ----

import math

from src.detectors.output_latency import row_slot, split_by_slot

NIGHT = "2026-11-15T03:00:04+00:00"
DAY = "2026-11-15T15:00:04+00:00"


def slot_window(night_scale=1.0, day_scale=1.0, runs=(1, 2, 3, 4), n_probes=20):
    """Rows for consecutive runs: odd run ids are night (03:00), even are day (15:00), 3 repeats each."""
    rows = []
    for run in runs:
        night = run % 2 == 1
        for p in range(n_probes):
            for j in range(3):
                latency = (1.0 + 0.01 * (j - 1)) * (night_scale if night else day_scale)
                rows.append(
                    {
                        "probe_id": f"p{p}",
                        "run_id": run,
                        "time_stamp": NIGHT if night else DAY,
                        "latency": latency,
                    }
                )
    return rows


SLOT_THRESHOLDS = {
    "latency": {
        "thresholds": {
            "m|math": 10.0,
            "m|math|night": 10.0,
            "m|math|day": 10.0,
            "m|coding|night": 10.0,
        }
    }
}


def run_slot(ref, cur, slot, category="math", thresholds=SLOT_THRESHOLDS):
    return detect_windows("m", category, ref, cur, 4, 4, thresholds, slot=slot)


def test_row_slot_comes_from_the_utc_hour():
    assert row_slot({"time_stamp": "2026-11-15T03:00:04+00:00"}) == "night"
    assert row_slot({"time_stamp": "2026-11-15T08:59:59+00:00"}) == "night"
    assert row_slot({"time_stamp": "2026-11-15T09:00:00+00:00"}) == "day"
    assert row_slot({"time_stamp": "2026-11-15T15:00:04+00:00"}) == "day"
    parts = split_by_slot(slot_window())
    assert {r["run_id"] for r in parts["night"]} == {1, 3} and {
        r["run_id"] for r in parts["day"]
    } == {2, 4}


def test_a_change_in_one_slot_is_caught_by_that_slot_but_diluted_when_mixed():
    ref, cur = slot_window(), slot_window(day_scale=1.16)  # day +16%, night unchanged
    assert (
        run("math", ref, cur).status == "ok"
    )  # mixed: about +8%, under the limit of 10
    day, night = run_slot(ref, cur, "day"), run_slot(ref, cur, "night")
    assert day.status == "alarm" and day.effect_size == pytest.approx(16.0)
    assert night.status == "ok" and night.effect_size == pytest.approx(0.0)
    assert "[day]" in day.reason and "[night]" in night.reason


def test_the_daily_cycle_is_not_a_change_when_each_slot_is_compared_with_itself():
    # day runs are 25% slower than night runs in BOTH windows: nothing changed
    ref, cur = slot_window(day_scale=1.25), slot_window(day_scale=1.25)
    assert (
        run_slot(ref, cur, "night").status == run_slot(ref, cur, "day").status == "ok"
    )


def test_each_slot_uses_its_own_limit_and_a_missing_slot_key_is_no_threshold():
    thr = {"latency": {"thresholds": {"m|math|night": 50.0, "m|math|day": 5.0}}}
    ref, cur = slot_window(), slot_window(night_scale=1.2, day_scale=1.2)
    assert run_slot(ref, cur, "night", thresholds=thr).status == "ok"
    assert run_slot(ref, cur, "day", thresholds=thr).status == "alarm"
    r = run_slot(
        ref, cur, "night", category="coding"
    )  # "m|coding|night" exists: +20% > 10 x sqrt(2)
    assert r.status == "alarm"
    r = run_slot(ref, cur, "day", category="coding")
    assert (
        r.status == "no_threshold"
        and "m|coding|day" in r.reason
        and "[day]" in r.reason
    )


def test_slot_window_size_comes_from_the_runs_in_that_slot():
    # runs 1-4 hold 2 runs per slot: the limit is the 12-repeat limit x sqrt(2)
    r = run_slot(slot_window(), slot_window(), "night")
    assert r.context.threshold == pytest.approx(10.0 * math.sqrt(2))
    wide = slot_window(
        runs=tuple(range(1, 9))
    )  # 4 runs per slot: the 12-repeat limit itself
    assert run_slot(wide, wide, "night").context.threshold == pytest.approx(10.0)


def test_slot_with_no_rows_is_no_data_and_a_bad_slot_is_an_error():
    only_night = slot_window(runs=(1, 3))
    assert run_slot(only_night, only_night, "day").status == "no_data"
    with pytest.raises(ValueError):
        run_slot(slot_window(), slot_window(), "noon")


def test_slot_none_still_means_all_rows_together():
    assert run("math", slot_window(), slot_window()).context.threshold == 10.0
