import pytest

from experiments.calibrate_latency import MAX_USABLE_LIMIT, build_latency_section
from src.detectors.output_latency import detect_windows, known_threshold
from src.detectors.pass_rate import load_thresholds


def entry(threshold, varying=10):
    return {"threshold": threshold, "null_sd": 1.0, "varying": varying}


def factor(applied):
    return {"applied": applied, "estimated": applied, "approximate": True}


def section(burst, factors=None, slot_factors=None):
    return build_latency_section(
        burst, factors or {}, n_splits=1, slot_factors=slot_factors
    )


def test_limit_is_burst_threshold_times_the_applied_factor():
    s = section({"m|a": entry(5.0)}, {"m|a": factor(2.0)})
    assert s["thresholds"]["m|a"] == pytest.approx(10.0)
    assert s["burst_thresholds"]["m|a"] == pytest.approx(5.0)
    assert s["unusable"] == {}


def test_a_limit_above_the_cutoff_becomes_null_and_is_listed_with_its_value():
    s = section({"m|b": entry(10.0)}, {"m|b": factor(15.0)})
    assert s["thresholds"]["m|b"] is None
    assert s["unusable"] == {"m|b": pytest.approx(150.0)}
    assert "m|b" not in s["approximate"]
    assert s["burst_thresholds"]["m|b"] == pytest.approx(
        10.0
    )  # the burst value is still recorded


def test_a_limit_exactly_at_the_cutoff_is_still_usable():
    s = section({"m|c": entry(MAX_USABLE_LIMIT / 4)}, {"m|c": factor(4.0)})
    assert s["thresholds"]["m|c"] == pytest.approx(MAX_USABLE_LIMIT)
    assert s["unusable"] == {}


def test_a_key_with_no_burst_threshold_is_null_but_not_listed_as_unusable():
    s = section({"m|d": entry(None)})
    assert s["thresholds"]["m|d"] is None and s["unusable"] == {}


def test_slot_keys_get_the_burst_threshold_times_their_own_factor_and_the_same_cutoff():
    s = section(
        {"m|a": entry(5.0)},
        {"m|a": factor(2.0)},
        {"m|a|night": factor(1.5), "m|a|day": factor(30.0)},
    )
    assert s["thresholds"]["m|a|night"] == pytest.approx(7.5)
    assert s["thresholds"]["m|a|day"] is None and s["unusable"][
        "m|a|day"
    ] == pytest.approx(150.0)
    assert "m|a|night" in s["approximate"] and "m|a|day" not in s["approximate"]


def test_a_null_limit_gives_cannot_be_determined_not_a_huge_limit():
    s = section({"m|a": entry(5.0)}, {"m|a": factor(40.0)})  # 200%: unusable
    window = [
        {"probe_id": f"p{p}", "latency": 1.0} for p in range(5) for _ in range(12)
    ]
    r = detect_windows("m", "a", window, window, 4, 4, {"latency": s})
    assert r.status == "no_threshold" and r.context.threshold is None
    assert "cannot be determined" in r.reason and "never calibrated" not in r.reason


def test_the_shipped_file_has_no_limit_above_the_cutoff_and_terra_day_is_null():
    section_ = load_thresholds()["latency"]
    assert section_["max_usable_limit"] == MAX_USABLE_LIMIT
    usable = [v for v in section_["thresholds"].values() if v is not None]
    assert usable and max(usable) <= MAX_USABLE_LIMIT
    for category in ("coding", "instructions", "math", "structured", "all"):
        key = f"gpt-6.1-terra|{category}|day"
        assert known_threshold(load_thresholds(), key) is None
        assert key in section_["unusable"]
    # every nulled-as-unusable key is null in the file and was above the cutoff
    assert section_["unusable"]
    for key, would_be in section_["unusable"].items():
        assert section_["thresholds"][key] is None and would_be > MAX_USABLE_LIMIT
