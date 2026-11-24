import pytest

from src.detectors.output_dist import (
    FEATURES,
    compare_window_features,
    detect_windows,
    extract_features,
    known_thresholds,
    window_features,
)

# ---- extract_features ----


def test_bare_answer():
    f = extract_features("42")
    assert f["lines"] == 1 and f["markdown"] == 0
    assert f["digit_share"] == 1.0 and f["word_len"] == 2.0
    assert f["short_start"] == 1.0 and f["distinct"] == 1.0


def test_chatty_markdown_reply():
    text = "Sure! Here's how you do it, step by step, with a long first line here:\n\n## Steps\n- **one** thing\n* two\n```py\nx = 1\n```\n"
    f = extract_features(text)
    # 1 header + 1 bold + 2 bullets + 2 fence lines
    assert f["markdown"] == 6
    assert f["lines"] == 8  # trailing newline is not a line
    assert f["short_start"] == 0.0  # first line is over 60 chars


def test_bold_is_not_counted_as_a_bullet():
    assert extract_features("**Answer** is 5")["markdown"] == 1
    assert extract_features("* item\n- item")["markdown"] == 2


def test_numbered_list_items_count_as_markers():
    assert extract_features("1. a\n2) b\n  10. c")["markdown"] == 3


def test_numbers_that_are_not_list_items_are_not_markers():
    for text in ("3.14 is pi", "1.5 apples", "2024. Then", "see item 1. next", "42"):
        assert extract_features(text)["markdown"] == 0, text


def test_hash_without_space_is_not_a_header():
    assert extract_features("#hashtag and C# code")["markdown"] == 0
    assert extract_features("### Title")["markdown"] == 1


def test_digit_share_and_word_stats():
    f = extract_features("ab ab cd1")
    assert f["digit_share"] == pytest.approx(1 / 9)
    assert f["word_len"] == pytest.approx((2 + 2 + 3) / 3)
    assert f["distinct"] == pytest.approx(2 / 3)


def test_distinct_ignores_case_and_apostrophes_stay_in_words():
    f = extract_features("The the don't")
    assert f["distinct"] == pytest.approx(2 / 3)
    assert f["word_len"] == pytest.approx((3 + 3 + 5) / 3)


def test_short_start_uses_first_non_blank_line_and_the_60_char_boundary():
    assert extract_features("\n\n" + "a" * 59)["short_start"] == 1.0
    assert extract_features("a" * 60)["short_start"] == 0.0


def test_empty_reply_has_defined_counts_and_undefined_text_features():
    f = extract_features("  \n ")
    assert f["lines"] == 0 and f["markdown"] == 0 and f["digit_share"] == 0
    assert f["word_len"] is None and f["short_start"] is None and f["distinct"] is None


def test_non_word_reply_has_no_words():
    f = extract_features("!!!")
    assert f["word_len"] is None and f["distinct"] is None and f["lines"] == 1


# ---- window_features / compare_window_features ----


def rows(texts_by_probe):
    return [
        {"probe_id": p, "raw_output": t}
        for p, texts in texts_by_probe.items()
        for t in texts
    ]


def test_window_features_skips_missing_output_and_undefined_values():
    feats = window_features(rows({"p": ["a b", None, ""]}))
    assert feats["p"]["lines"] == [1.0, 0.0]  # the empty string is a reply
    assert feats["p"]["word_len"] == [1.0]  # undefined for the empty reply


def test_no_change_gives_zero_effects():
    w = rows({"p0": ["hello world", "hi there"], "p1": ["x\ny", "z"]})
    r = compare_window_features(w, w)
    assert all(r["effects"][f] == pytest.approx(0) for f in FEATURES)
    assert all(n == 2 for n in r["n_probes"].values())


def test_effect_is_mean_over_probes_of_the_change_in_probe_means():
    ref = rows({"p0": ["a", "a"], "p1": ["a", "a"]})
    cur = rows({"p0": ["a\nb\nc", "a\nb\nc"], "p1": ["a", "a"]})
    r = compare_window_features(ref, cur)
    assert r["effects"]["lines"] == pytest.approx(1.0)  # (+2 + 0) / 2
    assert r["top"]["lines"][0] == ("p0", pytest.approx(2.0))


def test_probes_in_one_window_only_are_ignored():
    r = compare_window_features(rows({"a": ["x"]}), rows({"b": ["x\ny"]}))
    assert r["effects"]["lines"] is None and r["n_probes"]["lines"] == 0


def test_feature_undefined_in_one_window_drops_that_probe_for_that_feature_only():
    r = compare_window_features(rows({"p": ["words here"]}), rows({"p": [""]}))
    assert r["effects"]["word_len"] is None
    assert r["effects"]["lines"] == pytest.approx(-1.0)


# ---- known_thresholds ----

T = {
    "lines": 1.0,
    "markdown": 0.5,
    "digit_share": 0.01,
    "word_len": 0.1,
    "short_start": 0.1,
    "distinct": 0.02,
}
THRESHOLDS = {
    "text": {
        "thresholds": {
            "m|math": T,
            "m|coding": None,
            "m|partial": {**T, "markdown": None, "lines": 0, "word_len": "x"},
            "m|junk": "oops",
        }
    }
}


def test_known_thresholds_keeps_only_positive_finite_numbers():
    assert known_thresholds(THRESHOLDS, "m|math") == T
    assert known_thresholds(THRESHOLDS, "m|partial") == {
        k: v for k, v in T.items() if k in ("digit_share", "short_start", "distinct")
    }
    for key in ("m|coding", "m|junk", "m|missing"):
        assert known_thresholds(THRESHOLDS, key) == {}
    assert known_thresholds({}, "m|math") == {}
    assert known_thresholds({"text": {"thresholds": {"k": {"lines": True}}}}, "k") == {}
    assert (
        known_thresholds({"text": {"thresholds": {"k": {"lines": float("nan")}}}}, "k")
        == {}
    )


# ---- detect_windows ----


def window(text="Answer: 12\nbecause of it", n_probes=10, repeats=12):
    return [
        {"probe_id": f"p{p}", "raw_output": text}
        for p in range(n_probes)
        for _ in range(repeats)
    ]


LINES_ONLY = {
    "text": {"thresholds": {"m|math": {**{f: 1000.0 for f in FEATURES}, "lines": 1.0}}}
}


def run(ref, cur, category="math", n_ref=4, n_cur=4, thresholds=THRESHOLDS):
    return detect_windows("m", category, ref, cur, n_ref, n_cur, thresholds)


def test_no_change_is_ok():
    r = run(window(), window())
    assert r.status == "ok" and r.detector == "text" and r.metric_unit == "score"
    assert r.statistic == 0 and r.context.threshold == 1.0


def test_more_lines_alarms_and_names_the_feature():
    r = run(
        window(),
        window("Answer: 12\nbecause of it\nand\nmore\nlines"),
        thresholds=LINES_ONLY,
    )
    assert r.status == "alarm" and r.statistic == pytest.approx(
        3.0
    )  # +3 lines vs limit 1.0
    assert "line count +3 (limit 1)" in r.reason


def test_markdown_style_change_alarms_in_both_directions():
    plain = window("Answer: 12")
    md = window("## Answer\n- **12**")
    assert run(plain, md).status == "alarm"
    assert run(md, plain).status == "alarm"


def test_change_inside_threshold_is_ok_and_names_the_closest_feature():
    r = run(
        window(), window("Answer: 12\nbecause of it\nplus"), thresholds=LINES_ONLY
    )  # +1 line, limit 1.0: not beyond
    assert r.status == "ok" and r.statistic == pytest.approx(1.0)
    assert "Closest: line count" in r.reason


def test_threshold_scales_with_window_size():
    cur = window("Answer: 12\nbecause of it\nplus")  # +1 line
    kw = {"thresholds": LINES_ONLY}
    assert run(window(), cur, n_ref=4, n_cur=4, **kw).status == "ok"
    assert run(window(), cur, n_ref=1, n_cur=1, **kw).status == "ok"  # limit 2.0
    assert run(window(), cur, n_ref=16, n_cur=16, **kw).status == "alarm"  # limit 0.5


def test_unchecked_features_are_listed_and_never_alarm():
    thr = {"text": {"thresholds": {"m|math": {"lines": 1.0}}}}
    r = run(window(), window("## A\n- **b**\n- **c**"), thresholds=thr)
    assert r.status == "ok" and r.statistic == pytest.approx(
        1.0
    )  # +1 line = limit, not beyond; markdown moved but is not checked
    assert "Not checked (no threshold): markdown markers" in r.reason


def test_no_threshold_does_nothing():
    for cat in ("coding", "never-seen"):
        r = run(window(), window("a\nb\nc\nd\ne\nf"), category=cat)
        assert r.status == "no_threshold" and r.statistic is None


def test_no_threshold_wording_separates_null_keys_from_uncalibrated_keys():
    cur = window("a\nb\nc\nd\ne\nf")
    null_key = run(window(), cur, category="coding")  # in the file, value null
    unseen = run(window(), cur, category="never-seen")  # not in the file
    assert null_key.status == unseen.status == "no_threshold"
    assert "cannot be determined" in null_key.reason
    assert "never calibrated" not in null_key.reason
    assert "never calibrated" in unseen.reason
    assert "cannot be determined" not in unseen.reason


def test_no_data_cases():
    assert run([], [], n_ref=0, n_cur=0).status == "no_data"
    assert run(window(), [], n_cur=1).status == "no_data"
    no_overlap = [{"probe_id": "other", "raw_output": "x"}]
    assert run(window(n_probes=2), no_overlap).status == "no_data"
    no_text = [{"probe_id": "p", "raw_output": None}] * 3
    assert run(no_text, no_text).status == "no_data"


def test_only_common_probes_are_compared():
    ref = window(n_probes=5)
    cur = window(n_probes=5) + [
        {"probe_id": "extra", "raw_output": "1\n2\n3\n4\n5\n6\n7"} for _ in range(12)
    ]
    assert run(ref, cur).status == "ok"


def test_reason_shows_shares_in_points_with_two_decimals():
    ref = window("Answer 12")  # 2 digits of 9 characters
    cur = window("Answer 1234")  # 4 digits of 11 characters
    thr = {"text": {"thresholds": {"m|math": {"digit_share": 0.001}}}}
    r = run(ref, cur, thresholds=thr)
    assert r.status == "alarm"
    assert "digit share +14.14 points (limit 0.10 points)" in r.reason


def test_small_limits_do_not_round_to_zero_in_the_reason():
    thr = {"text": {"thresholds": {"m|math": {"distinct": 0.0012}}}}
    r = run(window("a b c d"), window("a a a a"), thresholds=thr)
    assert "(limit 0.0012)" in r.reason


def test_a_tie_at_the_limit_is_not_an_alarm_even_after_threshold_rounding():
    # +1 line against a limit stored as 0.99999 (6-digit rounding of 1.0): ratio 1.00001
    thr = {"text": {"thresholds": {"m|math": {"lines": 0.99999}}}}
    cur = window("Answer: 12\nbecause of it\nplus")
    assert run(window(), cur, thresholds=thr).status == "ok"
    thr = {"text": {"thresholds": {"m|math": {"lines": 0.99}}}}  # a real 1% excess
    assert run(window(), cur, thresholds=thr).status == "alarm"
