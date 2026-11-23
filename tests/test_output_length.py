import pytest

from src.detectors.output_length import (
    _average_ranks,
    bootstrap_ci_length,
    compare_window_output_lengths,
    signed_rank_p_value,
)


def window(means):
    """One row per probe, output_tokens = that probe's mean."""
    return [{"probe_id": f"p{i}", "output_tokens": m} for i, m in enumerate(means)]


def test_two_probes_one_longer_one_unchanged():
    r = compare_window_output_lengths(window([10, 20]), window([12, 20]))
    assert r["changes"]["p0"] == pytest.approx(20.0)
    assert r["changes"]["p1"] == pytest.approx(0.0)
    assert r["effect"] == pytest.approx(10.0)  # mean
    assert r["median"] == pytest.approx(10.0)  # same as the mean here
    assert (r["longer"], r["shorter"], r["same"], r["skipped"]) == (1, 0, 1, 0)


def test_effect_is_mean_and_differs_from_median_when_one_probe_dominates():
    # 4 probes at 0%, 1 probe at +100% -> mean +20%, median 0%
    r = compare_window_output_lengths(
        window([100, 100, 100, 100, 100]), window([100, 100, 100, 100, 200])
    )
    assert r["effect"] == pytest.approx(20.0)
    assert r["median"] == pytest.approx(0.0)
    assert (r["longer"], r["shorter"], r["same"]) == (1, 0, 4)


def test_even_count_median_averages_the_two_middle_values():
    # changes: +10, +20, +30, +40 -> median 25 (upper-middle bug would give 30)
    r = compare_window_output_lengths(
        window([100, 100, 100, 100]), window([110, 120, 130, 140])
    )
    assert r["median"] == pytest.approx(25.0)
    assert r["effect"] == pytest.approx(25.0)
    assert (r["longer"], r["shorter"], r["same"]) == (4, 0, 0)


def test_odd_count_median_is_the_middle_value():
    r = compare_window_output_lengths(window([100, 100, 100]), window([110, 130, 150]))
    assert r["median"] == pytest.approx(30.0)


def test_top_changes_are_largest_by_absolute_size_with_probe_ids():
    # changes: p0 +10, p1 -50, p2 +30, p3 0, p4 +100
    r = compare_window_output_lengths(
        window([100] * 5), window([110, 50, 130, 100, 200])
    )
    assert [pid for pid, _ in r["top_changes"]] == ["p4", "p1", "p2"]
    assert r["top_changes"][0][1] == pytest.approx(100.0)
    assert r["top_changes"][1][1] == pytest.approx(-50.0)


def test_top_changes_respects_top_n_and_handles_fewer_probes():
    r = compare_window_output_lengths(window([10, 20]), window([12, 20]), top_n=5)
    assert [pid for pid, _ in r["top_changes"]] == ["p0", "p1"]
    r = compare_window_output_lengths(window([10, 20]), window([12, 20]), top_n=1)
    assert [pid for pid, _ in r["top_changes"]] == ["p0"]


def test_no_usable_probes_gives_none_effect_and_median():
    r = compare_window_output_lengths([], [])
    assert r["effect"] is None
    assert r["median"] is None
    assert r["top_changes"] == []
    assert (r["longer"], r["shorter"], r["same"], r["skipped"]) == (0, 0, 0, 0)


def test_tiny_and_zero_means_are_skipped_and_counted():
    r = compare_window_output_lengths(
        window([0, 2, 100]), window([5, 4, 110]), min_mean_tokens=5
    )
    assert list(r["changes"]) == ["p2"]
    assert r["effect"] == pytest.approx(10.0)
    assert r["skipped"] == 2


def test_none_token_counts_are_ignored():
    ref = window([100, None, 100])  # p1 has only a None row -> dropped
    ref.append({"probe_id": "p0", "output_tokens": None})
    r = compare_window_output_lengths(ref, window([110, 50, 100]))
    assert set(r["changes"]) == {"p0", "p2"}
    assert r["skipped"] == 0


# --- signed_rank_p_value -------------------------------------------------


def test_average_ranks_with_ties():
    # sorted: 1, 2, 2, 3 -> ranks 1, 2.5, 2.5, 4
    assert _average_ranks([2, 1, 3, 2]) == [2.5, 1.0, 4.0, 2.5]


def test_signed_rank_all_positive_matches_exact_value():
    # n=5 all positive: exact two-sided p = 2 / 2**5 = 0.0625
    p = signed_rank_p_value([1, 2, 3, 4, 5], n_perm=20000)
    assert p == pytest.approx(0.0625, abs=0.01)


def test_signed_rank_balanced_changes_give_p_of_one():
    # ranks 1.5,1.5,3.5,3.5 with matching signs -> observed sum 0 -> every sum counts
    assert signed_rank_p_value([1, -1, 2, -2], n_perm=500) == 1.0


def test_signed_rank_strong_consistent_shift_is_tiny_p():
    p = signed_rank_p_value([float(i) for i in range(1, 21)], n_perm=10000)
    assert p < 0.001


def test_signed_rank_sign_does_not_matter_for_two_sided():
    pos = signed_rank_p_value([1, 2, 3, 4, 5], seed=3)
    neg = signed_rank_p_value([-1, -2, -3, -4, -5], seed=3)
    assert pos == neg


def test_signed_rank_drops_zeros():
    assert signed_rank_p_value([0, 0.0, 1, 2, 3, 4], seed=7) == signed_rank_p_value(
        [1, 2, 3, 4], seed=7
    )


def test_signed_rank_same_seed_is_reproducible():
    c = [3.0, -1.0, 4.0, 1.5, -2.5, 9.0]
    assert signed_rank_p_value(c, n_perm=1000, seed=1) == signed_rank_p_value(
        c, n_perm=1000, seed=1
    )


def test_signed_rank_no_nonzero_changes_is_none():
    assert signed_rank_p_value([]) is None
    assert signed_rank_p_value([0, 0.0]) is None


# --- bootstrap_ci_length -------------------------------------------------


def rows(lens_by_probe):
    """Rows from {probe_id: [lens]} (a None entry is a missing token count)."""
    return [
        {"probe_id": pid, "output_tokens": n}
        for pid, lens in lens_by_probe.items()
        for n in lens
    ]


REF = {f"p{i}": [90 + 5 * j + i for j in range(6)] for i in range(5)}
CUR = {f"p{i}": [110 + 6 * j + 2 * i for j in range(6)] for i in range(5)}


def test_ci_contains_observed_effect_roughly_centred():
    ref, cur = rows(REF), rows(CUR)
    effect = compare_window_output_lengths(ref, cur)["effect"]
    low, high = bootstrap_ci_length(ref, cur)
    assert low <= effect <= high
    assert abs((low + high) / 2 - effect) < 0.25 * (high - low)


def test_ci_for_identical_windows_is_centred_near_zero():
    w = rows(REF)
    low, high = bootstrap_ci_length(w, w)
    assert low < 0 < high
    assert abs((low + high) / 2) < 0.25 * (high - low)


def test_ci_same_seed_same_interval_and_different_seed_differs():
    ref, cur = rows(REF), rows(CUR)
    assert bootstrap_ci_length(ref, cur, seed=5) == bootstrap_ci_length(
        ref, cur, seed=5
    )
    assert bootstrap_ci_length(ref, cur, seed=5) != bootstrap_ci_length(
        ref, cur, seed=6
    )


def test_ci_is_zero_width_when_every_repeat_is_identical():
    ref = rows({"a": [100] * 6, "b": [50] * 6})
    cur = rows({"a": [110] * 6, "b": [55] * 6})
    assert bootstrap_ci_length(ref, cur) == pytest.approx((10.0, 10.0))
    assert bootstrap_ci_length(ref, ref) == (0.0, 0.0)


def test_ci_applies_tiny_mean_and_none_rules():
    ref = rows({"a": [100, 100, None], "zero": [0, 0, 0], "tiny": [2, 2, 2]})
    cur = rows({"a": [110, 110, 110], "zero": [9, 9, 9], "tiny": [8, 8, 8]})
    assert bootstrap_ci_length(ref, cur) == pytest.approx((10.0, 10.0))


def test_ci_none_when_no_usable_probes():
    assert bootstrap_ci_length([], []) is None
    assert bootstrap_ci_length(rows({"a": [1, 1]}), rows({"a": [9, 9]})) is None
    assert bootstrap_ci_length(rows({"a": [100]}), rows({"b": [100]})) is None


def test_ci_widens_with_more_variable_replies():
    ref = rows({"a": [100] * 6})
    tight = rows({"a": [120, 121, 119, 120, 120, 121]})
    loose = rows({"a": [60, 180, 100, 140, 80, 160]})
    t_low, t_high = bootstrap_ci_length(ref, tight)
    l_low, l_high = bootstrap_ci_length(ref, loose)
    assert (l_high - l_low) > 10 * (t_high - t_low)
