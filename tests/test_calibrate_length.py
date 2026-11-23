import json
import math
import random
from pathlib import Path

import pytest

from experiments.calibrate_length import (
    BURST_DB,
    THRESHOLDS_PATH,
    build_length_section,
    burst_rows,
    count_varying_probes,
    length_threshold,
    null_effects,
    split_burst,
    write_length_section,
)
from src.detectors.output_length import compare_window_output_lengths


def make_rows(counts):
    """counts: {probe_id: n_rows}; repeat_no is unique within a probe."""
    return [
        {"probe_id": pid, "repeat_no": i, "output_tokens": 100 + i}
        for pid, n in counts.items()
        for i in range(n)
    ]


def ids(rows):
    return [(r["probe_id"], r["repeat_no"]) for r in rows]


def test_twenty_repeats_split_ten_and_ten():
    a, b = split_burst(make_rows({"p0": 20, "p1": 20}), random.Random(0))
    for w in (a, b):
        assert sum(r["probe_id"] == "p0" for r in w) == 10
        assert sum(r["probe_id"] == "p1" for r in w) == 10


def test_fewer_rows_use_n_floor_half_and_drop_the_odd_one():
    a, b = split_burst(make_rows({"p0": 17, "p1": 20, "p2": 2}), random.Random(0))
    count = lambda w, p: sum(r["probe_id"] == p for r in w)
    assert (count(a, "p0"), count(b, "p0")) == (8, 8)  # one of 17 dropped
    assert (count(a, "p1"), count(b, "p1")) == (10, 10)
    assert (count(a, "p2"), count(b, "p2")) == (1, 1)
    assert len(set(ids(a)) | set(ids(b))) == len(a) + len(b)  # windows are disjoint


def test_probe_with_one_row_contributes_nothing():
    a, b = split_burst(make_rows({"p0": 1, "p1": 4}), random.Random(0))
    assert {r["probe_id"] for r in a + b} == {"p1"}


def test_split_is_reproducible_and_does_not_modify_input():
    rows = make_rows({"p0": 20, "p1": 20})
    before = ids(rows)
    first = split_burst(rows, random.Random(3))
    assert ids(rows) == before
    assert [ids(w) for w in first] == [
        ids(w) for w in split_burst(rows, random.Random(3))
    ]
    other = split_burst(rows, random.Random(4))
    assert ids(first[0]) != ids(other[0])


def test_split_windows_work_in_the_detector():
    a, b = split_burst(make_rows({"p0": 20, "p1": 20}), random.Random(0))
    r = compare_window_output_lengths(a, b)
    assert set(r["changes"]) == {"p0", "p1"}


@pytest.mark.skipif(not Path(BURST_DB).exists(), reason="mpt_burst.db not available")
def test_burst_rows_real_data_math_and_all():
    math_rows = burst_rows("claude-sonnet-5-5", "math")
    all_rows = burst_rows("claude-sonnet-5-5", "all")
    assert len(math_rows) == 36 * 20
    assert len(all_rows) > len(math_rows)  # "all" skips the category filter
    assert all("output_tokens" in r and "repeat_no" in r for r in all_rows)
    a, b = split_burst(math_rows, random.Random(0))
    assert len(a) == len(b) == 36 * 10


def test_missing_burst_db_raises_and_creates_no_file(tmp_path, monkeypatch):
    import experiments.calibrate_length as cal

    missing = tmp_path / "nope.db"
    monkeypatch.setattr(cal, "BURST_DB", missing)
    with pytest.raises(FileNotFoundError):
        cal.burst_rows("claude-sonnet-5-5", "math")
    assert not missing.exists()


# --- null_effects / length_threshold -------------------------------------


def varied_rows(n_probes=6, n=20):
    rng = random.Random(1)
    return [
        {"probe_id": f"p{i}", "repeat_no": j, "output_tokens": rng.randint(50, 150)}
        for i in range(n_probes)
        for j in range(n)
    ]


def test_null_effects_count_and_reproducibility():
    rows = varied_rows()
    e = null_effects(rows, 50, seed=0)
    assert len(e) == 50
    assert e == null_effects(rows, 50, seed=0)
    assert e != null_effects(rows, 50, seed=1)


def test_null_effects_skips_none_effects():
    # every probe under the tiny-mean cutoff -> every effect is None
    rows = [{"probe_id": "p0", "repeat_no": j, "output_tokens": 1} for j in range(20)]
    assert null_effects(rows, 10, seed=0) == []


def test_null_effects_are_centred_near_zero_relative_to_their_spread():
    # Not exactly 0: dividing by the reference-window mean gives a small positive
    # bias (about CV^2 of the window mean, Jensen's inequality). It stays small
    # next to the spread of the effects.
    e = null_effects(varied_rows(), 500, seed=0)
    mean = sum(e) / len(e)
    sd = (sum((x - mean) ** 2 for x in e) / len(e)) ** 0.5
    assert abs(mean) < 0.3 * sd
    assert 0.4 < sum(x > 0 for x in e) / len(e) < 0.75


def test_length_threshold_is_scaled_99_8th_percentile_of_abs():
    effects = [-(i / 1000) for i in range(1001)]  # |e| spans 0..1 evenly
    # 99.8th percentile of 0, .001, ..., 1.0 is 0.998
    assert length_threshold(effects) == pytest.approx(0.998 * math.sqrt(10 / 12))


def test_length_threshold_ignores_sign():
    assert length_threshold([-2.0, -1.0, 0.5]) == length_threshold([2.0, 1.0, -0.5])


def test_length_threshold_none_for_deterministic_or_empty():
    assert length_threshold([0.0, 0.0, 1e-12, -1e-12]) is None
    assert length_threshold([]) is None


def test_deterministic_probes_give_none_threshold():
    rows = [
        {"probe_id": f"p{i}", "repeat_no": j, "output_tokens": 100}
        for i in range(4)
        for j in range(20)
    ]
    assert length_threshold(null_effects(rows, 100, seed=0)) is None


@pytest.mark.skipif(not Path(BURST_DB).exists(), reason="mpt_burst.db not available")
def test_threshold_same_seed_twice_on_real_burst():
    rows = burst_rows("gpt-6.1-terra", "all")
    a = length_threshold(null_effects(rows, 200, seed=0))
    b = length_threshold(null_effects(rows, 200, seed=0))
    assert a == b and a is not None


# --- build_length_section / write_length_section -------------------------


SECTION = {"unit": "percent", "thresholds": {"m|all": 1.5, "m|math": None}}


def pass_rate_text():
    """The real file's text with any length key removed (the pass-rate part)."""
    data = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    data.pop("length", None)
    return json.dumps(data, indent=2) + "\n"


def test_write_leaves_pass_rate_part_byte_for_byte_unchanged(tmp_path):
    original = pass_rate_text()
    path = tmp_path / "thresholds.json"
    path.write_text(original, encoding="utf-8")

    prefix = original[: -len("\n}\n")]  # everything before the closing brace
    for section in (SECTION, {**SECTION, "seed": 1}):  # first write, then a rewrite
        write_length_section(section, path)
        text = path.read_text(encoding="utf-8")
        assert text.startswith(prefix + ',\n  "length": {')
        assert text.count('"length"') == 1  # replaced, not duplicated
        data = json.loads(text)
        assert data["length"] == section
        assert {k: v for k, v in data.items() if k != "length"} == json.loads(original)


def test_write_refuses_a_file_not_in_canonical_format(tmp_path):
    path = tmp_path / "thresholds.json"
    messy = '{"a":   1,\n "b": 2}'
    path.write_text(messy, encoding="utf-8")
    with pytest.raises(ValueError):
        write_length_section(SECTION, path)
    assert path.read_text(encoding="utf-8") == messy


@pytest.mark.skipif(not Path(BURST_DB).exists(), reason="mpt_burst.db not available")
def test_build_length_section_shape_on_real_burst():
    section = build_length_section(
        n_splits=20, seed=0, models=["gpt-6.1-terra"], categories=["math", "all"]
    )
    assert section["unit"] == "percent"
    assert section["repeats_per_probe"] == 12
    assert (section["n_splits"], section["seed"]) == (20, 0)
    assert set(section["thresholds"]) == {"gpt-6.1-terra|math", "gpt-6.1-terra|all"}
    assert section["generated"].endswith("Z") and section["commit"]


def test_count_varying_probes():
    rows = make_rows({"p0": 5, "p1": 5})  # lengths 100..104 -> both vary
    rows += [{"probe_id": "p2", "repeat_no": j, "output_tokens": 7} for j in range(5)]
    assert count_varying_probes(rows) == 2
    assert count_varying_probes([]) == 0


@pytest.mark.skipif(not Path(BURST_DB).exists(), reason="mpt_burst.db not available")
def test_approximate_lists_only_thresholds_resting_on_few_varying_probes():
    section = build_length_section(
        n_splits=50,
        seed=0,
        models=["gpt-6-sol", "gpt-6.1-terra"],
        categories=["math", "structured", "all"],
    )
    # sol math: 1 varying probe of 36 -> approximate; terra math/structured: null, not listed
    assert section["approximate"] == ["gpt-6-sol|math"]
    assert section["thresholds"]["gpt-6.1-terra|math"] is None
