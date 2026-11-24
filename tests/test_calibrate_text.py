import json

import pytest

from experiments.calibrate_length import BURST_DB, burst_rows
from experiments.calibrate_text import (
    build_text_section,
    count_varying_probes,
    text_thresholds,
    write_text_section,
)
from src.detectors.output_dist import FEATURES


def effects(**by_feature):
    """One effects dict per split; unnamed features are None (no data)."""
    n = len(next(iter(by_feature.values())))
    return [{f: by_feature.get(f, [None] * n)[i] for f in FEATURES} for i in range(n)]


def test_deterministic_key_is_none():
    assert text_thresholds(effects(lines=[0.0] * 50, markdown=[0.0] * 50)) is None
    assert text_thresholds([]) is None


def test_dead_feature_is_none_and_live_feature_has_a_threshold():
    t = text_thresholds(effects(lines=[-1.0, 1.0] * 50, markdown=[0.0] * 100))
    assert t["markdown"] is None and t["lines"] > 0
    assert all(t[f] is None for f in FEATURES if f not in ("lines",))


def test_thresholds_are_proportional_to_each_features_spread():
    # same shape, feature b is 10x wider -> 10x the threshold
    base = [-2.0, -1.0, 0.0, 1.0, 2.0] * 40
    t = text_thresholds(effects(lines=base, markdown=[10 * v for v in base]))
    assert t["markdown"] == pytest.approx(10 * t["lines"])


def test_family_wise_threshold_exceeds_the_single_feature_one():
    # Every |lines| effect is 1, so on its own its max ratio is exactly 1 and M = 1.
    # A second feature with a heavy tail (one split at ratio ~14) lifts M, so
    # lines' threshold must grow strictly. If the code ignored the second feature
    # the two would be equal and this would fail.
    one = [-1.0, 1.0] * 100
    alone = text_thresholds(effects(lines=one))["lines"]
    both = text_thresholds(effects(lines=one, markdown=[0.0] * 199 + [10.0]))["lines"]
    assert both > alone


def test_count_varying_probes():
    rows = [
        {"probe_id": 1, "raw_output": "a"},
        {"probe_id": 1, "raw_output": "b"},
        {"probe_id": 2, "raw_output": "a"},
        {"probe_id": 2, "raw_output": "a"},
    ]
    assert count_varying_probes(rows) == 1


def test_write_text_section_only_touches_the_text_key(tmp_path):
    path = tmp_path / "t.json"
    original = {"a": {"x": 1}, "length": {"y": 2}}
    path.write_text(json.dumps(original, indent=2) + "\n")
    write_text_section({"thresholds": {}}, path)
    data = json.loads(path.read_text())
    assert data["a"] == original["a"] and data["length"] == original["length"]
    assert data["text"] == {"thresholds": {}}
    path.write_text('{"a":1}')
    with pytest.raises(ValueError):
        write_text_section({}, path)


# ---- real data path (needs mpt_burst.db, git-ignored) ----

needs_burst = pytest.mark.skipif(
    not BURST_DB.exists(), reason="mpt_burst.db not available"
)


@needs_burst
def test_burst_rows_carry_reply_text():
    # If raw_output were missing, every reply would be skipped and every key would
    # silently calibrate to null ("deterministic everywhere").
    rows = burst_rows("claude-sonnet-5-5", "math")
    assert rows and all(isinstance(r["raw_output"], str) for r in rows)


@needs_burst
def test_small_calibration_on_burst_data_gives_live_thresholds():
    section = build_text_section(
        n_splits=30, models=["claude-sonnet-5-5"], categories=["math"]
    )
    entry = section["thresholds"]["claude-sonnet-5-5|math"]
    assert entry is not None
    assert all(entry[f] and entry[f] > 0 for f in ("lines", "digit_share", "word_len"))
